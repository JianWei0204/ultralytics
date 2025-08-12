# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy
import numpy as np
import os
import time

from ultralytics.models import yolo
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils import LOGGER, RANK
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer

from trans_discriminator import TransformerDiscriminator
from feature_extractor import FeatureExtractor


class DomainAdaptTrainer(DetectionTrainer):
    """
    Domain Adaptation Trainer for YOLOv8 object detection.

    This trainer extends the standard YOLOv8 DetectionTrainer to support
    unsupervised domain adaptation using a transformer-based discriminator.
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        # Additional parameters for domain adaptation
        self.source_domain = self.args.source_domain
        self.target_domain = self.args.target_domain

        # Source and target domain labels
        self.source_label = 0
        self.target_label = 1

        # Create discriminator
        self.discriminator = None
        self.optimizer_D = None
        self.feature_extractor = None

    def get_dataset(self, data):
        """Override to get both source and target datasets."""
        # Handle the default case
        if hasattr(self, 'source_domain') and self.source_domain:
            # Return source and target domains
            return data[self.source_domain], data[self.target_domain]
        else:
            # Regular behavior for non-domain adaptation
            return super().get_dataset(data)

    def _setup_train(self, world_size):
        """Set up training with domain adaptation components."""
        # Initialize standard training setup
        super()._setup_train(world_size)

        # Create feature extractor to access bridge layer outputs
        self.feature_extractor = FeatureExtractor(self.model, 'Adjust_Transformer')

        # Create source and target dataloaders
        self.source_loader = self.train_loader  # Default train_loader is for source domain

        # Create target domain dataloader
        if self.target_domain:
            batch_size = self.batch_size // max(world_size, 1)
            self.target_loader = self.get_dataloader(
                self.testset, batch_size=batch_size, rank=RANK, mode="train"
            )

        # Initialize discriminator and its optimizer
        self.setup_discriminator()

    def setup_discriminator(self):
        """Initialize the domain discriminator and its optimizer."""
        # Get feature dimensions from the bridge layer (adjust as needed)
        feature_channels = 128  # From the Adjust_Transformer layer in yaml

        # Create discriminator
        self.discriminator = TransformerDiscriminator(channels=feature_channels)
        self.discriminator.train()
        self.discriminator.cuda()

        if RANK != -1:
            self.discriminator = nn.parallel.DistributedDataParallel(
                self.discriminator, device_ids=[RANK]
            )

        # Create optimizer for discriminator
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.args.lr0_d,  # You need to add this to your args
            betas=(0.9, 0.99)
        )
        self.optimizer_D.zero_grad()

    def weightedMSE(self, D_out, label):
        """Compute weighted MSE loss for discriminator."""
        return torch.mean((D_out - label.cuda()).abs() ** 2)

    def adjust_learning_rate_D(self, epoch):
        """Polynomial learning rate decay for discriminator."""

        def lr_poly(base_lr, iter, max_iter, power):
            return base_lr * ((1 - float(iter) / max_iter) ** power)

        lr = lr_poly(self.args.lr0_d, epoch, self.args.epochs, 0.8)
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        return lr

    def _do_train(self, world_size=1):
        """Modified training loop to include domain adaptation."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")

        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.source_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {self.save_dir}\n"
                    f'Starting domain adaptation training for {self.epochs} epochs...')

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            self.model.train()

            # Adjust learning rate for discriminator
            cur_lr_d = self.adjust_learning_rate_D(epoch)

            if RANK != -1:
                self.source_loader.sampler.set_epoch(epoch)
                if hasattr(self, 'target_loader'):
                    self.target_loader.sampler.set_epoch(epoch)

            # Create iterators for source and target
            if hasattr(self, 'target_loader'):
                target_iter = iter(self.target_loader)
            source_iter = iter(self.source_loader)

            nb = len(self.source_loader)
            self.tloss = None

            # Progress bar
            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = self.progress_bar(nb)
            else:
                pbar = range(nb)

            self.optimizer.zero_grad()
            self.optimizer_D.zero_grad()

            for i in pbar:
                self.run_callbacks("on_train_batch_start")

                # Process a batch from target domain (for domain adaptation)
                if hasattr(self, 'target_loader'):
                    try:
                        target_batch = next(target_iter)
                    except StopIteration:
                        target_iter = iter(self.target_loader)
                        target_batch = next(target_iter)

                    target_batch = self.preprocess_batch(target_batch)

                    # 1. Train G - freeze discriminator parameters
                    for param in self.discriminator.parameters():
                        param.requires_grad = False

                    # Forward target domain data through model
                    with torch.cuda.amp.autocast(self.amp):
                        _ = self.model(target_batch["img"])
                        # Get features from bridge layer using our extractor
                        target_features = self.feature_extractor.get_features()

                    # Upsample features to appropriate size for discriminator
                    interp = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
                    target_features_up = interp(target_features)

                    # Pass through discriminator with source label (adversarial)
                    D_out = self.discriminator(F.softmax(target_features_up, dim=1))
                    D_source_label = torch.FloatTensor(D_out.data.size()).fill_(self.source_label)

                    # Calculate adversarial loss
                    loss_adv = 0.1 * self.weightedMSE(D_out, D_source_label)

                    if self.is_valid_number(loss_adv.data.item()):
                        self.scaler.scale(loss_adv).backward()

                # Process a batch from source domain (for standard training)
                try:
                    source_batch = next(source_iter)
                except StopIteration:
                    source_iter = iter(self.source_loader)
                    source_batch = next(source_iter)

                source_batch = self.preprocess_batch(source_batch)

                # Standard forward pass with source domain data
                with torch.cuda.amp.autocast(self.amp):
                    loss = self.model(source_batch["img"])
                    # Get features from bridge layer
                    source_features = self.feature_extractor.get_features()

                if self.is_valid_number(loss.data.item()):
                    self.scaler.scale(loss).backward()

                # 2. Train D - unfreeze discriminator parameters
                if hasattr(self, 'target_loader'):
                    for param in self.discriminator.parameters():
                        param.requires_grad = True

                    # Detach features to avoid backprop to generator
                    target_features = target_features.detach()
                    source_features = source_features.detach()

                    # Upsample features
                    target_features_up = interp(target_features)
                    source_features_up = interp(source_features)

                    # Target domain classification
                    D_out_t = self.discriminator(F.softmax(target_features_up, dim=1))
                    D_target_label = torch.FloatTensor(D_out_t.data.size()).fill_(self.target_label)
                    loss_d_t = 0.1 * self.weightedMSE(D_out_t, D_target_label)

                    if self.is_valid_number(loss_d_t.data.item()):
                        loss_d_t.backward()

                    # Source domain classification
                    D_out_s = self.discriminator(F.softmax(source_features_up, dim=1))
                    D_source_label = torch.FloatTensor(D_out_s.data.size()).fill_(self.source_label)
                    loss_d_s = 0.1 * self.weightedMSE(D_out_s, D_source_label)

                    if self.is_valid_number(loss_d_s.data.item()):
                        loss_d_s.backward()

                # Update parameters
                self.optimizer_step()
                if hasattr(self, 'target_loader'):
                    self.optimizer_D.step()
                    self.optimizer_D.zero_grad()

                # Update progress bar
                if RANK in (-1, 0):
                    # Include domain adaptation losses in progress display
                    if hasattr(self, 'target_loader'):
                        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
                        adv_loss = loss_adv.item() if 'loss_adv' in locals() else 0
                        d_loss = (loss_d_t.item() + loss_d_s.item()) / 2 if 'loss_d_t' in locals() else 0

                        pbar.set_description(
                            f"{epoch + 1}/{self.epochs} {mem} {loss.item():.4g} {adv_loss:.4g} {d_loss:.4g}"
                        )
                    else:
                        # Standard progress update
                        self.update_pbar(pbar, i, epoch)

                self.run_callbacks("on_train_batch_end")

            # Rest of the training loop remains the same...
            # (saving models, validation, etc.)