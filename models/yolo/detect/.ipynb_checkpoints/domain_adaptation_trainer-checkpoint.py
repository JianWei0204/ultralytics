# Domain Adaptation Trainer for YOLOv8
# Extends DetectionTrainer with domain adaptation capabilities

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.GRL import GradientScalarLayer
from ultralytics.nn.trans_discriminator import TransformerDiscriminator
from ultralytics.models.yolo.detect.yolo_feature_extractor import create_domain_adapted_model
from ultralytics.models.yolo.detect.domain_data_loader import DomainAwareTrainerMixin

__all__ = "DomainAdaptationTrainer"


class DomainAdaptationTrainer(DetectionTrainer, DomainAwareTrainerMixin):
    """
    Domain Adaptation Trainer extending DetectionTrainer for YOLO models.

    Implements alternating training strategy:
    1. Main network training (discriminator frozen): adversarial loss on target domain
    2. Discriminator training (discriminator unfrozen): domain classification loss

    Follows the UDAT-car approach for domain adaptation.
    """

#     def __init__(self, cfg=None, overrides=None, _callbacks=None):
#         """Initialize domain adaptation trainer."""
#         super().__init__(cfg, overrides, _callbacks)

#         # Domain adaptation parameters
#         self.source_label = 0
#         self.target_label = 1
#         self.adv_loss_weight = 0.1
#         self.discriminator_lr = 1e-4

#         # Initialize discriminator and optimizer
#         self.discriminator = None
#         self.discriminator_optimizer = None

#         # Feature interpolation layer for upsampling to 128x128
#         self.feature_interpolator = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """Initialize domain adaptation trainer."""
        
        # 确保 cfg 包含必要的默认参数
        if cfg is None:
            cfg = {}

        # 如果 cfg 是字典，确保包含必要的参数
        if isinstance(cfg, dict):
            # 添加缺失的必要参数
            default_params = {
                'seed': 0,
                'deterministic': True,
                'device': '',
                'epochs': 100,
                'batch': 16,
                'imgsz': 640,
                'save': True,
                'val': True,
                'verbose': True,
                'amp': True,
                'lr0': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3.0
            }

            # 合并默认参数，不覆盖已存在的参数
            for key, value in default_params.items():
                if key not in cfg:
                    cfg[key] = value
        
        # Ensure overrides is a dictionary
        if overrides is None:
            overrides = {}

        # Initialize parent trainer
        super().__init__(cfg, overrides, _callbacks)

        # Domain adaptation parameters (can be overridden)
        self.source_label = 0
        self.target_label = 1
        self.adv_loss_weight = overrides.get('adv_loss_weight', 0.1)
        self.discriminator_lr = overrides.get('discriminator_lr', 1e-4)

        # Initialize discriminator and optimizer
        self.discriminator = None
        self.discriminator_optimizer = None

        # Feature interpolation layer for upsampling to 128x128
        self.feature_interpolator = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)

    def setup_model(self):
        """Override to wrap model with feature extraction capability."""
        ckpt = super().setup_model()

        # Wrap the model to enable P4 feature extraction
        self.model = create_domain_adapted_model(self.model)

        return ckpt
        """Initialize discriminator and its optimizer."""
        if self.discriminator is None:
            # Initialize TransformerDiscriminator for P4 features (assuming 512 channels)
            self.discriminator = TransformerDiscriminator(channels=512)
            self.discriminator = self.discriminator.to(self.device)

            # Wrap with DataParallel if multiple GPUs
            if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
                self.discriminator = nn.DataParallel(self.discriminator)

            # Initialize discriminator optimizer
            self.discriminator_optimizer = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.discriminator_lr,
                betas=(0.9, 0.99)
            )

    def extract_p4_features(self, batch):
        """
        Extract P4 layer features from the model.

        Uses the wrapped model's feature extraction capability.
        """
        # Use the wrapped model's P4 feature extraction
        return self.model.extract_p4_features(batch['img'])

    def compute_domain_loss(self, features, domain_label):
        """Compute domain classification loss."""
        # Upsample features to 128x128 for discriminator
        features_upsampled = self.feature_interpolator(features)

        # Apply softmax to features before discriminator
        features_softmax = F.softmax(features_upsampled, dim=1)

        # Get discriminator output
        domain_pred = self.discriminator(features_softmax)

        # Create domain labels
        batch_size = features.size(0)
        target_labels = torch.full(
            (batch_size,), domain_label, dtype=torch.float32, device=self.device
        )

        # Compute MSE loss (following train_new.py approach)
        domain_loss = torch.mean((domain_pred.squeeze() - target_labels) ** 2)

        return domain_loss

    def train_main_network(self, source_batch, target_batch):
        """
        Train main network with frozen discriminator.

        - Source domain: supervised detection loss
        - Target domain: adversarial loss (fool discriminator to classify as source)
        """
        # Freeze discriminator parameters
        for param in self.discriminator.parameters():
            param.requires_grad = False

        total_loss = 0

        # Process target domain data for adversarial loss
        if target_batch is not None:
            target_features = self.extract_p4_features(target_batch)
            adv_loss = self.compute_domain_loss(target_features, self.source_label)
            total_loss += self.adv_loss_weight * adv_loss

        # Process source domain data for detection loss
        if source_batch is not None:
            # Standard detection loss computation
            with torch.cuda.amp.autocast(self.amp):
                source_batch = self.preprocess_batch(source_batch)
                detection_loss, loss_items = self.model(source_batch)
                total_loss += detection_loss

        return total_loss, loss_items if source_batch is not None else None

    def train_discriminator(self, source_batch, target_batch):
        """
        Train discriminator with frozen main network.

        - Source domain features: classify as source (label=0)
        - Target domain features: classify as target (label=1)
        """
        # Unfreeze discriminator parameters
        for param in self.discriminator.parameters():
            param.requires_grad = True

        total_discriminator_loss = 0

        # Train on target domain features
        if target_batch is not None:
            target_features = self.extract_p4_features(target_batch)
            target_features_detached = target_features.detach()  # Detach from main network
            target_loss = self.compute_domain_loss(target_features_detached, self.target_label)
            total_discriminator_loss += target_loss

        # Train on source domain features
        if source_batch is not None:
            source_features = self.extract_p4_features(source_batch)
            source_features_detached = source_features.detach()  # Detach from main network
            source_loss = self.compute_domain_loss(source_features_detached, self.source_label)
            total_discriminator_loss += source_loss

        return total_discriminator_loss

    def _do_train(self, world_size=1):
        """
        Override parent's _do_train to implement alternating training strategy.
        """
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        # Setup discriminator after main model is initialized
        self.setup_discriminator()

        # Get data loaders for source and target domains
        # This assumes data loaders provide both source and target domain batches
        # Implementation depends on how domain data is organized

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")

        epoch = self.epochs
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            self.model.train()
            self.discriminator.train()

            if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            pbar = enumerate(self.train_loader)
            if hasattr(self, 'RANK') and self.RANK in (-1, 0):
                from ultralytics.utils import TQDM
                pbar = TQDM(enumerate(self.train_loader), total=nb)

            self.tloss = None
            self.optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()

            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")

                # Split batch into source and target domain data
                source_batch, target_batch = self.split_domain_batch(batch)

                # Warmup logic (keeping parent's warmup implementation)
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Alternating training strategy

                # 1. Train main network (G) with frozen discriminator
                with torch.cuda.amp.autocast(self.amp):
                    main_loss, loss_items = self.train_main_network(source_batch, target_batch)

                if main_loss > 0:
                    self.scaler.scale(main_loss).backward()

                # 2. Train discriminator (D) with detached features
                discriminator_loss = self.train_discriminator(source_batch, target_batch)

                if discriminator_loss > 0:
                    self.scaler.scale(discriminator_loss).backward()

                # Update losses for logging
                if loss_items is not None:
                    self.tloss = (
                        (self.tloss * i + loss_items) / (i + 1) if self.tloss is not None else loss_items
                    )

                # Optimizer step
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()

                    # Also step discriminator optimizer
                    self.scaler.unscale_(self.discriminator_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=10.0)
                    self.scaler.step(self.discriminator_optimizer)
                    self.discriminator_optimizer.zero_grad()

                    last_opt_step = ni

                # Logging (simplified from parent class)
                if hasattr(self, 'RANK') and self.RANK in (-1, 0):
                    mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
                    loss_len = self.tloss.shape[0] if hasattr(self.tloss, 'shape') and len(self.tloss.shape) else 1
                    losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss,
                                                                             0) if self.tloss is not None else [0]

                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (3 + loss_len))
                        % (f"{epoch + 1}/{self.epochs}", mem, *losses, float(discriminator_loss),
                           batch["cls"].shape[0] if "cls" in batch else 0,
                           batch["img"].shape[-1] if "img" in batch else 0)
                    )

                self.run_callbacks("on_train_batch_end")

            # End of epoch processing (simplified from parent)
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")

            if hasattr(self, 'RANK') and self.RANK in (-1, 0):
                # Validation and model saving logic from parent class
                final_epoch = epoch + 1 == self.epochs
                if hasattr(self, 'ema'):
                    self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch:
                    if hasattr(self, 'validator'):
                        self.metrics, self.fitness = self.validate()
                    else:
                        self.metrics, self.fitness = {}, 0.0

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.save_discriminator()  # Also save discriminator
                    self.run_callbacks("on_model_save")

            # Scheduler step
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            self.run_callbacks("on_fit_epoch_end")

            torch.cuda.empty_cache()

        # Final evaluation
        if hasattr(self, 'RANK') and self.RANK in (-1, 0):
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")

        torch.cuda.empty_cache()
        self.run_callbacks("teardown")

    def save_discriminator(self):
        """Save discriminator checkpoint."""
        if hasattr(self, 'save_dir') and self.discriminator is not None:
            discriminator_path = self.save_dir / "weights" / f"discriminator_epoch{self.epoch}.pt"
            torch.save({
                'epoch': self.epoch,
                'discriminator_state_dict': self.discriminator.state_dict(),
                'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
            }, discriminator_path)

    def load_discriminator(self, checkpoint_path):
        """Load discriminator from checkpoint."""
        if checkpoint_path and self.discriminator is not None:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.discriminator.load_state_dict(ckpt['discriminator_state_dict'])
            if self.discriminator_optimizer is not None:
                self.discriminator_optimizer.load_state_dict(ckpt['discriminator_optimizer'])
