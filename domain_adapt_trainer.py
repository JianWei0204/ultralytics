# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import yaml
from torch.utils.data import DataLoader

from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.models.yolo.detect import DetectionTrainer

from trans_discriminator import TransformerDiscriminator
from feature_extractor import FeatureExtractor


class DomainAdaptTrainer(DetectionTrainer):
    """
    Domain Adaptation Trainer for YOLOv8 object detection.
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # 仅使用标准YOLOv8参数初始化
        super().__init__(cfg, overrides, _callbacks)

        # 初始化域适应组件和参数
        self.target_data = None
        self.target_dataset = None
        self.target_loader = None
        self.disc_lr = 0.001  # 默认判别器学习率

        # 源域和目标域标签
        self.source_label = 0
        self.target_label = 1

        # 域适应组件
        self.discriminator = None
        self.optimizer_D = None
        self.feature_extractor = None
        self.domain_adapt_enabled = False

    def setup_domain_adaptation(self, target_data, disc_lr=0.001):
        """设置域适应训练需要的参数和组件"""
        self.target_data = target_data
        self.disc_lr = disc_lr
        self.domain_adapt_enabled = True

        LOGGER.info(f"Domain adaptation enabled with target data: {self.target_data}")
        LOGGER.info(f"Discriminator learning rate set to: {self.disc_lr}")

    def _setup_train(self, world_size):
        """设置训练与域适应组件"""
        # 初始化标准训练设置
        super()._setup_train(world_size)

        # 如果启用了域适应，则设置相关组件
        if self.domain_adapt_enabled:
            # 创建特征提取器访问桥接层输出
            self.feature_extractor = FeatureExtractor(self.model, 'Adjust_Transformer')

            # 将源域训练数据加载器保存为source_loader
            self.source_loader = self.train_loader

            # 加载目标域数据集并创建数据加载器
            LOGGER.info(f"Loading target domain dataset: {self.target_data}")

            # 解析目标域YAML文件以获取实际图像路径
            try:
                with open(self.target_data, 'r') as f:
                    target_yaml = yaml.safe_load(f)

                # 获取基础路径和训练图像路径
                base_path = target_yaml.get('path', '')
                train_path = target_yaml.get('train', '')

                if not base_path or not train_path:
                    raise ValueError(f"Missing 'path' or 'train' in {self.target_data}")

                # 构建完整的训练图像路径
                target_img_path = os.path.join(base_path, train_path)
                LOGGER.info(f"Target domain train path: {target_img_path}")

                # 检查路径是否存在
                if not os.path.exists(target_img_path):
                    raise FileNotFoundError(f"Target domain path does not exist: {target_img_path}")

                # 创建目标域数据集
                self.target_dataset = self.build_dataset(
                    img_path=target_img_path,
                    mode="train",
                    batch=self.args.batch
                )

                # 直接创建DataLoader
                batch_size = self.batch_size // max(world_size, 1)
                nw = min([os.cpu_count() // max(world_size, 1), self.args.workers, 8])
                collate_fn = self.target_dataset.collate_fn

                if RANK == -1:
                    # 非分布式训练
                    self.target_loader = DataLoader(
                        dataset=self.target_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=nw,
                        collate_fn=collate_fn,
                        pin_memory=True,
                        drop_last=True,  # 丢弃最后不完整的批次
                    )
                else:
                    # 分布式训练
                    sampler = torch.utils.data.distributed.DistributedSampler(
                        self.target_dataset, shuffle=True
                    )
                    self.target_loader = DataLoader(
                        dataset=self.target_dataset,
                        batch_size=batch_size,
                        sampler=sampler,
                        num_workers=nw,
                        collate_fn=collate_fn,
                        pin_memory=True,
                        drop_last=True,
                    )

                # 初始化判别器及其优化器
                self.setup_discriminator()

                LOGGER.info(f"Target domain dataloader created with {len(self.target_loader)} batches")

            except Exception as e:
                LOGGER.error(f"Error loading target domain dataset: {e}")
                raise

    def setup_discriminator(self):
        """初始化域判别器及其优化器"""
        # 桥接层特征通道数
        feature_channels = 128  # 从yaml文件中的Adjust_Transformer层获取

        # 创建判别器
        self.discriminator = TransformerDiscriminator(channels=feature_channels)
        self.discriminator.train()
        self.discriminator.cuda()

        if RANK != -1:
            self.discriminator = nn.parallel.DistributedDataParallel(
                self.discriminator, device_ids=[RANK]
            )

        # 创建判别器优化器
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.disc_lr,
            betas=(0.9, 0.99)
        )
        self.optimizer_D.zero_grad()

        LOGGER.info(f"Discriminator initialized with learning rate {self.disc_lr}")

    def weightedMSE(self, D_out, label):
        """计算判别器的加权MSE损失"""
        return torch.mean((D_out - label.cuda()).abs() ** 2)

    def adjust_learning_rate_D(self, epoch):
        """判别器的多项式学习率衰减"""

        def lr_poly(base_lr, iter, max_iter, power):
            return base_lr * ((1 - float(iter) / max_iter) ** power)

        lr = lr_poly(self.disc_lr, epoch, self.epochs, 0.8)
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        return lr

    def _do_train(self, world_size=1):
        """修改后的训练循环，包含域适应"""
        # 如果域适应未启用，则执行标准训练
        if not self.domain_adapt_enabled:
            LOGGER.info("Domain adaptation not enabled, performing standard training")
            return super()._do_train(world_size)

        # 设置分布式训练和一般训练组件
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

            # 调整判别器学习率
            cur_lr_d = self.adjust_learning_rate_D(epoch)

            if RANK != -1:
                self.source_loader.sampler.set_epoch(epoch)
                self.target_loader.sampler.set_epoch(epoch)

            # 创建源域和目标域数据迭代器
            target_iter = iter(self.target_loader)
            source_iter = iter(self.source_loader)

            nb = len(self.source_loader)
            self.tloss = None

            # 进度条
            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = self.progress_bar(nb)
            else:
                pbar = range(nb)

            self.optimizer.zero_grad()
            self.optimizer_D.zero_grad()

            for i in pbar:
                self.run_callbacks("on_train_batch_start")

                # 处理目标域数据（用于域适应）
                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.target_loader)
                    target_batch = next(target_iter)

                target_batch = self.preprocess_batch(target_batch)

                # 1. 训练生成器 - 冻结判别器参数
                for param in self.discriminator.parameters():
                    param.requires_grad = False

                # 将目标域数据通过模型前向传播
                with torch.cuda.amp.autocast(self.amp):
                    _ = self.model(target_batch["img"])
                    # 使用特征提取器获取桥接层特征
                    target_features = self.feature_extractor.get_features()

                # 上采样特征到适合判别器的尺寸
                interp = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
                target_features_up = interp(target_features)

                # 将特征通过判别器，标记为源域（对抗性）
                D_out = self.discriminator(F.softmax(target_features_up, dim=1))
                D_source_label = torch.FloatTensor(D_out.data.size()).fill_(self.source_label)

                # 计算对抗损失
                loss_adv = 0.1 * self.weightedMSE(D_out, D_source_label)

                if self.is_valid_number(loss_adv.data.item()):
                    self.scaler.scale(loss_adv).backward()

                # 处理源域数据（用于标准训练）
                try:
                    source_batch = next(source_iter)
                except StopIteration:
                    source_iter = iter(self.source_loader)
                    source_batch = next(source_iter)

                source_batch = self.preprocess_batch(source_batch)

                # 使用源域数据进行标准前向传播
                with torch.cuda.amp.autocast(self.amp):
                    loss = self.model(source_batch["img"])
                    # 获取桥接层特征
                    source_features = self.feature_extractor.get_features()

                if self.is_valid_number(loss.data.item()):
                    self.scaler.scale(loss).backward()

                # 2. 训练判别器 - 解冻判别器参数
                for param in self.discriminator.parameters():
                    param.requires_grad = True

                # 分离特征以避免反向传播到生成器
                target_features = target_features.detach()
                source_features = source_features.detach()

                # 上采样特征
                target_features_up = interp(target_features)
                source_features_up = interp(source_features)

                # 目标域分类
                D_out_t = self.discriminator(F.softmax(target_features_up, dim=1))
                D_target_label = torch.FloatTensor(D_out_t.data.size()).fill_(self.target_label)
                loss_d_t = 0.1 * self.weightedMSE(D_out_t, D_target_label)

                if self.is_valid_number(loss_d_t.data.item()):
                    loss_d_t.backward()

                # 源域分类
                D_out_s = self.discriminator(F.softmax(source_features_up, dim=1))
                D_source_label = torch.FloatTensor(D_out_s.data.size()).fill_(self.source_label)
                loss_d_s = 0.1 * self.weightedMSE(D_out_s, D_source_label)

                if self.is_valid_number(loss_d_s.data.item()):
                    loss_d_s.backward()

                # 更新参数
                self.optimizer_step()
                self.optimizer_D.step()
                self.optimizer_D.zero_grad()

                # 更新进度条
                if RANK in (-1, 0):
                    # 在进度显示中包含域适应损失
                    mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
                    adv_loss = loss_adv.item() if 'loss_adv' in locals() else 0
                    d_loss = (loss_d_t.item() + loss_d_s.item()) / 2 if 'loss_d_t' in locals() else 0

                    pbar.set_description(
                        f"{epoch + 1}/{self.epochs} {mem} {loss.item():.4g} {adv_loss:.4g} {d_loss:.4g}"
                    )

                self.run_callbacks("on_train_batch_end")

            # 训练循环结束部分
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.lr["lr/discriminator"] = cur_lr_d

            self.run_callbacks("on_train_epoch_end")

            # 验证和模型保存
            if RANK in (-1, 0):
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # 验证
                if self.args.val or epoch + 1 == self.epochs:
                    self.metrics, self.fitness = self.validate()

                # 保存模型
                if (epoch + 1) % self.args.save_period == 0 or epoch + 1 == self.epochs:
                    self.save_model()

                    # 保存判别器
                    discriminator_state = {
                        'epoch': epoch,
                        'state_dict': de_parallel(self.discriminator).state_dict(),
                        'optimizer': self.optimizer_D.state_dict(),
                        'disc_lr': self.disc_lr
                    }

                    torch.save(
                        discriminator_state,
                        os.path.join(self.save_dir, f'd_checkpoint_e{epoch + 1}.pt')
                    )

            # 更新调度器
            self.scheduler.step()

            torch.cuda.empty_cache()

            # 检查是否早停
            if self.stopper(epoch + 1, self.fitness):
                break

        # 训练结束
        if RANK in (-1, 0):
            LOGGER.info(f"\n{self.epochs} epochs completed in "
                        f"{(time.time() - self.train_time_start) / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")

        torch.cuda.empty_cache()
        self.run_callbacks("teardown")

    def is_valid_number(self, x):
        """检查值是否为有效数字"""
        import math
        return not (math.isnan(x) or math.isinf(x) or x > 1e4)

    def save_model(self):
        """保存模型和判别器"""
        # 首先保存标准模型
        super().save_model()

        # 如果域适应启用，保存判别器
        if self.domain_adapt_enabled and hasattr(self, 'discriminator') and self.discriminator is not None:
            discriminator_state = {
                'epoch': self.epoch,
                'state_dict': de_parallel(self.discriminator).state_dict(),
                'optimizer': self.optimizer_D.state_dict(),
                'disc_lr': self.disc_lr
            }

            # 保存最新状态
            disc_last_path = os.path.join(self.wdir, 'discriminator_last.pt')
            torch.save(discriminator_state, disc_last_path)

            # 如果是最佳模型，也保存对应的判别器
            if self.best_fitness == self.fitness:
                disc_best_path = os.path.join(self.wdir, 'discriminator_best.pt')
                torch.save(discriminator_state, disc_best_path)