# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import de_parallel
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
        """初始化域适应训练器"""
        # 使用标准YOLOv8参数初始化
        super().__init__(cfg, overrides, _callbacks)

        # 初始化域适应组件和参数
        self.target_data = None
        self.target_dataset = None
        self.target_loader = None
        self.source_loader = None
        self.disc_lr = 0.001  # 默认判别器学习率

        # 源域和目标域标签
        self.source_label = 0
        self.target_label = 1

        # 域适应组件
        self.discriminator = None
        self.optimizer_D = None
        self.feature_extractor = None
        self.domain_adapt_enabled = False

        # 保存设备信息
        self.device = None

        # 进度条描述格式
        self.epoch_desc = "Epoch {epoch}"

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

        # 设置设备
        self.device = next(self.model.parameters()).device
        LOGGER.info(f"Model is on device: {self.device}")

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

        # 设置到相同设备上
        self.discriminator.to(self.device)
        LOGGER.info(f"Discriminator initialized on device: {self.device}")

        # 处理分布式训练
        if RANK != -1 and torch.cuda.is_available():
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

    def update_optimizer(self, epoch):
        """更新优化器的学习率"""
        # 计算主模型的学习率 - 使用余弦退火策略
        if self.args.cos_lr:
            # 余弦退火学习率
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # cosine
        else:
            # 线性学习率
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear

        # 计算当前epoch的学习率因子
        lr_factor = self.lf(epoch)

        # 更新主模型的学习率
        for param_group in self.optimizer.param_groups:
            if 'initial_lr' in param_group:
                param_group['lr'] = param_group['initial_lr'] * lr_factor
            else:
                # 如果没有initial_lr，直接设置lr
                param_group['lr'] = self.args.lr0 * lr_factor

        # 记录主模型的学习率
        if epoch % 10 == 0 or epoch == 0:  # 每10个epoch记录一次
            LOGGER.info(f'Optimizer learning rate adjusted to {self.optimizer.param_groups[0]["lr"]:.6f}')

        # 更新判别器的学习率（如果启用了域适应）
        if self.domain_adapt_enabled and self.optimizer_D is not None:
            # 使用同样的余弦退火调整判别器学习率
            current_lr = self.disc_lr * lr_factor
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = current_lr

            if epoch % 10 == 0 or epoch == 0:  # 每10个epoch记录一次
                LOGGER.info(f'Discriminator learning rate adjusted to {current_lr:.6f}')

    def _do_train(self, world_size=1):
        """执行训练，包括域适应部分"""
        # 设置训练环境与组件
        self._setup_train(world_size)

        # 如果没有启用域适应，则使用标准训练
        if not self.domain_adapt_enabled:
            return super()._do_train(world_size)

        # 开始训练循环
        for epoch in range(self.start_epoch, self.epochs):
            # 确保设置为训练模式
            self.model.train()
            self.discriminator.train()

            # 更新学习率
            self.update_optimizer(epoch)

            # 设置进度条 - 使用tqdm直接创建
            if RANK in (-1, 0):
                pbar = tqdm(enumerate(self.train_loader),
                            total=len(self.train_loader),
                            desc=f"Epoch {epoch + 1}/{self.epochs}")
            else:
                pbar = enumerate(self.train_loader)

            self.batch_i = 0

            # 创建目标域数据迭代器以便循环使用
            target_iter = iter(self.target_loader)

            # 计算每个批次的累积步数
            accumulate = max(round(self.args.nbs / self.batch_size), 1)

            # 批次迭代
            for batch_idx, batch in pbar:
                self.batch_i = batch_idx

                # 调用回调函数
                self.run_callbacks('on_train_batch_start')

                # 打印第一个批次的图像信息以进行调试
                if batch_idx == 0 and epoch == 0:
                    if 'img' in batch:
                        LOGGER.info(f"Image batch shape: {batch['img'].shape}, dtype: {batch['img'].dtype}, "
                                    f"min: {batch['img'].min().item()}, max: {batch['img'].max().item()}")

                # 载入批次到指定设备
                batch = self.preprocess_batch(batch)

                # 前向传播和计算源域损失
                preds = self.model(batch['img'])
                self.loss, self.loss_items = self.compute_loss(preds, batch)

                # 标准反向传播与参数更新
                self.scaler.scale(self.loss).backward()

                # 域对抗训练部分 --------------------
                if (batch_idx + 1) % self.args.accumulate == 0:
                    try:
                        # 获取源域特征
                        source_features = self.feature_extractor.get_features()
                        if source_features is None:
                            LOGGER.warning("Failed to extract source domain features.")
                            continue

                        # 获取目标域数据
                        try:
                            target_batch = next(target_iter)
                        except StopIteration:
                            # 重新初始化目标域数据迭代器
                            target_iter = iter(self.target_loader)
                            target_batch = next(target_iter)

                        # 预处理目标域批次
                        target_batch = self.preprocess_batch(target_batch)

                        # 前向传播获取目标域特征
                        _ = self.model(target_batch['img'])
                        target_features = self.feature_extractor.get_features()

                        if target_features is None:
                            LOGGER.warning("Failed to extract target domain features.")
                            continue

                        # 判别器训练 - 源域 (给定标签0)
                        self.optimizer_D.zero_grad()
                        D_out_source = self.discriminator(source_features.detach())
                        D_source_label = torch.FloatTensor(D_out_source.data.size()).fill_(self.source_label).to(
                            self.device)
                        D_source_loss = F.mse_loss(D_out_source, D_source_label)

                        # 判别器训练 - 目标域 (给定标签1)
                        D_out_target = self.discriminator(target_features.detach())
                        D_target_label = torch.FloatTensor(D_out_target.data.size()).fill_(self.target_label).to(
                            self.device)
                        D_target_loss = F.mse_loss(D_out_target, D_target_label)

                        # 总判别器损失并更新
                        D_loss = (D_source_loss + D_target_loss) / 2
                        D_loss.backward()
                        self.optimizer_D.step()

                        # 源域特征对抗训练 (混淆判别器)
                        self.optimizer.zero_grad()
                        source_D_out = self.discriminator(source_features)
                        # 这里我们希望源域特征被判别为目标域
                        G_source_loss = F.mse_loss(source_D_out, D_target_label)
                        G_source_loss.backward()

                        # 记录损失
                        if self.args.amp:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()

                        # 打印域适应损失信息
                        if batch_idx % 10 == 0:
                            LOGGER.info(
                                f"{colorstr('green', 'bold', 'Domain Adapt')} Epoch: {epoch}, Batch: {batch_idx}, "
                                f"D_loss: {D_loss.item():.4f}, "
                                f"G_loss: {G_source_loss.item():.4f}")

                    except Exception as e:
                        LOGGER.error(f"Error in domain adaptation training: {e}")
                        import traceback
                        LOGGER.error(traceback.format_exc())

                # 执行余下的标准训练步骤
                if self.args.amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # 调用回调函数
                self.run_callbacks('on_train_batch_end')

                # 更新进度条
                self.update_pbar(pbar, batch_idx, epoch)

            # 处理每个epoch结束时的操作
            self.run_callbacks('on_train_epoch_end')

            # 执行验证
            if (epoch + 1) % self.args.val_interval == 0:
                self.validate()

            # 保存模型
            self.save_model()

            # 调用回调函数
            self.run_callbacks('on_fit_epoch_end')

            # 检查提前终止
            if self.stopper.possible_stop:
                LOGGER.info(
                    f'Stopping training early as no improvement observed in last {self.stopper.patience} epochs.')
                break

        # 训练完成的操作
        self.run_callbacks('on_train_end')
        if world_size > 1 and RANK == 0:
            LOGGER.info(f"Training completed successfully after {epoch + 1} epochs.")

        # 清理内存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.run_callbacks('teardown')

    def update_pbar(self, pbar, batch_idx, epoch):
        """更新进度条显示"""
        # 获取内存信息
        mem = f'{torch.cuda.memory_reserved() / 1E9:.3g}G' if torch.cuda.is_available() else 'CPU'

        # 如果是tqdm进度条对象，更新描述
        if RANK in (-1, 0) and hasattr(pbar, 'set_description'):
            pbar.set_description(
                f"{epoch + 1}/{self.epochs} {mem} {self.loss_items[0]:.4f} {self.loss_items[1]:.4f} {self.loss_items[2]:.4f}"
            )

    def preprocess_batch(self, batch):
        """预处理批次数据，确保数据在正确的设备上和正确的数据类型"""
        # 如果批次为空（可能是目标域没有标签），则创建空字典
        if batch is None:
            return {'img': None}

        # 将图像移动到指定设备，并确保数据类型正确
        if 'img' in batch:
            # 检查图像数据类型并进行必要的转换
            img = batch['img']

            # 如果图像是浮点型，确保已经归一化
            if img.dtype == torch.float32:
                # 确保数值范围在[0,1]
                if img.max() > 1.0:
                    img = img / 255.0
            # 如果图像是整型，转换为浮点型并归一化
            elif img.dtype == torch.uint8:
                img = img.float() / 255.0

            # 将处理后的图像移动到指定设备
            batch['img'] = img.to(self.device, non_blocking=True)

        # 将标签移动到指定设备（如果存在）
        if 'cls' in batch:
            batch['cls'] = batch['cls'].to(self.device)
        if 'bboxes' in batch:
            batch['bboxes'] = batch['bboxes'].to(self.device)

        return batch

    def save_model(self):
        """保存包含域适应组件的模型"""
        # 保存标准检测模型
        super().save_model()

        # 如果启用了域适应，还保存判别器
        if self.domain_adapt_enabled and self.discriminator is not None:
            discriminator = de_parallel(self.discriminator)
            disc_path = str(self.save_dir / f'discriminator_{self.epoch}.pt')
            torch.save(discriminator.state_dict(), disc_path)
            LOGGER.info(f"Saved discriminator to {disc_path}")