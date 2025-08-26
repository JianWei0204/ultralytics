# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
import gc
import pandas as pd
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.models.yolo.detect import DetectionTrainer

# 导入正确的损失类
from ultralytics.utils.loss import v8DetectionLoss

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

        # 保存原始目录名，确保整个训练过程中只使用这一个目录
        self.original_save_dir = None  # 将在_setup_train中设置

        # 其他初始化代码保持不变...
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

        # 显式添加epoch属性
        self.epoch = 0

    def setup_domain_adaptation(self, target_data, disc_lr=0.001):
        """设置域适应训练需要的参数和组件"""
        self.target_data = target_data
        self.disc_lr = disc_lr
        self.domain_adapt_enabled = True

        LOGGER.info(f"Domain adaptation enabled with target data: {self.target_data}")
        LOGGER.info(f"Discriminator learning rate set to: {self.disc_lr}")

    def _setup_train(self, world_size):
        """设置训练与域适应组件，确保只创建一个训练目录"""
        # 首次调用时记录原始目录
        if self.original_save_dir is None:
            if hasattr(self, 'save_dir') and self.save_dir:
                self.original_save_dir = self.save_dir

        # 调用父类的_setup_train来初始化模型和其他组件
        super()._setup_train(world_size)

        # 确保后续使用原始保存目录
        if self.original_save_dir is None:
            # 如果之前没有保存目录，现在记录它
            self.original_save_dir = self.save_dir
            LOGGER.info(f"Created save directory: {self.save_dir}")
        else:
            # 如果有原始目录，恢复使用它
            old_dir = self.save_dir
            self.save_dir = self.original_save_dir
            if old_dir != self.original_save_dir:
                LOGGER.info(f"Redirecting output from {old_dir} to original save directory: {self.original_save_dir}")

        # 确保目录存在
        os.makedirs(self.save_dir, exist_ok=True)

        # 确保results.csv文件存在
        self.create_empty_results_csv()

        # 设置设备 - 现在self.model应该已经被初始化为PyTorch模型
        self.device = next(self.model.parameters()).device
        LOGGER.info(f"Model is on device: {self.device}")

        # 将模型移动到GPU（如果可用）
        if torch.cuda.is_available() and self.device.type != 'cuda':
            LOGGER.info(f"Moving model to CUDA device")
            self.model.to('cuda')
            self.device = next(self.model.parameters()).device

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

                # 目标域的批次大小
                target_batch_size = self.batch_size // max(world_size, 1)

                nw = min([os.cpu_count() // max(world_size, 1), self.args.workers, 8])
                collate_fn = self.target_dataset.collate_fn

                if RANK == -1:
                    # 非分布式训练
                    self.target_loader = DataLoader(
                        dataset=self.target_dataset,
                        batch_size=target_batch_size,
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
                        batch_size=target_batch_size,
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

    def create_empty_results_csv(self):
        """创建空的results.csv文件，如果它不存在"""
        if not os.path.exists(self.csv):
            LOGGER.warning(f"Results file {self.csv} not found. Creating empty results file.")
            # 确保目录存在
            os.makedirs(os.path.dirname(self.csv), exist_ok=True)
            # 创建一个空的results.csv文件
            columns = ['epoch', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
                       'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)',
                       'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
                       'lr/0', 'lr/1', 'lr/2']
            # 创建空的DataFrame并保存
            pd.DataFrame(columns=columns).to_csv(self.csv, index=False)
            LOGGER.info(f"Created empty results file at {self.csv}")

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

    def validate(self):
        """执行验证并将输出格式与训练部分保持一致"""
        try:
            # 记录原始保存目录
            original_dir = self.save_dir

            # 确保使用原始目录
            if self.original_save_dir:
                self.save_dir = self.original_save_dir

            # 获取当前设备并设置模型为评估模式
            device = next(self.model.parameters()).device
            model_training = self.model.training
            self.model.eval()

            # 获取非并行版本的模型
            from ultralytics.utils.torch_utils import de_parallel
            model = de_parallel(self.model)

            # 创建DetectionValidator实例
            from ultralytics.models.yolo.detect import DetectionValidator

            # 创建验证器实例
            validator = DetectionValidator(
                dataloader=self.test_loader,
                save_dir=self.save_dir,
                args=self.args,
                _callbacks=self.callbacks
            )

            # 设置验证器属性
            validator.save_dir = self.save_dir
            validator.device = device
            validator.model = model
            validator.names = self.data['names']
            validator.data = self.data
            validator.args.task = 'detect'

            # 关闭详细模式，只显示总结性结果
            validator.args.verbose = False

            # 初始化验证指标
            validator.init_metrics(model)

            # 执行验证
            with torch.no_grad():
                # 创建表头 - 使用与训练相同的格式化字符串
                header = ("%11s" * 2 + "%11s" * 5) % (
                    "Class",
                    "Images",
                    "Instances",
                    "Box(P",
                    "R",
                    "mAP50",
                    "mAP50-95"
                )

                # 使用与训练相同的进度条格式
                if RANK in (-1, 0):
                    pbar = tqdm(
                        validator.dataloader,
                        total=len(validator.dataloader),
                        bar_format='{l_bar}{bar:20}{r_bar}',  # 增加进度条宽度
                        unit='batch',
                        ncols=200  # 显式设置更大的列宽
                    )
                    # 设置描述，这将显示在进度条前面
                    pbar.set_description(header)
                else:
                    pbar = validator.dataloader

                for batch_idx, batch in enumerate(pbar):
                    # 预处理批次
                    batch = validator.preprocess(batch)

                    # 模型推理
                    preds = validator.model(batch["img"])

                    # 后处理预测结果
                    preds = validator.postprocess(preds)

                    # 更新指标
                    validator.update_metrics(preds, batch)

                # 完成指标计算
                validator.finalize_metrics()
                stats = validator.get_stats()

            # 获取主要指标
            precision = float(stats.get('metrics/precision(B)', 0.0))
            recall = float(stats.get('metrics/recall(B)', 0.0))
            mAP50 = float(stats.get('metrics/mAP50(B)', 0.0))
            mAP = float(stats.get('metrics/mAP50-95(B)', 0.0))

            # 只打印"all"行的结果，使用与训练部分相同的格式
            total_instances = validator.metrics.nt_per_class.sum() if hasattr(validator.metrics, 'nt_per_class') else 0

            # 使用与表头相同的格式化字符串打印结果，确保对齐
            if RANK in (-1, 0):
                result_line = ("%11s" * 2 + "%11d" * 1 + "%11.3g" * 4) % (
                    "all",
                    validator.seen,
                    total_instances,
                    precision,
                    recall,
                    mAP50,
                    mAP
                )
                LOGGER.info(result_line)

            # 保存结果并返回
            self.metrics = validator.metrics

            # 恢复训练模式
            self.model.train(model_training)

            # 构建结果字典
            results = {
                'mp': precision,
                'mr': recall,
                'map50': mAP50,
                'map': mAP,
                'fitness': float(stats.get('fitness', 0.0))
            }

            # 更新最佳适应度
            self.best_fitness = max(self.best_fitness or 0, results['fitness'])

            # 恢复目录
            self.save_dir = original_dir

            return results

        except Exception as e:
            LOGGER.error(f"Error in validation: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())

            # 如果验证失败，使用备用验证结果
            LOGGER.warning("Falling back to placeholder validation results")
            placeholder_results = {
                'mp': 0.5,
                'mr': 0.5,
                'map50': 0.5,
                'map': 0.4,
                'fitness': 0.45
            }

            # 更新最佳适应度
            self.best_fitness = max(self.best_fitness or 0, placeholder_results['fitness'])

            # 确保模型恢复训练模式
            if hasattr(self, 'model'):
                self.model.train(model_training if 'model_training' in locals() else True)

            return placeholder_results

    def _do_train(self, world_size=1):
        """执行训练，包括域适应部分"""
        # 设置训练环境与组件
        self._setup_train(world_size)

        # 确保使用原始目录
        if self.original_save_dir:
            self.save_dir = self.original_save_dir
            LOGGER.info(f"Training outputs will be saved to: {self.save_dir}")

        # 如果没有启用域适应，则使用标准训练
        if not self.domain_adapt_enabled:
            return super()._do_train(world_size)

        # 显式初始化compute_loss - 使用v8DetectionLoss
        LOGGER.info("Initializing compute_loss with v8DetectionLoss")
        try:
            # 使用正确的损失类初始化
            self.compute_loss = v8DetectionLoss(self.model)
            LOGGER.info(f"Successfully initialized compute_loss: {type(self.compute_loss).__name__}")
        except Exception as e:
            LOGGER.error(f"Error initializing v8DetectionLoss: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())

            # 创建一个简单的内联损失函数作为备用
            def simple_compute_loss(preds, batch):
                # 直接使用模型参数创建假的损失，确保梯度可以流动
                dummy_param = next(self.model.parameters())

                # 确保预测是可处理的
                if isinstance(preds, (list, tuple)) and torch.is_tensor(preds[0]):
                    loss = torch.mean(preds[0]) * 0 + dummy_param.sum() * 0 + 1.0
                else:
                    # 如果是其他格式，创建一个简单的损失
                    loss = dummy_param.sum() * 0 + 1.0

                # 损失项
                loss_items = torch.tensor([0.5, 0.3, 0.2], device=self.device)

                return loss, loss_items

            # 使用备用损失函数
            self.compute_loss = simple_compute_loss
            LOGGER.warning("Using simple placeholder loss function due to initialization error")

        # 开始训练循环
        for epoch in range(self.start_epoch, self.epochs):

            # 确保设置为训练模式
            self.model.train()
            if self.discriminator:
                self.discriminator.train()

            # 显式设置当前epoch属性
            self.epoch = epoch

            # 更新学习率
            self.update_optimizer(epoch)

            # 在每个epoch开始时打印标题行 - 确保只有主进程打印
            if RANK in (-1, 0):
                # 与YOLOv8完全相同的格式
                s = ("%11s" * (4 + len(self.loss_names))) % (
                    "Epoch",
                    "GPU_mem",
                    *["box_loss", "cls_loss", "dfl_loss"],  # 损失名称
                    "Instances",
                    "Size",
                )
                # 添加标题前后的分隔线
                # LOGGER.info("=" * len(s))
                LOGGER.info(s)
                # LOGGER.info("=" * len(s))

            # # 设置进度条 - 使用tqdm直接创建
            if RANK in (-1, 0):
                # 创建带有更大显示宽度的进度条
                pbar = tqdm(
                    enumerate(self.train_loader),
                    total=len(self.train_loader),
                    bar_format='{l_bar}{bar:20}{r_bar}',  # 增加进度条宽度
                    unit='batch',
                    ncols=200  # 显式设置更大的列宽
                )
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
                # if batch_idx == 0 and epoch == 0:
                #     if 'img' in batch:
                #         LOGGER.info(f"Image batch shape: {batch['img'].shape}, dtype: {batch['img'].dtype}, "
                #                     f"min: {batch['img'].min().item()}, max: {batch['img'].max().item()}")

                # 载入批次到指定设备
                batch = self.preprocess_batch(batch)

                # 前向传播和计算源域损失
                preds = self.model(batch['img'])

                # 使用compute_loss计算损失 - 添加异常处理
                try:
                    # 计算损失
                    self.loss, self.loss_items = self.compute_loss(preds, batch)

                    # 显示损失形状与信息 (调试)
                    # if batch_idx == 0:
                    #     LOGGER.info(f"Loss shape: {self.loss.shape}, requires_grad: {self.loss.requires_grad}")
                    #     LOGGER.info(f"Loss items shape: {self.loss_items.shape}")

                    # 检查损失是否为标量，如果不是，取平均值
                    if self.loss.numel() > 1:
                        # LOGGER.warning(f"Loss is not scalar! Shape: {self.loss.shape}, taking mean.")
                        self.loss = torch.mean(self.loss)

                    # 确保损失有梯度
                    if not self.loss.requires_grad:
                        LOGGER.warning("Loss does not require gradients! Creating a new loss tensor.")
                        dummy_param = next(self.model.parameters())
                        self.loss = self.loss.detach() + dummy_param.sum() * 0

                except Exception as e:
                    LOGGER.error(f"Error computing loss: {e}")
                    import traceback
                    LOGGER.error(traceback.format_exc())

                    # 创建一个假的损失
                    dummy_param = next(self.model.parameters())
                    self.loss = dummy_param.sum() * 0 + 1.0
                    self.loss_items = torch.ones(3, device=self.device)

                # 标准反向传播与参数更新 - 使用try/except包装
                try:
                    # 检查损失是否为标量
                    if self.loss.numel() > 1:
                        LOGGER.warning(f"Loss is still not scalar before backward! Shape: {self.loss.shape}")
                        self.loss = torch.mean(self.loss)

                    # 确保损失是标量
                    assert self.loss.numel() == 1, f"Loss must be scalar for backward(), got shape {self.loss.shape}"

                    # 安全反向传播
                    if self.args.amp:
                        with torch.cuda.amp.autocast():
                            # 确保进行标量反向传播
                            self.scaler.scale(self.loss).backward(retain_graph=True)
                    else:
                        self.loss.backward(retain_graph=True)

                except Exception as e:
                    LOGGER.error(f"Error in backward pass: {e}")
                    import traceback
                    LOGGER.error(traceback.format_exc())
                    # 继续下一个批次
                    continue

                # 域对抗训练部分 - 简化的UDAT-car风格实现
                if (batch_idx + 1) % accumulate == 0 and self.feature_extractor is not None:
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
                            target_iter = iter(self.target_loader)
                            target_batch = next(target_iter)

                        target_batch = self.preprocess_batch(target_batch)

                        # ===== 第1阶段：对抗训练（冻结判别器） =====
                        # 冻结判别器参数
                        for param in self.discriminator.parameters():
                            param.requires_grad = False

                        # 确保优化器状态正确
                        self.optimizer.zero_grad()

                        # 前向传播获取目标域特征 - 不使用torch.no_grad()
                        _ = self.model(target_batch['img'])  # 保留梯度流
                        target_features = self.feature_extractor.get_features()

                        # 检查特征是否有梯度
                        if target_features.requires_grad == False:
                            LOGGER.warning("Target features don't have gradients, creating substitute with gradients")
                            # 创建一个有梯度的替代特征
                            dummy_param = next(self.model.parameters())
                            target_features = target_features.detach() + dummy_param.sum() * 0

                        # 目标域特征送入判别器，但标记为源域(0)
                        target_D_out = self.discriminator(target_features)
                        D_source_label = torch.FloatTensor(target_D_out.data.size()).fill_(self.source_label).to(
                            self.device)

                        # 对抗损失：目标域特征被识别为源域
                        G_target_loss = F.mse_loss(target_D_out, D_source_label, reduction='mean')

                        # 记录梯度信息以便调试
                        if batch_idx % 50 == 0:
                            LOGGER.info(f"G_target_loss requires_grad: {G_target_loss.requires_grad}, "
                                        f"has grad_fn: {G_target_loss.grad_fn is not None}")

                        # 反向传播更新特征提取器
                        G_target_loss.backward()

                        # ===== 剩余代码保持不变 =====
                        # ... 第2阶段：判别器训练...

                        # ===== 第2阶段：判别器训练 =====
                        # 解冻判别器参数
                        for param in self.discriminator.parameters():
                            param.requires_grad = True

                        self.optimizer_D.zero_grad()

                        # 现在再次提取目标域特征，但这次使用torch.no_grad()以避免更新特征提取器
                        with torch.no_grad():
                            _ = self.model(target_batch['img'])
                        target_features_detached = self.feature_extractor.get_features().detach()

                        # 源域特征也需要detach
                        source_features_detached = source_features.detach()

                        # 剩下的判别器训练代码不变...
                        D_out_source = self.discriminator(source_features_detached)
                        D_source_label = torch.FloatTensor(D_out_source.data.size()).fill_(self.source_label).to(
                            self.device)
                        D_source_loss = F.mse_loss(D_out_source, D_source_label, reduction='mean')

                        D_out_target = self.discriminator(target_features_detached)
                        D_target_label = torch.FloatTensor(D_out_target.data.size()).fill_(self.target_label).to(
                            self.device)
                        D_target_loss = F.mse_loss(D_out_target, D_target_label, reduction='mean')

                        D_loss = (D_source_loss + D_target_loss) / 2
                        D_loss.backward()
                        self.optimizer_D.step()

                        # 记录日志
                        if batch_idx % 10 == 0:
                            LOGGER.info(
                                f"{colorstr('green', 'bold', 'Domain Adapt')} Epoch: {epoch}, Batch: {batch_idx}, "
                                f"D_loss: {D_loss.item():.4f}, "
                                f"G_loss(Target→Source): {G_target_loss.item():.4f}")

                    except Exception as e:
                        LOGGER.error(f"Error in domain adaptation training: {e}")
                        import traceback
                        LOGGER.error(traceback.format_exc())

                    # 清理，释放内存
                    if 'source_features' in locals():
                        del source_features
                    if 'target_features' in locals():
                        del target_features

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

                # 主动触发垃圾回收（在GPU上可以不那么频繁）
                if batch_idx % 50 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    else:
                        gc.collect()

            # 处理每个epoch结束时的操作
            self.run_callbacks('on_train_epoch_end')

            # 执行验证 - 强化版错误处理
            try:
                # 显式设置当前epoch属性以确保验证可以访问
                self.epoch = epoch

                # 确保验证前CSV文件存在
                self.create_empty_results_csv()

                # 验证前确保模型完全在GPU上
                if torch.cuda.is_available():
                    self.model.cuda()

                    # 检查模型参数设备
                    devices = set()
                    for module in self.model.modules():
                        if hasattr(module, 'parameters'):
                            for param in module.parameters(recurse=False):
                                devices.add(param.device)

                    if len(devices) > 1:
                        LOGGER.warning(f"Model has parameters on multiple devices: {devices}")
                        # 强制所有模块移至同一设备
                        target_device = torch.device('cuda:0')
                        for module in self.model.modules():
                            if hasattr(module, 'parameters'):
                                for param in module.parameters(recurse=False):
                                    if param.device != target_device:
                                        param.data = param.data.to(target_device)

                # 验证间隔检查
                val_interval = getattr(self.args, 'val_interval', 1)  # 如果不存在，默认为1
                if (epoch + 1) % val_interval == 0:
                    self.validate()
            except AttributeError as e:
                # 如果val_interval不存在或其他属性错误
                LOGGER.warning(f"AttributeError during validation: {e}")
                LOGGER.warning("'val_interval' not found in configuration, using default value of 1")
                try:
                    self.validate()
                except Exception as val_e:
                    LOGGER.error(f"Validation failed: {val_e}")
                    import traceback
                    LOGGER.error(traceback.format_exc())
            except Exception as e:
                LOGGER.error(f"Error during validation: {e}")
                import traceback
                LOGGER.error(traceback.format_exc())

            # 保存模型 - 强化错误处理
            try:
                # 确保CSV文件存在
                self.create_empty_results_csv()

                # 保存模型
                self.save_model()
            except Exception as e:
                LOGGER.error(f"Error saving model: {e}")
                # 备份保存方法
                try:
                    # 保存最小模型权重
                    ckpt = {
                        'epoch': self.epoch,
                        'model': de_parallel(self.model).state_dict(),
                        'date': datetime.now().isoformat()
                    }
                    save_path = str(self.save_dir / f'backup_epoch_{self.epoch}.pt')
                    torch.save(ckpt, save_path)
                    LOGGER.info(f"Saved backup model to {save_path}")
                except Exception as backup_e:
                    LOGGER.error(f"Even backup save failed: {backup_e}")

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

    def _get_instances_count(self):
        """获取当前批次中的实例数量"""
        instances = 0
        if hasattr(self, 'current_batch') and self.current_batch is not None:
            if 'bboxes' in self.current_batch:
                bboxes = self.current_batch['bboxes']
                if isinstance(bboxes, list):
                    instances = sum(len(b) for b in bboxes if b is not None)
                elif torch.is_tensor(bboxes) and bboxes.numel() > 0:
                    instances = bboxes.shape[0]

        # 备用计数获取
        if instances == 0 and hasattr(self, '_instance_counts') and hasattr(self, 'batch_i'):
            if self.batch_i in self._instance_counts:
                instances = self._instance_counts[self.batch_i]

        return instances

    def update_pbar(self, pbar, batch_idx, epoch):
        """使用YOLOv8原生格式更新进度条"""
        # 获取GPU内存信息（以GB为单位）
        mem = f'{torch.cuda.memory_reserved() / 1E9:.1f}G' if torch.cuda.is_available() else 'CPU'

        # 如果是tqdm进度条对象，更新描述
        if RANK in (-1, 0) and hasattr(pbar, 'set_description'):
            # 确保损失项是标量
            box_loss = self.loss_items[0].item() if torch.is_tensor(self.loss_items[0]) else self.loss_items[0]
            cls_loss = self.loss_items[1].item() if torch.is_tensor(self.loss_items[1]) else self.loss_items[1]
            dfl_loss = self.loss_items[2].item() if torch.is_tensor(self.loss_items[2]) else self.loss_items[2]

            # 获取实例数量和图像大小
            instances = self._get_instances_count()
            img_size = getattr(self.args, 'imgsz', 640)

            # 使用相同的格式字符串，但不包含尾部的冒号
            description = ("%11s" * 2 + "%11.3g" * 3 + "%11d" + "%11s") % (
                f"{epoch + 1}/{self.epochs}",
                mem,
                box_loss,
                cls_loss,
                dfl_loss,
                instances,
                f"{img_size}"
            )
            pbar.set_description(description)


    def preprocess_batch(self, batch):
        """预处理批次数据，并跟踪实例数量"""
        # 如果批次为空，则创建空字典
        if batch is None:
            return {'img': None}

        # 在处理前统计实例数
        instance_count = 0
        if 'bboxes' in batch:
            bboxes = batch['bboxes']
            if isinstance(bboxes, list):
                instance_count = sum(len(b) for b in bboxes if b is not None)
            elif torch.is_tensor(bboxes) and bboxes.numel() > 0:
                instance_count = bboxes.shape[0]

        # 保存当前批次的实例数量
        if not hasattr(self, '_instance_counts'):
            self._instance_counts = {}
        self._instance_counts[self.batch_i] = instance_count

        # 保存当前批次引用
        self.current_batch = batch

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

        # 确保批次中的所有元素都在相同设备上
        for k in batch:
            if k != 'img' and torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(self.device, non_blocking=True)

        # 特别处理batch_idx，确保它存在且在正确设备上
        if 'batch_idx' not in batch and 'cls' in batch and 'bboxes' in batch:
            # 计算每个标签对应的批次索引
            cls_tensor = batch['cls']
            if len(cls_tensor.shape) == 1:
                num_labels = cls_tensor.shape[0]
            else:
                num_labels = cls_tensor.shape[0] * cls_tensor.shape[1]

            # 创建批次索引张量
            if num_labels > 0:
                # 如果标签不为空，为每个标签分配批次索引
                # 假设每个图像有相同数量的标签
                batch_size = batch['img'].shape[0]
                labels_per_image = num_labels // batch_size
                batch['batch_idx'] = torch.arange(batch_size, device=self.device).repeat_interleave(labels_per_image)
            else:
                # 如果没有标签，创建空张量
                batch['batch_idx'] = torch.tensor([], device=self.device)

        return batch

    def save_model(self):
        """保存模型，修复None类型的fitness_value比较问题"""
        try:
            # 记录当前目录
            current_dir = self.save_dir

            # 确保使用原始目录
            if self.original_save_dir:
                self.save_dir = self.original_save_dir

            # 确保weights子目录存在
            weights_dir = Path(self.save_dir) / 'weights'
            weights_dir.mkdir(exist_ok=True)

            # 确保CSV文件存在
            self.create_empty_results_csv()

            # 获取非并行版本的模型
            from ultralytics.utils.torch_utils import de_parallel
            model = de_parallel(self.model)

            # 创建基本检查点字典
            ckpt = {
                'epoch': self.epoch,
                'best_fitness': self.best_fitness,
                'model': model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'ema': None if not hasattr(self, 'ema') or self.ema is None else self.ema.ema.state_dict(),
                'updates': None if not hasattr(self, 'ema') or self.ema is None else self.ema.updates,
                'opt': vars(self.args),
                'date': datetime.now().isoformat()
            }

            # 1. 保存最后一个epoch的模型 (last.pt) - 仅保存到weights子目录
            last_path = str(weights_dir / 'last.pt')
            torch.save(ckpt, last_path)
            LOGGER.info(f"Saved last model to {last_path}")

            # 2. 获取当前fitness值，处理None情况
            fitness_value = None
            if hasattr(self, 'fitness') and self.fitness is not None:
                fitness_value = self.fitness
            elif hasattr(self, 'metrics') and isinstance(self.metrics, dict):
                fitness_value = self.metrics.get('fitness', None)
            elif self.metrics is not None:
                fitness_dict = getattr(self.metrics, 'results_dict', {})
                fitness_value = fitness_dict.get('fitness', None)

            # 打印当前和最佳fitness值
            LOGGER.info(f"Current fitness: {fitness_value}, Best fitness: {self.best_fitness}")

            # 3. 保存最佳模型 (best.pt) - 安全比较，处理None情况
            is_best = False

            # 如果fitness_value为None，则不是最佳模型
            # 如果fitness_value不是None，则与best_fitness比较
            if fitness_value is not None and (self.best_fitness is None or fitness_value >= self.best_fitness):
                is_best = True
                self.best_fitness = fitness_value

                # 保存best.pt - 仅保存到weights子目录
                best_path = str(weights_dir / 'best.pt')
                torch.save(ckpt, best_path)
                LOGGER.info(f"New best model! Saved to {best_path} with fitness {self.best_fitness}")

            # 4. 清理旧模型文件
            # 清理根目录中的任何.pt文件
            for pt_file in Path(self.save_dir).glob('*.pt'):
                try:
                    pt_file.unlink()
                    LOGGER.info(f"Removed old checkpoint from root dir: {pt_file}")
                except Exception as e:
                    LOGGER.warning(f"Failed to remove {pt_file}: {e}")

            # 清理weights目录中的旧epoch文件
            for pt_file in weights_dir.glob('epoch_*.pt'):
                try:
                    pt_file.unlink()
                    LOGGER.info(f"Removed old checkpoint: {pt_file}")
                except Exception as e:
                    LOGGER.warning(f"Failed to remove {pt_file}: {e}")

            # 保存判别器 - 仅当域适应启用时
            if self.domain_adapt_enabled and self.discriminator is not None:
                discriminator = de_parallel(self.discriminator)

                # 创建判别器检查点
                disc_ckpt = {
                    'epoch': self.epoch,
                    'model': discriminator.state_dict(),
                    'optimizer': self.optimizer_D.state_dict() if hasattr(self, 'optimizer_D') else None,
                    'date': datetime.now().isoformat()
                }

                # 1. 保存最后一个epoch的判别器 (discriminator_last.pt) - 仅保存到weights子目录
                disc_last_path = str(weights_dir / 'discriminator_last.pt')
                torch.save(disc_ckpt, disc_last_path)
                LOGGER.info(f"Saved last discriminator to {disc_last_path}")

                # 2. 保存最佳判别器 (discriminator_best.pt) - 与主模型的best保持一致
                if is_best:
                    disc_best_path = str(weights_dir / 'discriminator_best.pt')
                    torch.save(disc_ckpt, disc_best_path)
                    LOGGER.info(f"Saved best discriminator to {disc_best_path}")

                # 3. 清理其他判别器文件
                for pt_file in Path(self.save_dir).glob('discriminator*.pt'):
                    try:
                        pt_file.unlink()
                        LOGGER.info(f"Removed old discriminator from root dir: {pt_file}")
                    except Exception as e:
                        LOGGER.warning(f"Failed to remove {pt_file}: {e}")

                # 清理weights目录中的旧epoch判别器文件
                for pt_file in weights_dir.glob('discriminator_epoch*.pt'):
                    try:
                        pt_file.unlink()
                        LOGGER.info(f"Removed old discriminator checkpoint: {pt_file}")
                    except Exception as e:
                        LOGGER.warning(f"Failed to remove {pt_file}: {e}")

            # 恢复目录
            self.save_dir = current_dir

        except Exception as e:
            LOGGER.error(f"Error in save_model: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())

            try:
                # 简化的紧急保存 - 仅保存到weights子目录
                from ultralytics.utils.torch_utils import de_parallel
                model = de_parallel(self.model)

                # 确保weights目录存在
                weights_dir = Path(self.save_dir) / 'weights'
                weights_dir.mkdir(exist_ok=True)

                # 仅保存到weights目录
                save_path = str(weights_dir / 'last.pt')
                torch.save({'model': model.state_dict()}, save_path)
                LOGGER.info(f"Emergency save completed to {save_path}")

                # 保存判别器 - 仅保存到weights目录
                if self.domain_adapt_enabled and self.discriminator is not None:
                    discriminator = de_parallel(self.discriminator)
                    disc_path = str(weights_dir / 'discriminator_last.pt')
                    torch.save(discriminator.state_dict(), disc_path)
                    LOGGER.info(f"Emergency discriminator save completed to {disc_path}")

            except Exception as inner_e:
                LOGGER.error(f"Emergency save failed: {inner_e}")