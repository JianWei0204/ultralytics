# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
MFDAM训练器 (Multi-scale Fusion Domain Adaptation Module Trainer)
专门用于Day→Rain域适应的YOLOv8训练器

主要功能：
1. 集成多尺度特征提取
2. 域对抗训练
3. 特征融合和对齐
4. 训练过程监控和可视化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import math
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from tqdm import tqdm

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.loss import v8DetectionLoss

try:
    from .multi_scale_extractor import MultiScaleFeatureExtractor, DomainInvariantHead
    from .domain_losses import MFDAMLoss
    from .dual_dataloader import DualDomainDataLoader
    from .config import MFDAMConfig
except ImportError:
    # 处理相对导入问题
    from mfda.multi_scale_extractor import MultiScaleFeatureExtractor, DomainInvariantHead
    from mfda.domain_losses import MFDAMLoss
    from mfda.dual_dataloader import DualDomainDataLoader
    from mfda.config import MFDAMConfig


class MFDAMTrainer(DetectionTrainer):
    """
    MFDAM训练器
    
    扩展标准YOLOv8检测训练器，添加多尺度融合域适应功能
    """
    
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """
        初始化MFDAM训练器
        
        Args:
            cfg: 配置对象
            overrides: 覆盖参数
            _callbacks: 回调函数
        """
        super().__init__(cfg, overrides, _callbacks)
        
        # MFDAM特定组件
        self.mfdam_config = None
        self.feature_extractor = None
        self.domain_loss_fn = None
        self.dual_dataloader = None
        self.domain_invariant_head = None
        
        # 优化器
        self.optimizer_da = None  # 域适应优化器
        
        # 训练状态
        self.da_losses = {}
        self.da_metrics = {}
        
        LOGGER.info(f"{colorstr('cyan', 'bold', 'MFDAM Trainer')} 初始化完成")
        
    def setup_mfdam(self, 
                    target_data: str,
                    mfdam_config: Optional[MFDAMConfig] = None):
        """
        设置MFDAM组件
        
        Args:
            target_data: 目标域数据配置路径
            mfdam_config: MFDAM配置对象
        """
        # 使用默认配置或提供的配置
        self.mfdam_config = mfdam_config or MFDAMConfig()
        
        # 初始化多尺度特征提取器
        self.feature_extractor = MultiScaleFeatureExtractor(
            feature_dims=self.mfdam_config.feature_dims,
            extract_layers=self.mfdam_config.extract_layers
        )
        
        # 注册特征提取钩子
        if self.model is not None:
            self.feature_extractor.register_hooks(self.model)
            
        # 初始化域不变特征学习头
        self.domain_invariant_head = DomainInvariantHead(
            input_dim=self.mfdam_config.fusion_dim,
            hidden_dim=self.mfdam_config.da_hidden_dim
        )
        
        # 初始化域适应损失函数
        self.domain_loss_fn = MFDAMLoss(
            domain_weight=self.mfdam_config.domain_weight,
            consistency_weight=self.mfdam_config.consistency_weight,
            alignment_weight=self.mfdam_config.alignment_weight,
            weather_weight=self.mfdam_config.weather_weight
        )
        
        # 创建双域数据加载器
        self.dual_dataloader = DualDomainDataLoader(
            source_cfg=self.args.data,  # 使用标准训练数据作为源域
            target_cfg=target_data,
            batch_size=self.args.batch,
            img_size=self.args.imgsz,
            workers=self.args.workers,
            domain_balance=self.mfdam_config.domain_balance
        )
        
        # 移动到设备
        if hasattr(self, 'device'):
            self.domain_invariant_head.to(self.device)
            self.domain_loss_fn.to(self.device)
            
        LOGGER.info(f"{colorstr('green', 'MFDAM组件设置完成')}")
        
    def _setup_train(self, world_size):
        """
        设置训练环境
        
        Args:
            world_size: 世界大小（分布式训练）
        """
        # 调用父类设置
        super()._setup_train(world_size)
        
        # 设置域适应优化器
        if self.domain_invariant_head is not None and self.domain_loss_fn is not None:
            da_params = list(self.domain_invariant_head.parameters()) + \
                       list(self.domain_loss_fn.parameters())
            
            self.optimizer_da = torch.optim.Adam(
                da_params,
                lr=self.mfdam_config.da_lr,
                weight_decay=self.mfdam_config.da_weight_decay
            )
            
        LOGGER.info(f"{colorstr('green', 'MFDAM训练环境设置完成')}")
        
    def _do_train(self, world_size=1):
        """
        执行MFDAM训练
        
        Args:
            world_size: 世界大小
        """
        if self.dual_dataloader is None:
            LOGGER.warning("双域数据加载器未设置，使用标准训练")
            return super()._do_train(world_size)
            
        # 训练循环
        for epoch in range(self.epochs):
            self.epoch = epoch
            
            # 设置模型为训练模式
            self.model.train()
            if self.domain_invariant_head is not None:
                self.domain_invariant_head.train()
                
            # 重置统计信息
            self.da_losses = {}
            self.da_metrics = {}
            
            # 进度条
            pbar = enumerate(self.dual_dataloader)
            if RANK in (-1, 0):
                pbar = tqdm(pbar, total=len(self.dual_dataloader), desc=f"Epoch {epoch+1}/{self.epochs}")
                
            for batch_idx, batch in pbar:
                # 执行一步训练
                self._train_step(batch, batch_idx)
                
                # 更新进度条
                if RANK in (-1, 0) and batch_idx % 10 == 0:
                    pbar.set_postfix(self._get_progress_info())
                    
            # 验证和保存
            self._end_epoch()
            
    def _train_step(self, batch: Dict, batch_idx: int):
        """
        执行一步训练
        
        Args:
            batch: 批次数据
            batch_idx: 批次索引
        """
        # 移动数据到设备
        imgs = batch['img'].to(self.device, non_blocking=True)
        domain_labels = batch['domain_labels'].to(self.device, non_blocking=True)
        weather_labels = batch['weather_labels'].to(self.device, non_blocking=True)
        
        # 分离源域和目标域数据
        source_mask = domain_labels == 0
        target_mask = domain_labels == 1
        
        source_imgs = imgs[source_mask]
        target_imgs = imgs[target_mask]
        
        # 前向传播
        with torch.cuda.amp.autocast(enabled=self.amp):
            # 标准检测损失（仅用源域数据）
            if source_imgs.size(0) > 0:
                # 准备源域数据用于检测训练
                source_batch = self._prepare_detection_batch(batch, source_mask)
                preds = self.model(source_imgs)
                self.loss, self.loss_items = self.criterion(preds, source_batch)
            else:
                self.loss = torch.tensor(0.0, device=self.device)
                self.loss_items = torch.zeros(3, device=self.device)
            
            # 多尺度特征提取
            _ = self.model(imgs)  # 触发特征提取钩子
            feature_results = self.feature_extractor()
            
            # 域适应损失
            da_loss = torch.tensor(0.0, device=self.device)
            if feature_results and 'fused_features' in feature_results:
                fused_features = feature_results['fused_features']
                
                if fused_features is not None and source_imgs.size(0) > 0 and target_imgs.size(0) > 0:
                    source_features = fused_features[source_mask]
                    target_features = fused_features[target_mask]
                    
                    # 计算域适应损失
                    da_results = self.domain_loss_fn(
                        source_features=source_features,
                        target_features=target_features,
                        weather_labels=weather_labels
                    )
                    
                    da_loss = da_results['total_da_loss']
                    self.da_losses.update(da_results)
                    
            # 总损失
            total_loss = self.loss + self.mfdam_config.da_loss_weight * da_loss
            
        # 反向传播
        self.scaler.scale(total_loss).backward()
        
        # 更新参数
        if batch_idx % self.accumulate == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # 更新域适应参数
            if self.optimizer_da is not None and da_loss > 0:
                self.optimizer_da.step()
                self.optimizer_da.zero_grad()
                
    def _prepare_detection_batch(self, batch: Dict, mask: torch.Tensor) -> Dict:
        """
        准备检测训练批次
        
        Args:
            batch: 原始批次
            mask: 源域掩码
            
        Returns:
            Dict: 检测训练批次
        """
        detection_batch = {}
        
        # 复制需要的字段
        for key in ['bboxes', 'cls', 'batch_idx']:
            if key in batch:
                data = batch[key]
                if hasattr(data, 'shape') and len(data.shape) > 0:
                    # 过滤源域数据
                    if key == 'batch_idx':
                        # 重新映射批次索引
                        source_indices = torch.where(mask)[0]
                        batch_mask = torch.isin(data, source_indices)
                        filtered_data = data[batch_mask]
                        # 重新编号
                        for i, old_idx in enumerate(source_indices):
                            filtered_data[filtered_data == old_idx] = i
                        detection_batch[key] = filtered_data
                    else:
                        # 其他数据直接过滤
                        if data.size(0) > 0:
                            batch_indices = data[:, 0].long()
                            source_indices = torch.where(mask)[0]
                            batch_mask = torch.isin(batch_indices, source_indices)
                            detection_batch[key] = data[batch_mask]
                        else:
                            detection_batch[key] = data
                else:
                    detection_batch[key] = data
                    
        return detection_batch
        
    def _get_progress_info(self) -> Dict:
        """
        获取训练进度信息
        
        Returns:
            Dict: 进度信息
        """
        info = {}
        
        # 检测损失
        if hasattr(self, 'loss_items'):
            info['det_loss'] = f"{self.loss_items.mean():.4f}"
            
        # 域适应损失
        if 'total_da_loss' in self.da_losses:
            info['da_loss'] = f"{self.da_losses['total_da_loss']:.4f}"
            
        if 'domain_accuracy' in self.da_losses:
            info['dom_acc'] = f"{self.da_losses['domain_accuracy']:.3f}"
            
        return info
        
    def _end_epoch(self):
        """结束一个epoch的处理"""
        # 学习率调度
        if self.scheduler is not None:
            self.scheduler.step()
            
        if self.optimizer_da is not None and hasattr(self, 'da_scheduler'):
            self.da_scheduler.step()
            
        # 验证
        if self.epoch % self.args.val_period == 0:
            self.validate()
            
        # 保存检查点
        if self.epoch % self.args.save_period == 0:
            self.save_model()
            
    def validate(self):
        """验证模型性能"""
        # 调用父类验证
        results = super().validate()
        
        # 添加域适应相关的验证指标
        if self.dual_dataloader is not None:
            # 可以在这里添加域适应特定的验证逻辑
            pass
            
        return results
        
    def save_model(self):
        """保存模型"""
        # 调用父类保存
        super().save_model()
        
        # 保存MFDAM特定组件
        if self.domain_invariant_head is not None:
            save_path = self.save_dir / f'mfdam_components_epoch_{self.epoch}.pt'
            torch.save({
                'domain_head': self.domain_invariant_head.state_dict(),
                'domain_loss': self.domain_loss_fn.state_dict() if self.domain_loss_fn else None,
                'optimizer_da': self.optimizer_da.state_dict() if self.optimizer_da else None,
                'config': self.mfdam_config.__dict__ if self.mfdam_config else None
            }, save_path)
            
    def load_mfdam_components(self, checkpoint_path: str):
        """
        加载MFDAM组件
        
        Args:
            checkpoint_path: 检查点路径
        """
        if not os.path.exists(checkpoint_path):
            LOGGER.warning(f"MFDAM检查点不存在: {checkpoint_path}")
            return
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.domain_invariant_head is not None and 'domain_head' in checkpoint:
            self.domain_invariant_head.load_state_dict(checkpoint['domain_head'])
            
        if self.domain_loss_fn is not None and 'domain_loss' in checkpoint:
            self.domain_loss_fn.load_state_dict(checkpoint['domain_loss'])
            
        if self.optimizer_da is not None and 'optimizer_da' in checkpoint:
            self.optimizer_da.load_state_dict(checkpoint['optimizer_da'])
            
        LOGGER.info(f"MFDAM组件加载完成: {checkpoint_path}")
        
    def __del__(self):
        """析构函数"""
        # 清理特征提取钩子
        if hasattr(self, 'feature_extractor') and self.feature_extractor is not None:
            self.feature_extractor.clear_hooks()


def train_mfdam(model_cfg: str,
                source_data: str,
                target_data: str,
                epochs: int = 100,
                batch_size: int = 16,
                imgsz: int = 640,
                **kwargs):
    """
    MFDAM训练便捷函数
    
    Args:
        model_cfg: 模型配置文件路径
        source_data: 源域数据配置路径
        target_data: 目标域数据配置路径
        epochs: 训练轮数
        batch_size: 批次大小
        imgsz: 图像尺寸
        **kwargs: 其他参数
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    
    # 创建训练参数
    train_args = {
        'model': model_cfg,
        'data': source_data,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        **kwargs
    }
    
    # 创建配置
    cfg = get_cfg(DEFAULT_CFG, {})
    
    # 创建训练器
    trainer = MFDAMTrainer(cfg=cfg, overrides=train_args)
    
    # 设置MFDAM
    trainer.setup_mfdam(target_data=target_data)
    
    # 开始训练
    trainer.train()
    
    return trainer