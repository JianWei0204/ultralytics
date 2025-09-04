# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
权重加载器 (Weight Loader)
用于加载和管理YOLOv8预训练权重

主要功能：
1. 加载预训练YOLOv8权重
2. 权重兼容性检查
3. 层级权重映射
4. 域适应权重初始化
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Union
import logging
import time

from ultralytics.utils import LOGGER


def load_pretrained_weights(model: nn.Module, 
                          weights_path: Union[str, Path],
                          strict: bool = False,
                          exclude_keys: Optional[list] = None) -> Dict:
    """
    加载预训练权重到模型
    
    Args:
        model: 目标模型
        weights_path: 权重文件路径
        strict: 是否严格匹配键名
        exclude_keys: 要排除的键名列表
        
    Returns:
        Dict: 加载结果信息
    """
    weights_path = Path(weights_path)
    
    if not weights_path.exists():
        LOGGER.warning(f"权重文件不存在: {weights_path}")
        return {"status": "failed", "reason": "file_not_found"}
    
    try:
        # 加载权重文件
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # 提取模型状态字典
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict()
            
        # 获取模型状态字典
        model_dict = model.state_dict()
        
        # 过滤不兼容的键
        exclude_keys = exclude_keys or []
        filtered_dict = {}
        
        for k, v in state_dict.items():
            # 跳过排除的键
            if any(exclude_key in k for exclude_key in exclude_keys):
                continue
                
            # 检查键是否存在于目标模型中
            if k in model_dict:
                # 检查形状是否匹配
                if v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    LOGGER.warning(f"形状不匹配: {k} {v.shape} vs {model_dict[k].shape}")
            else:
                LOGGER.debug(f"键不存在于目标模型: {k}")
                
        # 更新模型字典
        model_dict.update(filtered_dict)
        
        # 加载权重
        missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=strict)
        
        # 统计信息
        loaded_keys = len(filtered_dict)
        total_keys = len(state_dict)
        
        LOGGER.info(f"权重加载完成: {loaded_keys}/{total_keys} 层")
        
        if missing_keys:
            LOGGER.warning(f"缺失的键: {len(missing_keys)} 个")
            for key in missing_keys[:5]:  # 只显示前5个
                LOGGER.debug(f"  - {key}")
                
        if unexpected_keys:
            LOGGER.warning(f"意外的键: {len(unexpected_keys)} 个")
            for key in unexpected_keys[:5]:  # 只显示前5个
                LOGGER.debug(f"  - {key}")
                
        return {
            "status": "success",
            "loaded_keys": loaded_keys,
            "total_keys": total_keys,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys
        }
        
    except Exception as e:
        LOGGER.error(f"权重加载失败: {e}")
        return {"status": "failed", "reason": str(e)}


def load_yolov8_backbone_weights(model: nn.Module, 
                                weights_path: Union[str, Path]) -> Dict:
    """
    仅加载YOLOv8骨干网络权重
    
    Args:
        model: 目标模型
        weights_path: 权重文件路径
        
    Returns:
        Dict: 加载结果
    """
    # 只加载backbone相关的权重
    exclude_keys = ['head', 'detect', 'anchor', 'loss']
    
    return load_pretrained_weights(
        model=model,
        weights_path=weights_path,
        strict=False,
        exclude_keys=exclude_keys
    )


def initialize_domain_adaptation_weights(model: nn.Module, 
                                       feature_dim: int = 256) -> None:
    """
    初始化域适应相关的权重
    
    Args:
        model: 模型
        feature_dim: 特征维度
    """
    # 初始化域适应相关层的权重
    for name, module in model.named_modules():
        if 'domain' in name.lower() or 'discriminator' in name.lower():
            if isinstance(module, nn.Linear):
                # Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, nn.Conv2d):
                # He初始化
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
    LOGGER.info("域适应权重初始化完成")


def save_mfdam_checkpoint(model: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         epoch: int,
                         save_path: Union[str, Path],
                         **kwargs) -> None:
    """
    保存MFDAM检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮数
        save_path: 保存路径
        **kwargs: 其他要保存的信息
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        **kwargs
    }
    
    torch.save(checkpoint, save_path)
    LOGGER.info(f"检查点已保存: {save_path}")


def load_mfdam_checkpoint(checkpoint_path: Union[str, Path],
                         model: Optional[nn.Module] = None,
                         optimizer: Optional[torch.optim.Optimizer] = None,
                         device: str = 'cpu') -> Dict:
    """
    加载MFDAM检查点
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型 (可选)
        optimizer: 优化器 (可选)
        device: 设备
        
    Returns:
        Dict: 检查点信息
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型权重
    if model is not None and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        LOGGER.info("模型权重加载完成")
        
    # 加载优化器状态
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        LOGGER.info("优化器状态加载完成")
        
    return checkpoint


class WeightManager:
    """
    权重管理器
    
    用于管理MFDAM训练过程中的权重加载、保存和更新
    """
    
    def __init__(self, save_dir: Union[str, Path]):
        """
        初始化权重管理器
        
        Args:
            save_dir: 保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def save_best_weights(self, 
                         model: nn.Module,
                         metric_value: float,
                         metric_name: str = 'mAP') -> None:
        """
        保存最佳权重
        
        Args:
            model: 模型
            metric_value: 指标值
            metric_name: 指标名称
        """
        save_path = self.save_dir / f'best_{metric_name}.pt'
        
        checkpoint = {
            'model': model.state_dict(),
            f'best_{metric_name}': metric_value,
            'timestamp': torch.tensor(time.time())
        }
        
        torch.save(checkpoint, save_path)
        LOGGER.info(f"最佳{metric_name}权重已保存: {metric_value:.4f}")
        
    def save_epoch_weights(self,
                          model: nn.Module,
                          epoch: int,
                          **kwargs) -> None:
        """
        保存epoch权重
        
        Args:
            model: 模型
            epoch: 轮数
            **kwargs: 其他信息
        """
        save_path = self.save_dir / f'epoch_{epoch}.pt'
        
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            **kwargs
        }
        
        torch.save(checkpoint, save_path)
        
    def load_latest_weights(self, model: nn.Module) -> Optional[int]:
        """
        加载最新的权重
        
        Args:
            model: 模型
            
        Returns:
            Optional[int]: 最后的epoch数，如果没有找到则返回None
        """
        # 查找所有epoch权重文件
        epoch_files = list(self.save_dir.glob('epoch_*.pt'))
        
        if not epoch_files:
            LOGGER.info("未找到epoch权重文件")
            return None
            
        # 找到最大的epoch数
        latest_epoch = max([
            int(f.stem.split('_')[1]) for f in epoch_files
        ])
        
        latest_file = self.save_dir / f'epoch_{latest_epoch}.pt'
        
        # 加载权重
        checkpoint = torch.load(latest_file, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        
        LOGGER.info(f"已加载最新权重: epoch {latest_epoch}")
        return latest_epoch