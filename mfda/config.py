# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
MFDAM配置管理 (MFDAM Configuration Management)
用于管理多尺度融合域适应模块的所有配置参数

主要功能：
1. 训练超参数配置
2. 网络结构配置
3. 损失函数权重配置
4. 数据加载配置
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import yaml
import json
from pathlib import Path


@dataclass
class MFDAMConfig:
    """
    MFDAM配置类
    
    包含所有MFDAM相关的配置参数
    """
    
    # 特征提取配置
    feature_dims: List[int] = None  # [256, 512, 1024]
    extract_layers: List[str] = None  # ['backbone.9', 'backbone.12', 'backbone.15']
    fusion_dim: int = 256
    
    # 域适应配置
    da_lr: float = 0.001  # 域适应学习率
    da_weight_decay: float = 1e-4
    da_hidden_dim: int = 128
    
    # 损失函数权重
    domain_weight: float = 1.0  # 域对抗损失权重
    consistency_weight: float = 0.5  # 一致性损失权重
    alignment_weight: float = 0.3  # 特征对齐损失权重
    weather_weight: float = 0.7  # 天气对抗损失权重
    da_loss_weight: float = 0.1  # 整体域适应损失权重
    
    # 数据配置
    domain_balance: float = 0.5  # 域平衡比例 (0.5表示源域和目标域各占50%)
    
    # 梯度反转层配置
    grl_weight: float = 1.0
    grl_schedule: bool = True  # 是否使用动态GRL权重
    
    # 训练配置
    warmup_epochs: int = 5  # 预热轮数
    da_start_epoch: int = 10  # 域适应开始轮数
    
    def __post_init__(self):
        """初始化后处理"""
        if self.feature_dims is None:
            self.feature_dims = [256, 512, 1024]
            
        if self.extract_layers is None:
            self.extract_layers = ['backbone.9', 'backbone.12', 'backbone.15']
            
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'MFDAMConfig':
        """
        从YAML文件加载配置
        
        Args:
            yaml_path: YAML配置文件路径
            
        Returns:
            MFDAMConfig: 配置对象
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            
        return cls(**config_dict.get('mfdam', {}))
        
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'MFDAMConfig':
        """
        从字典创建配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            MFDAMConfig: 配置对象
        """
        return cls(**config_dict)
        
    def to_yaml(self, yaml_path: Union[str, Path]):
        """
        保存配置到YAML文件
        
        Args:
            yaml_path: 保存路径
        """
        config_dict = {'mfdam': self.__dict__}
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
    def to_dict(self) -> Dict:
        """
        转换为字典
        
        Returns:
            Dict: 配置字典
        """
        return self.__dict__.copy()
        
    def update(self, **kwargs):
        """
        更新配置参数
        
        Args:
            **kwargs: 要更新的参数
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def get_grl_weight(self, epoch: int, total_epochs: int) -> float:
        """
        计算动态GRL权重
        
        Args:
            epoch: 当前轮数
            total_epochs: 总轮数
            
        Returns:
            float: GRL权重
        """
        if not self.grl_schedule:
            return self.grl_weight
            
        # 使用论文中的动态权重公式: λp = 2/(1+exp(-10*p)) - 1
        # 其中 p = epoch / total_epochs
        p = epoch / total_epochs
        lambda_p = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
        return self.grl_weight * lambda_p


# 预定义配置
DEFAULT_MFDAM_CONFIG = MFDAMConfig()

DAY_RAIN_CONFIG = MFDAMConfig(
    # 专门为Day→Rain适应优化的配置
    feature_dims=[256, 512, 1024],
    extract_layers=['backbone.9', 'backbone.12', 'backbone.15'],
    fusion_dim=256,
    
    # 域适应参数
    da_lr=0.0005,
    da_weight_decay=1e-4,
    da_hidden_dim=128,
    
    # 损失权重 - 为雨天适应调整
    domain_weight=1.2,
    consistency_weight=0.6,
    alignment_weight=0.4,
    weather_weight=0.8,
    da_loss_weight=0.15,
    
    # 数据配置
    domain_balance=0.6,  # 稍微偏向目标域(雨天)
    
    # GRL配置
    grl_weight=1.0,
    grl_schedule=True,
    
    # 训练配置
    warmup_epochs=5,
    da_start_epoch=8
)

LIGHTWEIGHT_CONFIG = MFDAMConfig(
    # 轻量级配置，适用于资源受限环境
    feature_dims=[128, 256, 512],
    extract_layers=['backbone.6', 'backbone.9', 'backbone.12'],
    fusion_dim=128,
    
    # 较小的网络
    da_hidden_dim=64,
    
    # 降低损失权重
    domain_weight=0.8,
    consistency_weight=0.3,
    alignment_weight=0.2,
    weather_weight=0.5,
    da_loss_weight=0.08,
)


def create_day_rain_dataset_configs(
    source_train_path: str,
    source_val_path: str,
    target_train_path: str,
    target_val_path: str,
    num_classes: int = 80,
    class_names: Optional[List[str]] = None
) -> tuple:
    """
    创建Day→Rain域适应的数据集配置文件
    
    Args:
        source_train_path: 源域训练数据路径
        source_val_path: 源域验证数据路径
        target_train_path: 目标域训练数据路径
        target_val_path: 目标域验证数据路径
        num_classes: 类别数量
        class_names: 类别名称列表
        
    Returns:
        tuple: (源域配置, 目标域配置)
    """
    if class_names is None:
        # COCO数据集的默认类别
        class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ][:num_classes]
    
    # 源域配置 (Day)
    source_config = {
        'path': str(Path(source_train_path).parent),
        'train': source_train_path,
        'val': source_val_path,
        'nc': num_classes,
        'names': class_names,
        'domain': 'source',
        'weather': 'day',
        'description': '源域数据集 - 白天晴朗天气条件'
    }
    
    # 目标域配置 (Rain)
    target_config = {
        'path': str(Path(target_train_path).parent),
        'train': target_train_path,
        'val': target_val_path,
        'nc': num_classes,
        'names': class_names,
        'domain': 'target',
        'weather': 'rain',
        'description': '目标域数据集 - 雨天天气条件'
    }
    
    return source_config, target_config


def save_dataset_configs(
    source_config: Dict,
    target_config: Dict,
    save_dir: Union[str, Path]
):
    """
    保存数据集配置文件
    
    Args:
        source_config: 源域配置
        target_config: 目标域配置
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存源域配置
    source_path = save_dir / 'source_day.yaml'
    with open(source_path, 'w', encoding='utf-8') as f:
        yaml.dump(source_config, f, default_flow_style=False, allow_unicode=True)
        
    # 保存目标域配置
    target_path = save_dir / 'target_rain.yaml'
    with open(target_path, 'w', encoding='utf-8') as f:
        yaml.dump(target_config, f, default_flow_style=False, allow_unicode=True)
        
    return source_path, target_path


def create_mfdam_model_config(
    base_model: str = 'yolov8n.yaml',
    num_classes: int = 80,
    save_path: Optional[Union[str, Path]] = None
) -> Dict:
    """
    创建MFDAM模型配置
    
    Args:
        base_model: 基础模型配置
        num_classes: 类别数量
        save_path: 保存路径
        
    Returns:
        Dict: 模型配置
    """
    # 基础YOLOv8配置
    config = {
        # Model
        'nc': num_classes,  # number of classes
        'depth_multiple': 0.33,  # model depth multiple
        'width_multiple': 0.25,  # layer channel multiple
        'max_channels': 1024,
        
        # anchors
        'anchors': 3,
        
        # YOLOv8.0n backbone
        'backbone': [
            [-1, 1, 'Conv', [64, 3, 2]],  # 0-P1/2
            [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
            [-1, 3, 'C2f', [128, True]],
            [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
            [-1, 6, 'C2f', [256, True]],
            [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
            [-1, 6, 'C2f', [512, True]],
            [-1, 1, 'Conv', [1024, 3, 2]],  # 7-P5/32
            [-1, 3, 'C2f', [1024, True]],
            [-1, 1, 'SPPF', [1024, 5]],  # 9
        ],
        
        # YOLOv8.0n head
        'head': [
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
            [-1, 3, 'C2f', [512]],  # 12
            
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
            [-1, 3, 'C2f', [256]],  # 15 (P3/8-small)
            
            [-1, 1, 'Conv', [256, 3, 2]],
            [[-1, 12], 1, 'Concat', [1]],  # cat head P4
            [-1, 3, 'C2f', [512]],  # 18 (P4/16-medium)
            
            [-1, 1, 'Conv', [512, 3, 2]],
            [[-1, 9], 1, 'Concat', [1]],  # cat head P5
            [-1, 3, 'C2f', [1024]],  # 21 (P5/32-large)
            
            [[15, 18, 21], 1, 'Detect', [num_classes]],  # Detect(P3, P4, P5)
        ],
        
        # MFDAM specific configurations
        'mfdam': {
            'enabled': True,
            'feature_extract_layers': [9, 12, 15],  # 对应backbone.9, head.12, head.15
            'fusion_module': {
                'input_dims': [1024, 512, 256],
                'output_dim': 256
            },
            'domain_adaptation': {
                'hidden_dim': 128,
                'num_domains': 2,  # source, target
                'num_weather_types': 2  # day, rain
            }
        }
    }
    
    # 保存配置
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
    return config


# 导入时的默认配置
import math  # 为get_grl_weight方法提供math模块