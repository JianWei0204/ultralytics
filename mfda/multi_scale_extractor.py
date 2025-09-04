# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
多尺度特征提取器 (Multi-scale Feature Extractor)
用于Day→Rain域适应的多尺度特征提取和融合

主要功能：
1. 从YOLOv8骨干网络提取多尺度特征
2. 跨尺度特征融合
3. 域不变特征学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class MultiScaleFeatureExtractor(nn.Module):
    """
    多尺度特征提取器
    
    从YOLOv8的不同层提取多尺度特征，用于域适应训练
    """
    
    def __init__(self, 
                 feature_dims: List[int] = [256, 512, 1024],
                 extract_layers: List[str] = ['backbone.9', 'backbone.12', 'backbone.15']):
        """
        初始化多尺度特征提取器
        
        Args:
            feature_dims: 各尺度特征的维度列表
            extract_layers: 要提取特征的层名称列表
        """
        super(MultiScaleFeatureExtractor, self).__init__()
        
        self.feature_dims = feature_dims
        self.extract_layers = extract_layers
        self.extracted_features = {}
        self.hooks = []
        
        # 特征适配层，将不同维度的特征转换为统一维度
        self.feature_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for dim in feature_dims
        ])
        
        # 多尺度融合模块
        self.fusion_module = FusionModule(256, len(feature_dims))
        
    def register_hooks(self, model):
        """
        为指定的层注册前向钩子函数
        
        Args:
            model: YOLOv8模型
        """
        self.clear_hooks()
        
        for name, module in model.named_modules():
            if name in self.extract_layers:
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)
                
    def _hook_fn(self, layer_name):
        """
        创建钩子函数
        
        Args:
            layer_name: 层名称
        """
        def hook(module, input, output):
            self.extracted_features[layer_name] = output
        return hook
        
    def clear_hooks(self):
        """清除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.extracted_features = {}
        
    def forward(self, x=None):
        """
        前向传播
        
        Args:
            x: 输入张量 (可选，主要通过钩子获取特征)
            
        Returns:
            Dict: 包含原始特征、适配特征和融合特征的字典
        """
        if not self.extracted_features:
            return {}
            
        # 获取提取的特征
        raw_features = []
        for layer_name in self.extract_layers:
            if layer_name in self.extracted_features:
                raw_features.append(self.extracted_features[layer_name])
                
        if not raw_features:
            return {}
            
        # 特征适配
        adapted_features = []
        for i, feature in enumerate(raw_features):
            if i < len(self.feature_adapters):
                adapted = self.feature_adapters[i](feature)
                adapted_features.append(adapted)
                
        # 多尺度融合
        if adapted_features:
            fused_features = self.fusion_module(adapted_features)
        else:
            fused_features = None
            
        return {
            'raw_features': raw_features,
            'adapted_features': adapted_features,
            'fused_features': fused_features
        }


class FusionModule(nn.Module):
    """
    特征融合模块
    
    将多个尺度的特征进行融合，生成域不变的表示
    """
    
    def __init__(self, feature_dim: int = 256, num_scales: int = 3):
        """
        初始化融合模块
        
        Args:
            feature_dim: 特征维度
            num_scales: 尺度数量
        """
        super(FusionModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_scales = num_scales
        
        # 注意力权重计算
        self.attention_conv = nn.Conv2d(feature_dim * num_scales, num_scales, 1)
        
        # 特征融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim * num_scales, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim)
        )
        
        # 残差连接
        self.residual_conv = nn.Conv2d(feature_dim, feature_dim, 1, bias=False)
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 多尺度特征列表
            
        Returns:
            torch.Tensor: 融合后的特征
        """
        if not features:
            return None
            
        # 将所有特征调整到相同的空间尺寸
        target_size = features[0].shape[2:]  # 使用第一个特征的尺寸作为目标
        
        resized_features = []
        for feature in features:
            if feature.shape[2:] != target_size:
                # 使用双线性插值调整尺寸
                resized = F.interpolate(feature, size=target_size, mode='bilinear', align_corners=False)
            else:
                resized = feature
            resized_features.append(resized)
            
        # 拼接所有特征
        concatenated = torch.cat(resized_features, dim=1)  # [B, C*N, H, W]
        
        # 计算注意力权重
        attention_weights = torch.softmax(self.attention_conv(concatenated), dim=1)  # [B, N, H, W]
        
        # 应用注意力权重
        weighted_features = []
        for i, feature in enumerate(resized_features):
            weight = attention_weights[:, i:i+1, :, :]  # [B, 1, H, W]
            weighted = feature * weight
            weighted_features.append(weighted)
            
        # 加权求和
        weighted_sum = torch.stack(weighted_features, dim=0).sum(dim=0)  # [B, C, H, W]
        
        # 特征融合
        fused = self.fusion_conv(concatenated)
        
        # 残差连接
        residual = self.residual_conv(weighted_sum)
        output = fused + residual
        
        return F.relu(output, inplace=True)


class DomainInvariantHead(nn.Module):
    """
    域不变特征学习头
    
    用于学习域不变的特征表示
    """
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        """
        初始化域不变特征学习头
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
        """
        super(DomainInvariantHead, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C, H, W]
            
        Returns:
            torch.Tensor: 域不变特征 [B, hidden_dim//2]
        """
        return self.feature_extractor(x)