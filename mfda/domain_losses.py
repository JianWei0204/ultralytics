# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
域对抗损失函数 (Domain Adversarial Loss Functions)
用于Day→Rain域适应的损失函数集合

主要功能：
1. 域分类损失
2. 一致性损失
3. 对抗损失
4. 特征对齐损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class DomainAdversarialLoss(nn.Module):
    """
    域对抗损失
    
    通过梯度反转层实现域对抗训练，使特征提取器学习域不变特征
    """
    
    def __init__(self, 
                 input_dim: int = 256,
                 hidden_dim: int = 128,
                 grl_weight: float = 1.0):
        """
        初始化域对抗损失
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度  
            grl_weight: 梯度反转层权重
        """
        super(DomainAdversarialLoss, self).__init__()
        
        self.grl_weight = grl_weight
        
        # 域分类器
        self.domain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)  # 二分类：源域(0) vs 目标域(1)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def gradient_reverse_layer(self, x: torch.Tensor) -> torch.Tensor:
        """
        梯度反转层
        
        Args:
            x: 输入特征
            
        Returns:
            torch.Tensor: 经过梯度反转的特征
        """
        return GradientReverseFunction.apply(x, self.grl_weight)
        
    def forward(self, 
                source_features: torch.Tensor,
                target_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            source_features: 源域特征 [B, C, H, W]
            target_features: 目标域特征 [B, C, H, W]
            
        Returns:
            Dict: 包含各种损失的字典
        """
        # 合并源域和目标域特征
        all_features = torch.cat([source_features, target_features], dim=0)
        
        # 通过梯度反转层
        reversed_features = self.gradient_reverse_layer(all_features)
        
        # 域分类
        domain_predictions = self.domain_classifier(reversed_features)
        
        # 创建域标签
        batch_size = source_features.size(0)
        source_labels = torch.zeros(batch_size, 1, device=source_features.device)
        target_labels = torch.ones(batch_size, 1, device=target_features.device)
        domain_labels = torch.cat([source_labels, target_labels], dim=0)
        
        # 计算域分类损失
        domain_loss = F.binary_cross_entropy_with_logits(domain_predictions, domain_labels)
        
        # 计算域分类准确率
        with torch.no_grad():
            predictions = torch.sigmoid(domain_predictions) > 0.5
            accuracy = (predictions.float() == domain_labels).float().mean()
            
        return {
            'domain_loss': domain_loss,
            'domain_accuracy': accuracy,
            'domain_predictions': torch.sigmoid(domain_predictions)
        }


class ConsistencyLoss(nn.Module):
    """
    一致性损失
    
    确保模型在不同域上的预测保持一致性
    """
    
    def __init__(self, 
                 temperature: float = 0.5,
                 consistency_weight: float = 1.0):
        """
        初始化一致性损失
        
        Args:
            temperature: 软标签温度参数
            consistency_weight: 一致性损失权重
        """
        super(ConsistencyLoss, self).__init__()
        
        self.temperature = temperature
        self.consistency_weight = consistency_weight
        
    def forward(self, 
                source_predictions: torch.Tensor,
                target_predictions: torch.Tensor) -> torch.Tensor:
        """
        计算一致性损失
        
        Args:
            source_predictions: 源域预测 [B, num_classes, ...]
            target_predictions: 目标域预测 [B, num_classes, ...]
            
        Returns:
            torch.Tensor: 一致性损失
        """
        # 将预测转换为概率分布
        source_probs = F.softmax(source_predictions / self.temperature, dim=1)
        target_probs = F.softmax(target_predictions / self.temperature, dim=1)
        
        # 计算KL散度
        kl_loss = F.kl_div(
            F.log_softmax(target_predictions / self.temperature, dim=1),
            source_probs,
            reduction='batchmean'
        )
        
        return self.consistency_weight * kl_loss


class FeatureAlignmentLoss(nn.Module):
    """
    特征对齐损失
    
    使源域和目标域的特征分布对齐
    """
    
    def __init__(self, alignment_weight: float = 1.0):
        """
        初始化特征对齐损失
        
        Args:
            alignment_weight: 对齐损失权重
        """
        super(FeatureAlignmentLoss, self).__init__()
        
        self.alignment_weight = alignment_weight
        
    def mmd_loss(self, 
                 source_features: torch.Tensor,
                 target_features: torch.Tensor,
                 kernel_type: str = 'gaussian') -> torch.Tensor:
        """
        最大均值差异(MMD)损失
        
        Args:
            source_features: 源域特征
            target_features: 目标域特征
            kernel_type: 核函数类型
            
        Returns:
            torch.Tensor: MMD损失
        """
        def gaussian_kernel(x, y, sigma=1.0):
            """高斯核函数"""
            dist = torch.cdist(x, y, p=2)
            return torch.exp(-dist ** 2 / (2 * sigma ** 2))
        
        # 展平特征
        source_flat = source_features.view(source_features.size(0), -1)
        target_flat = target_features.view(target_features.size(0), -1)
        
        # 计算核矩阵
        if kernel_type == 'gaussian':
            k_ss = gaussian_kernel(source_flat, source_flat).mean()
            k_tt = gaussian_kernel(target_flat, target_flat).mean()
            k_st = gaussian_kernel(source_flat, target_flat).mean()
        else:
            # 线性核
            k_ss = torch.mm(source_flat, source_flat.t()).mean()
            k_tt = torch.mm(target_flat, target_flat.t()).mean()
            k_st = torch.mm(source_flat, target_flat.t()).mean()
            
        # MMD损失
        mmd = k_ss + k_tt - 2 * k_st
        return mmd
        
    def forward(self, 
                source_features: torch.Tensor,
                target_features: torch.Tensor) -> torch.Tensor:
        """
        计算特征对齐损失
        
        Args:
            source_features: 源域特征
            target_features: 目标域特征
            
        Returns:
            torch.Tensor: 特征对齐损失
        """
        mmd = self.mmd_loss(source_features, target_features)
        return self.alignment_weight * mmd


class AdversarialWeatherLoss(nn.Module):
    """
    对抗天气损失
    
    专门用于Day→Rain域适应的损失函数
    """
    
    def __init__(self, 
                 weather_weight: float = 1.0,
                 num_weather_classes: int = 2):  # Day, Rain
        """
        初始化对抗天气损失
        
        Args:
            weather_weight: 天气损失权重
            num_weather_classes: 天气类别数量
        """
        super(AdversarialWeatherLoss, self).__init__()
        
        self.weather_weight = weather_weight
        self.num_weather_classes = num_weather_classes
        
        # 天气分类器
        self.weather_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_weather_classes)
        )
        
    def forward(self, 
                features: torch.Tensor,
                weather_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算对抗天气损失
        
        Args:
            features: 输入特征 [B, C, H, W]
            weather_labels: 天气标签 [B] (0: Day, 1: Rain)
            
        Returns:
            Dict: 包含天气损失和准确率的字典
        """
        # 天气分类预测
        weather_predictions = self.weather_classifier(features)
        
        # 计算交叉熵损失
        weather_loss = F.cross_entropy(weather_predictions, weather_labels.long())
        
        # 计算准确率
        with torch.no_grad():
            predictions = torch.argmax(weather_predictions, dim=1)
            accuracy = (predictions == weather_labels).float().mean()
            
        return {
            'weather_loss': self.weather_weight * weather_loss,
            'weather_accuracy': accuracy,
            'weather_predictions': F.softmax(weather_predictions, dim=1)
        }


class GradientReverseFunction(torch.autograd.Function):
    """
    梯度反转函数
    
    前向传播时保持输入不变，反向传播时将梯度乘以负权重
    """
    
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x)
        
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.weight * grad_output, None


class MFDAMLoss(nn.Module):
    """
    MFDAM综合损失函数
    
    整合所有域适应相关的损失函数
    """
    
    def __init__(self, 
                 domain_weight: float = 1.0,
                 consistency_weight: float = 0.5,
                 alignment_weight: float = 0.3,
                 weather_weight: float = 0.7):
        """
        初始化MFDAM损失
        
        Args:
            domain_weight: 域对抗损失权重
            consistency_weight: 一致性损失权重
            alignment_weight: 特征对齐损失权重
            weather_weight: 天气对抗损失权重
        """
        super(MFDAMLoss, self).__init__()
        
        self.domain_adversarial = DomainAdversarialLoss()
        self.consistency = ConsistencyLoss(consistency_weight=consistency_weight)
        self.alignment = FeatureAlignmentLoss(alignment_weight=alignment_weight)
        self.weather = AdversarialWeatherLoss(weather_weight=weather_weight)
        
        self.domain_weight = domain_weight
        
    def forward(self, 
                source_features: torch.Tensor,
                target_features: torch.Tensor,
                source_predictions: Optional[torch.Tensor] = None,
                target_predictions: Optional[torch.Tensor] = None,
                weather_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            source_features: 源域特征
            target_features: 目标域特征
            source_predictions: 源域预测 (可选)
            target_predictions: 目标域预测 (可选)
            weather_labels: 天气标签 (可选)
            
        Returns:
            Dict: 包含各种损失的字典
        """
        losses = {}
        total_loss = 0
        
        # 域对抗损失
        domain_results = self.domain_adversarial(source_features, target_features)
        losses.update(domain_results)
        total_loss += self.domain_weight * domain_results['domain_loss']
        
        # 一致性损失
        if source_predictions is not None and target_predictions is not None:
            consistency_loss = self.consistency(source_predictions, target_predictions)
            losses['consistency_loss'] = consistency_loss
            total_loss += consistency_loss
            
        # 特征对齐损失
        alignment_loss = self.alignment(source_features, target_features)
        losses['alignment_loss'] = alignment_loss
        total_loss += alignment_loss
        
        # 天气对抗损失
        if weather_labels is not None:
            all_features = torch.cat([source_features, target_features], dim=0)
            weather_results = self.weather(all_features, weather_labels)
            losses.update(weather_results)
            total_loss += weather_results['weather_loss']
            
        losses['total_da_loss'] = total_loss
        return losses