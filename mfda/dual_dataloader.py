# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
双域数据加载器 (Dual-Domain DataLoader)
用于Day→Rain域适应的数据加载和预处理

主要功能：
1. 同时加载源域(Day)和目标域(Rain)数据
2. 数据增强和预处理
3. 批次同步和平衡
4. 天气标签生成
"""

import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, Sampler
from typing import Dict, List, Tuple, Optional, Union
import yaml
from pathlib import Path

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.build import build_dataloader
from ultralytics.utils import LOGGER


class DualDomainDataLoader:
    """
    双域数据加载器
    
    同时处理源域(Day)和目标域(Rain)数据，确保每个批次包含两个域的数据
    """
    
    def __init__(self,
                 source_cfg: Union[str, Dict],
                 target_cfg: Union[str, Dict],
                 batch_size: int = 16,
                 img_size: int = 640,
                 workers: int = 8,
                 shuffle: bool = True,
                 domain_balance: float = 0.5,
                 augment: bool = True):
        """
        初始化双域数据加载器
        
        Args:
            source_cfg: 源域数据集配置文件路径或配置字典
            target_cfg: 目标域数据集配置文件路径或配置字典  
            batch_size: 批次大小
            img_size: 图像尺寸
            workers: 工作线程数
            shuffle: 是否打乱数据
            domain_balance: 域平衡比例 (0.5表示源域和目标域各占50%)
            augment: 是否进行数据增强
        """
        self.batch_size = batch_size
        self.img_size = img_size
        self.workers = workers
        self.shuffle = shuffle
        self.domain_balance = domain_balance
        self.augment = augment
        
        # 加载配置
        self.source_cfg = self._load_config(source_cfg)
        self.target_cfg = self._load_config(target_cfg)
        
        # 创建数据集
        self.source_dataset = self._create_dataset(self.source_cfg, domain_type='source')
        self.target_dataset = self._create_dataset(self.target_cfg, domain_type='target')
        
        # 计算每个域的批次大小
        self.source_batch_size = int(batch_size * domain_balance)
        self.target_batch_size = batch_size - self.source_batch_size
        
        # 创建数据加载器
        self.source_loader = self._create_dataloader(self.source_dataset, self.source_batch_size)
        self.target_loader = self._create_dataloader(self.target_dataset, self.target_batch_size)
        
        # 创建迭代器
        self.source_iter = iter(self.source_loader)
        self.target_iter = iter(self.target_loader)
        
        LOGGER.info(f"双域数据加载器初始化完成:")
        LOGGER.info(f"  源域(Day): {len(self.source_dataset)} 样本, 批次大小: {self.source_batch_size}")
        LOGGER.info(f"  目标域(Rain): {len(self.target_dataset)} 样本, 批次大小: {self.target_batch_size}")
        
    def _load_config(self, cfg: Union[str, Dict]) -> Dict:
        """
        加载数据集配置
        
        Args:
            cfg: 配置文件路径或配置字典
            
        Returns:
            Dict: 配置字典
        """
        if isinstance(cfg, str):
            with open(cfg, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return cfg
        
    def _create_dataset(self, cfg: Dict, domain_type: str) -> YOLODataset:
        """
        创建YOLO数据集
        
        Args:
            cfg: 数据集配置
            domain_type: 域类型 ('source' 或 'target')
            
        Returns:
            YOLODataset: YOLO数据集实例
        """
        # 构建数据集参数
        dataset_args = {
            'img_path': cfg['train'],
            'imgsz': self.img_size,
            'augment': self.augment,
            'cache': False,  # 不使用缓存以节省内存
            'prefix': f'{domain_type}: '
        }
        
        # 创建自定义数据集，添加域标签
        dataset = DomainYOLODataset(**dataset_args, domain_type=domain_type)
        
        return dataset
        
    def _create_dataloader(self, dataset: Dataset, batch_size: int) -> DataLoader:
        """
        创建数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            
        Returns:
            DataLoader: 数据加载器
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers=self.workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            drop_last=True  # 确保批次大小一致
        )
        
    def __iter__(self):
        """创建迭代器"""
        return self
        
    def __next__(self) -> Dict[str, torch.Tensor]:
        """
        获取下一个批次
        
        Returns:
            Dict: 包含源域和目标域数据的字典
        """
        try:
            # 获取源域数据
            try:
                source_batch = next(self.source_iter)
            except StopIteration:
                self.source_iter = iter(self.source_loader)
                source_batch = next(self.source_iter)
                
            # 获取目标域数据
            try:
                target_batch = next(self.target_iter)
            except StopIteration:
                self.target_iter = iter(self.target_loader)
                target_batch = next(self.target_iter)
                
            # 合并批次数据
            combined_batch = self._combine_batches(source_batch, target_batch)
            
            return combined_batch
            
        except Exception as e:
            LOGGER.error(f"数据加载错误: {e}")
            raise StopIteration
            
    def _combine_batches(self, source_batch: Dict, target_batch: Dict) -> Dict[str, torch.Tensor]:
        """
        合并源域和目标域批次
        
        Args:
            source_batch: 源域批次数据
            target_batch: 目标域批次数据
            
        Returns:
            Dict: 合并后的批次数据
        """
        # 合并图像
        source_imgs = source_batch['img']  # [B1, C, H, W]
        target_imgs = target_batch['img']  # [B2, C, H, W]
        combined_imgs = torch.cat([source_imgs, target_imgs], dim=0)
        
        # 创建域标签 (0: 源域, 1: 目标域)
        source_domain_labels = torch.zeros(source_imgs.size(0), dtype=torch.long)
        target_domain_labels = torch.ones(target_imgs.size(0), dtype=torch.long)
        domain_labels = torch.cat([source_domain_labels, target_domain_labels], dim=0)
        
        # 创建天气标签 (0: Day, 1: Rain)
        weather_labels = domain_labels.clone()  # 简化版本，域标签即天气标签
        
        # 合并标签数据 (如果存在)
        combined_batch = {
            'img': combined_imgs,
            'domain_labels': domain_labels,
            'weather_labels': weather_labels,
            'batch_idx': torch.arange(combined_imgs.size(0)),
        }
        
        # 处理目标检测标签
        if 'bboxes' in source_batch and 'bboxes' in target_batch:
            # 合并边界框
            source_bboxes = source_batch['bboxes']
            target_bboxes = target_batch['bboxes']
            
            # 调整批次索引
            if hasattr(target_bboxes, 'shape') and len(target_bboxes.shape) > 1:
                if target_bboxes.size(1) > 0:  # 如果有边界框
                    target_bboxes[:, 0] += source_imgs.size(0)  # 调整批次索引
                    
            combined_bboxes = torch.cat([source_bboxes, target_bboxes], dim=0)
            combined_batch['bboxes'] = combined_bboxes
            
        # 处理类别标签
        if 'cls' in source_batch and 'cls' in target_batch:
            combined_cls = torch.cat([source_batch['cls'], target_batch['cls']], dim=0)
            combined_batch['cls'] = combined_cls
            
        return combined_batch
        
    def __len__(self) -> int:
        """返回数据集长度"""
        return min(len(self.source_loader), len(self.target_loader))
        
    def reset(self):
        """重置迭代器"""
        self.source_iter = iter(self.source_loader)
        self.target_iter = iter(self.target_loader)


class DomainYOLODataset(YOLODataset):
    """
    带域标签的YOLO数据集
    
    扩展标准YOLO数据集，添加域信息
    """
    
    def __init__(self, domain_type: str = 'source', **kwargs):
        """
        初始化域YOLO数据集
        
        Args:
            domain_type: 域类型 ('source' 或 'target')
            **kwargs: 其他YOLO数据集参数
        """
        super().__init__(**kwargs)
        self.domain_type = domain_type
        self.domain_label = 0 if domain_type == 'source' else 1
        
    def __getitem__(self, index):
        """
        获取数据项
        
        Args:
            index: 数据索引
            
        Returns:
            Dict: 包含域信息的数据项
        """
        # 获取原始数据
        data = super().__getitem__(index)
        
        # 添加域信息
        if isinstance(data, dict):
            data['domain_label'] = self.domain_label
            data['domain_type'] = self.domain_type
        
        return data


class WeatherAugmentation:
    """
    天气增强类
    
    专门用于模拟不同天气条件的数据增强
    """
    
    def __init__(self, rain_intensity: float = 0.3, fog_density: float = 0.2):
        """
        初始化天气增强
        
        Args:
            rain_intensity: 雨强度
            fog_density: 雾密度
        """
        self.rain_intensity = rain_intensity
        self.fog_density = fog_density
        
    def add_rain(self, image: torch.Tensor) -> torch.Tensor:
        """
        添加雨效果
        
        Args:
            image: 输入图像 [C, H, W]
            
        Returns:
            torch.Tensor: 带雨效果的图像
        """
        if random.random() > 0.5:  # 50%概率添加雨效果
            return image
            
        # 简单的雨效果：添加垂直线条
        _, h, w = image.shape
        rain_mask = torch.zeros_like(image)
        
        # 随机生成雨滴位置
        num_drops = int(h * w * self.rain_intensity * 0.001)
        for _ in range(num_drops):
            x = random.randint(0, w - 1)
            y_start = random.randint(0, h // 2)
            y_end = min(y_start + random.randint(10, 30), h - 1)
            
            # 绘制雨滴
            for y in range(y_start, y_end):
                if y < h and x < w:
                    rain_mask[:, y, x] = 0.3
                    
        # 应用雨效果
        rainy_image = image * (1 - rain_mask) + rain_mask
        return torch.clamp(rainy_image, 0, 1)
        
    def add_fog(self, image: torch.Tensor) -> torch.Tensor:
        """
        添加雾效果
        
        Args:
            image: 输入图像 [C, H, W]
            
        Returns:
            torch.Tensor: 带雾效果的图像
        """
        if random.random() > 0.3:  # 30%概率添加雾效果
            return image
            
        # 简单的雾效果：添加白色遮罩
        fog_mask = torch.ones_like(image) * self.fog_density
        foggy_image = image * (1 - self.fog_density) + fog_mask
        return torch.clamp(foggy_image, 0, 1)
        
    def __call__(self, image: torch.Tensor, weather_type: str = 'rain') -> torch.Tensor:
        """
        应用天气增强
        
        Args:
            image: 输入图像
            weather_type: 天气类型 ('rain' 或 'fog')
            
        Returns:
            torch.Tensor: 增强后的图像
        """
        if weather_type == 'rain':
            return self.add_rain(image)
        elif weather_type == 'fog':
            return self.add_fog(image)
        else:
            return image


def create_dual_domain_dataloader(source_data: str,
                                  target_data: str,
                                  batch_size: int = 16,
                                  img_size: int = 640,
                                  workers: int = 8,
                                  **kwargs) -> DualDomainDataLoader:
    """
    创建双域数据加载器的便捷函数
    
    Args:
        source_data: 源域数据配置文件路径
        target_data: 目标域数据配置文件路径
        batch_size: 批次大小
        img_size: 图像尺寸
        workers: 工作线程数
        **kwargs: 其他参数
        
    Returns:
        DualDomainDataLoader: 双域数据加载器
    """
    return DualDomainDataLoader(
        source_cfg=source_data,
        target_cfg=target_data,
        batch_size=batch_size,
        img_size=img_size,
        workers=workers,
        **kwargs
    )