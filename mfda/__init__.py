# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
多尺度融合域适应模块 (Multi-scale Fusion Domain Adaptation Module - MFDAM)
专门用于YOLOv8的Day→Rain域适应训练

该模块包含：
- 多尺度特征提取器
- 域对抗损失函数
- 双域数据加载器
- MFDAM训练器
- 配置管理
"""

try:
    from .multi_scale_extractor import MultiScaleFeatureExtractor, FusionModule
    from .domain_losses import DomainAdversarialLoss, ConsistencyLoss
    from .config import MFDAMConfig
except ImportError:
    # 处理相对导入问题
    try:
        from mfda.multi_scale_extractor import MultiScaleFeatureExtractor, FusionModule
        from mfda.domain_losses import DomainAdversarialLoss, ConsistencyLoss
        from mfda.config import MFDAMConfig
    except ImportError:
        pass

# 需要ultralytics依赖的模块，可选导入
try:
    from .dual_dataloader import DualDomainDataLoader
    from .mfdam_trainer import MFDAMTrainer
except ImportError:
    try:
        from mfda.dual_dataloader import DualDomainDataLoader
        from mfda.mfdam_trainer import MFDAMTrainer
    except ImportError:
        # 在没有ultralytics环境时跳过这些模块
        DualDomainDataLoader = None
        MFDAMTrainer = None

__all__ = [
    'MultiScaleFeatureExtractor',
    'FusionModule', 
    'DomainAdversarialLoss',
    'ConsistencyLoss',
    'DualDomainDataLoader',
    'MFDAMTrainer',
    'MFDAMConfig'
]