#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MFDAM使用示例 (MFDAM Usage Example)
演示如何使用多尺度融合域适应模块进行Day→Rain域适应训练

运行示例:
python example_mfdam_usage.py
"""

import torch
import sys
from pathlib import Path

# 添加ultralytics到路径
sys.path.append('.')
sys.path.append('ultralytics')

def example_1_basic_config():
    """示例1: 基础配置使用"""
    print("=" * 50)
    print("示例1: MFDAM基础配置")
    print("=" * 50)
    
    from mfda.config import MFDAMConfig, DAY_RAIN_CONFIG
    
    # 使用默认配置
    default_config = MFDAMConfig()
    print(f"默认配置:")
    print(f"  特征维度: {default_config.feature_dims}")
    print(f"  融合维度: {default_config.fusion_dim}")
    print(f"  域适应学习率: {default_config.da_lr}")
    
    # 使用Day→Rain优化配置
    day_rain_config = DAY_RAIN_CONFIG
    print(f"\nDay→Rain优化配置:")
    print(f"  域平衡比例: {day_rain_config.domain_balance}")
    print(f"  域损失权重: {day_rain_config.domain_weight}")
    print(f"  天气损失权重: {day_rain_config.weather_weight}")
    
    # 保存配置
    config_path = Path('example_config.yaml')
    day_rain_config.to_yaml(config_path)
    print(f"  配置已保存到: {config_path}")
    
    return day_rain_config


def example_2_feature_extraction():
    """示例2: 多尺度特征提取"""
    print("\n" + "=" * 50)
    print("示例2: 多尺度特征提取和融合")
    print("=" * 50)
    
    from mfda.multi_scale_extractor import MultiScaleFeatureExtractor, FusionModule
    
    # 创建特征提取器
    extractor = MultiScaleFeatureExtractor(
        feature_dims=[256, 512, 1024],
        extract_layers=['layer1', 'layer2', 'layer3']
    )
    
    print(f"特征提取器信息:")
    print(f"  适配器数量: {len(extractor.feature_adapters)}")
    print(f"  提取层: {extractor.extract_layers}")
    
    # 模拟多尺度特征
    features = [
        torch.randn(2, 256, 64, 64),   # P3
        torch.randn(2, 512, 32, 32),   # P4
        torch.randn(2, 1024, 16, 16),  # P5
    ]
    
    # 特征融合
    fusion_module = FusionModule(feature_dim=256, num_scales=3)
    
    # 模拟适配后的特征
    adapted_features = []
    for i, feature in enumerate(features):
        if i < len(extractor.feature_adapters):
            adapted = extractor.feature_adapters[i](feature)
            adapted_features.append(adapted)
    
    fused_features = fusion_module(adapted_features)
    
    print(f"\n特征融合结果:")
    print(f"  输入特征数量: {len(adapted_features)}")
    print(f"  融合后特征尺寸: {fused_features.shape}")
    
    return fused_features


def example_3_domain_losses():
    """示例3: 域对抗损失计算"""
    print("\n" + "=" * 50)
    print("示例3: 域对抗损失计算")
    print("=" * 50)
    
    from mfda.domain_losses import DomainAdversarialLoss, MFDAMLoss
    
    # 创建域对抗损失
    da_loss = DomainAdversarialLoss(input_dim=256)
    
    # 模拟源域和目标域特征
    source_features = torch.randn(4, 256, 32, 32)  # Day特征
    target_features = torch.randn(4, 256, 32, 32)  # Rain特征
    
    # 计算域对抗损失
    loss_results = da_loss(source_features, target_features)
    
    print(f"域对抗损失结果:")
    print(f"  域分类损失: {loss_results['domain_loss'].item():.4f}")
    print(f"  域分类准确率: {loss_results['domain_accuracy'].item():.3f}")
    
    # 综合MFDAM损失
    mfdam_loss = MFDAMLoss()
    
    # 模拟天气标签 (0: Day, 1: Rain)
    weather_labels = torch.cat([
        torch.zeros(4, dtype=torch.long),  # Day
        torch.ones(4, dtype=torch.long)    # Rain
    ])
    
    total_results = mfdam_loss(
        source_features=source_features,
        target_features=target_features,
        weather_labels=weather_labels
    )
    
    print(f"\nMFDAM综合损失:")
    print(f"  总域适应损失: {total_results['total_da_loss'].item():.4f}")
    print(f"  域分类损失: {total_results['domain_loss'].item():.4f}")
    print(f"  特征对齐损失: {total_results['alignment_loss'].item():.4f}")
    if 'weather_loss' in total_results:
        print(f"  天气分类损失: {total_results['weather_loss'].item():.4f}")
    
    return total_results


def example_4_training_simulation():
    """示例4: 训练过程模拟"""
    print("\n" + "=" * 50)
    print("示例4: MFDAM训练过程模拟")
    print("=" * 50)
    
    from mfda.config import DAY_RAIN_CONFIG
    from mfda.multi_scale_extractor import MultiScaleFeatureExtractor
    from mfda.domain_losses import MFDAMLoss
    
    # 使用Day→Rain配置
    config = DAY_RAIN_CONFIG
    
    # 创建组件
    feature_extractor = MultiScaleFeatureExtractor()
    mfdam_loss = MFDAMLoss(
        domain_weight=config.domain_weight,
        weather_weight=config.weather_weight
    )
    
    print(f"模拟训练配置:")
    print(f"  域平衡比例: {config.domain_balance}")
    print(f"  域适应学习率: {config.da_lr}")
    print(f"  域损失权重: {config.domain_weight}")
    
    # 模拟训练步骤
    num_epochs = 3
    batch_size = 4
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        
        # 模拟批次数据
        source_batch = torch.randn(batch_size, 256, 32, 32)
        target_batch = torch.randn(batch_size, 256, 32, 32) 
        
        # 天气标签
        weather_labels = torch.cat([
            torch.zeros(batch_size, dtype=torch.long),
            torch.ones(batch_size, dtype=torch.long)
        ])
        
        # 计算损失
        loss_results = mfdam_loss(
            source_features=source_batch,
            target_features=target_batch,
            weather_labels=weather_labels
        )
        
        # 计算动态GRL权重
        grl_weight = config.get_grl_weight(epoch, num_epochs)
        
        print(f"  GRL权重: {grl_weight:.3f}")
        print(f"  总损失: {loss_results['total_da_loss'].item():.4f}")
        print(f"  域准确率: {loss_results['domain_accuracy'].item():.3f}")
    
    print(f"\n✅ 训练模拟完成!")


def example_5_dataset_config():
    """示例5: 数据集配置创建"""
    print("\n" + "=" * 50)
    print("示例5: Day→Rain数据集配置")
    print("=" * 50)
    
    from mfda.config import create_day_rain_dataset_configs, save_dataset_configs
    
    # 创建数据集配置
    source_config, target_config = create_day_rain_dataset_configs(
        source_train_path='datasets/day/images/train',
        source_val_path='datasets/day/images/val',
        target_train_path='datasets/rain/images/train',
        target_val_path='datasets/rain/images/val',
        num_classes=80
    )
    
    print(f"源域配置:")
    print(f"  域类型: {source_config['domain']}")
    print(f"  天气条件: {source_config['weather']}")
    print(f"  类别数量: {source_config['nc']}")
    
    print(f"\n目标域配置:")
    print(f"  域类型: {target_config['domain']}")
    print(f"  天气条件: {target_config['weather']}")
    print(f"  类别数量: {target_config['nc']}")
    
    # 保存配置文件
    config_dir = Path('example_configs')
    source_path, target_path = save_dataset_configs(
        source_config, target_config, config_dir
    )
    
    print(f"\n配置文件已保存:")
    print(f"  源域: {source_path}")
    print(f"  目标域: {target_path}")


def main():
    """主函数"""
    print("🌟 MFDAM (Multi-scale Fusion Domain Adaptation Module) 使用示例")
    print("专门用于YOLOv8的Day→Rain域适应")
    print("=" * 80)
    
    try:
        # 运行所有示例
        config = example_1_basic_config()
        fused_features = example_2_feature_extraction()
        loss_results = example_3_domain_losses()
        example_4_training_simulation()
        example_5_dataset_config()
        
        print("\n" + "=" * 80)
        print("🎉 所有示例运行完成!")
        print("\n📋 MFDAM功能总览:")
        print("  ✅ 多尺度特征提取和融合")
        print("  ✅ 域对抗训练损失")
        print("  ✅ Day→Rain专项优化")
        print("  ✅ 配置管理系统")
        print("  ✅ 数据集配置生成")
        print("  ✅ 训练过程模拟")
        
        print("\n🚀 开始使用MFDAM:")
        print("  1. 准备Day和Rain数据集")
        print("  2. 配置数据集路径")
        print("  3. 运行: python train_mfdam.py --source-data day.yaml --target-data rain.yaml")
        
    except Exception as e:
        print(f"❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()