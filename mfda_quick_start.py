#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MFDAM快速入门教程 (MFDAM Quick Start Tutorial)
帮助用户快速上手Day→Rain域适应训练

使用方法:
python mfda_quick_start.py
"""

import torch
import sys
import os
from pathlib import Path

# 添加ultralytics到路径
sys.path.append('.')
sys.path.append('ultralytics')

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"📋 {title}")
    print("=" * 60)

def print_step(step_num, title):
    """打印步骤"""
    print(f"\n🔹 步骤 {step_num}: {title}")
    print("-" * 40)

def check_environment():
    """检查环境依赖"""
    print_header("环境检查")
    
    requirements = {
        'torch': 'PyTorch深度学习框架',
        'yaml': 'YAML配置文件支持',
        'pathlib': 'Python路径操作'
    }
    
    missing = []
    for pkg, desc in requirements.items():
        try:
            __import__(pkg)
            print(f"✅ {pkg}: {desc}")
        except ImportError:
            print(f"❌ {pkg}: {desc} - 未安装")
            missing.append(pkg)
    
    if missing:
        print(f"\n⚠️ 缺少依赖: {', '.join(missing)}")
        print("请先安装: pip install torch pyyaml")
        return False
    
    print("\n✅ 环境检查通过!")
    return True

def demonstrate_core_features():
    """演示核心功能"""
    print_header("MFDAM核心功能演示")
    
    try:
        from mfda.config import MFDAMConfig, DAY_RAIN_CONFIG
        from mfda.multi_scale_extractor import MultiScaleFeatureExtractor
        from mfda.domain_losses import MFDAMLoss
        
        print_step(1, "配置管理")
        config = DAY_RAIN_CONFIG
        print(f"  Day→Rain优化配置已加载")
        print(f"  域平衡比例: {config.domain_balance}")
        print(f"  域损失权重: {config.domain_weight}")
        
        print_step(2, "多尺度特征提取")
        extractor = MultiScaleFeatureExtractor()
        print(f"  特征提取器已创建")
        print(f"  支持 {len(extractor.feature_dims)} 个尺度")
        print(f"  特征维度: {extractor.feature_dims}")
        
        print_step(3, "域适应损失")
        loss_fn = MFDAMLoss()
        print(f"  MFDAM综合损失函数已创建")
        print(f"  包含域对抗、一致性、对齐等损失")
        
        # 简单测试
        print_step(4, "功能测试")
        source_feat = torch.randn(2, 256, 32, 32)
        target_feat = torch.randn(2, 256, 32, 32)
        weather_labels = torch.cat([torch.zeros(2), torch.ones(2)]).long()
        
        with torch.no_grad():
            results = loss_fn(source_feat, target_feat, weather_labels=weather_labels)
        
        print(f"  测试成功! 总损失: {results['total_da_loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 功能演示失败: {e}")
        return False

def create_sample_configs():
    """创建示例配置文件"""
    print_header("创建示例配置文件")
    
    try:
        from mfda.config import create_day_rain_dataset_configs, save_dataset_configs
        
        print_step(1, "生成数据集配置")
        
        # 创建示例配置
        source_config, target_config = create_day_rain_dataset_configs(
            source_train_path='datasets/day_dataset/images/train',
            source_val_path='datasets/day_dataset/images/val',
            target_train_path='datasets/rain_dataset/images/train', 
            target_val_path='datasets/rain_dataset/images/val',
            num_classes=80
        )
        
        # 保存配置
        config_dir = Path('tutorial_configs')
        source_path, target_path = save_dataset_configs(
            source_config, target_config, config_dir
        )
        
        print(f"✅ 配置文件已创建:")
        print(f"  源域(Day): {source_path}")
        print(f"  目标域(Rain): {target_path}")
        
        print_step(2, "配置文件说明")
        print(f"  - source_day.yaml: 白天晴朗天气数据配置")
        print(f"  - target_rain.yaml: 雨天天气数据配置")
        print(f"  - 两个配置包含相同的80个COCO类别")
        print(f"  - 支持不同的数据增强策略")
        
        return config_dir
        
    except Exception as e:
        print(f"❌ 配置创建失败: {e}")
        return None

def show_training_commands(config_dir):
    """显示训练命令"""
    print_header("训练命令示例")
    
    if config_dir is None:
        print("⚠️ 配置文件未创建，使用默认路径")
        source_config = "cfg/datasets/mfda/source_day.yaml"
        target_config = "cfg/datasets/mfda/target_rain.yaml"
    else:
        source_config = f"{config_dir}/source_day.yaml"
        target_config = f"{config_dir}/target_rain.yaml"
    
    print_step(1, "基础训练命令")
    basic_cmd = f"""python train_mfdam.py \\
    --source-data {source_config} \\
    --target-data {target_config} \\
    --epochs 100 \\
    --batch-size 16 \\
    --project runs/mfdam \\
    --name day2rain_experiment"""
    
    print(basic_cmd)
    
    print_step(2, "高级配置训练")
    advanced_cmd = f"""python train_mfdam.py \\
    --source-data {source_config} \\
    --target-data {target_config} \\
    --model cfg/models/mfda/yolov8n-mfdam.yaml \\
    --epochs 200 \\
    --batch-size 32 \\
    --lr0 0.01 \\
    --da-lr 0.0005 \\
    --domain-weight 1.2 \\
    --domain-balance 0.6 \\
    --config-preset day_rain \\
    --device 0"""
    
    print(advanced_cmd)
    
    print_step(3, "参数说明")
    params = {
        '--source-data': '源域(Day)数据配置文件',
        '--target-data': '目标域(Rain)数据配置文件', 
        '--epochs': '训练轮数',
        '--batch-size': '批次大小',
        '--da-lr': '域适应学习率',
        '--domain-weight': '域对抗损失权重',
        '--domain-balance': '域平衡比例 (>0.5偏向目标域)',
        '--config-preset': 'MFDAM配置预设 (day_rain推荐)'
    }
    
    for param, desc in params.items():
        print(f"  {param}: {desc}")

def show_dataset_preparation():
    """显示数据集准备指南"""
    print_header("数据集准备指南")
    
    print_step(1, "目录结构")
    directory_structure = """
datasets/
├── day_dataset/              # 源域(白天数据)
│   ├── images/
│   │   ├── train/           # 训练图像
│   │   └── val/             # 验证图像
│   └── labels/
│       ├── train/           # 训练标签(YOLO格式)
│       └── val/             # 验证标签
└── rain_dataset/            # 目标域(雨天数据)
    ├── images/
    │   ├── train/           # 训练图像
    │   └── val/             # 验证图像
    └── labels/              # 标签(可选，可无监督)
        ├── train/
        └── val/
"""
    print(directory_structure)
    
    print_step(2, "数据要求")
    requirements = [
        "源域: 白天晴朗天气条件下的目标检测数据",
        "目标域: 雨天天气条件下的数据",
        "图像格式: JPG, PNG等常见格式",
        "标签格式: YOLO格式 (.txt文件)",
        "建议源域数据量: 5000+ 张", 
        "建议目标域数据量: 3000+ 张",
        "目标域可使用70%无标签数据"
    ]
    
    for req in requirements:
        print(f"  • {req}")
    
    print_step(3, "数据质量建议")
    quality_tips = [
        "确保图像质量良好，标注准确",
        "雨天数据应包含不同雨强(小雨、中雨、大雨)",
        "保持类别分布相对平衡",
        "图像分辨率建议不低于640x640",
        "避免过度曝光或过暗的图像"
    ]
    
    for tip in quality_tips:
        print(f"  💡 {tip}")

def show_next_steps():
    """显示后续步骤"""
    print_header("后续步骤")
    
    steps = [
        ("准备数据集", "按照指南准备Day和Rain数据集"),
        ("修改配置", "编辑生成的配置文件，设置正确的数据路径"),
        ("开始训练", "运行训练命令开始Day→Rain域适应"),
        ("监控训练", "观察域分类准确率和检测性能"),
        ("评估结果", "在雨天测试集上评估模型性能"),
        ("调优参数", "根据结果调整域适应参数")
    ]
    
    for i, (title, desc) in enumerate(steps, 1):
        print(f"\n{i}. {title}")
        print(f"   {desc}")
    
    print("\n🎯 目标: 显著提升模型在雨天条件下的检测性能!")

def main():
    """主函数"""
    print("🌟 MFDAM (Multi-scale Fusion Domain Adaptation Module)")
    print("Day→Rain域适应快速入门教程")
    print("专为YOLOv8设计的域适应解决方案")
    
    # 检查环境
    if not check_environment():
        return
    
    # 演示核心功能
    if not demonstrate_core_features():
        print("❌ 核心功能演示失败，请检查安装")
        return
    
    # 创建配置文件
    config_dir = create_sample_configs()
    
    # 显示训练命令
    show_training_commands(config_dir)
    
    # 显示数据集准备指南
    show_dataset_preparation()
    
    # 显示后续步骤
    show_next_steps()
    
    print("\n" + "=" * 60)
    print("🎉 MFDAM快速入门教程完成!")
    print("🚀 准备好开始你的Day→Rain域适应之旅了吗?")
    print("=" * 60)
    
    # 显示有用的链接
    print("\n📚 更多资源:")
    print("  • 详细文档: mfda/README_CN.md")
    print("  • 使用示例: example_mfdam_usage.py")
    print("  • 配置文件: cfg/datasets/mfda/ 和 cfg/models/mfda/")
    print("  • 问题反馈: 提交GitHub Issue")

if __name__ == "__main__":
    main()