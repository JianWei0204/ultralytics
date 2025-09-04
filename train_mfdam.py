# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
MFDAM训练脚本 (MFDAM Training Script)
专门用于Day→Rain域适应的YOLOv8训练脚本

使用方法:
python train_mfdam.py --source-data path/to/day_dataset.yaml --target-data path/to/rain_dataset.yaml

功能特性:
1. 多尺度特征融合
2. 域对抗训练
3. Day→Rain特定优化
4. 训练过程监控
"""

import argparse
import yaml
from pathlib import Path

from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr

from mfda import MFDAMTrainer, MFDAMConfig, DAY_RAIN_CONFIG


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='YOLOv8 MFDAM Day→Rain域适应训练',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基础配置
    parser.add_argument('--model', type=str, default='yolov8n.yaml', 
                       help='模型配置文件路径')
    parser.add_argument('--source-data', type=str, required=True,
                       help='源域(Day)数据集配置文件路径')
    parser.add_argument('--target-data', type=str, required=True,
                       help='目标域(Rain)数据集配置文件路径')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='图像尺寸')
    parser.add_argument('--workers', type=int, default=8,
                       help='数据加载工作线程数')
    
    # 学习率配置
    parser.add_argument('--lr0', type=float, default=0.01,
                       help='初始学习率')
    parser.add_argument('--da-lr', type=float, default=0.0005,
                       help='域适应学习率')
    
    # 权重配置
    parser.add_argument('--pretrained', type=str, default='yolov8n.pt',
                       help='预训练权重路径')
    parser.add_argument('--domain-weight', type=float, default=1.2,
                       help='域对抗损失权重')
    parser.add_argument('--da-loss-weight', type=float, default=0.15,
                       help='总域适应损失权重')
    
    # 设备配置
    parser.add_argument('--device', default='',
                       help='训练设备 (例如: 0 或 0,1,2,3 或 cpu)')
    
    # 保存配置
    parser.add_argument('--project', type=str, default='runs/mfdam',
                       help='项目保存目录')
    parser.add_argument('--name', type=str, default='day2rain',
                       help='实验名称')
    parser.add_argument('--save-period', type=int, default=10,
                       help='保存检查点间隔(轮数)')
    
    # MFDAM特定配置
    parser.add_argument('--config-preset', type=str, default='day_rain',
                       choices=['default', 'day_rain', 'lightweight'],
                       help='MFDAM配置预设')
    parser.add_argument('--domain-balance', type=float, default=0.6,
                       help='域平衡比例 (0.5=平衡, >0.5偏向目标域)')
    
    # 验证配置
    parser.add_argument('--val-period', type=int, default=5,
                       help='验证间隔(轮数)')
    
    return parser.parse_args()


def load_mfdam_config(preset: str, **kwargs) -> MFDAMConfig:
    """
    加载MFDAM配置
    
    Args:
        preset: 配置预设名称
        **kwargs: 额外的配置参数
        
    Returns:
        MFDAMConfig: MFDAM配置对象
    """
    if preset == 'day_rain':
        config = DAY_RAIN_CONFIG
    elif preset == 'lightweight':
        from mfda.config import LIGHTWEIGHT_CONFIG
        config = LIGHTWEIGHT_CONFIG
    else:
        config = MFDAMConfig()
    
    # 应用命令行参数覆盖
    config.update(**kwargs)
    
    return config


def validate_data_configs(source_data: str, target_data: str):
    """
    验证数据集配置文件
    
    Args:
        source_data: 源域数据配置路径
        target_data: 目标域数据配置路径
    """
    # 检查文件存在性
    if not Path(source_data).exists():
        raise FileNotFoundError(f"源域数据配置文件不存在: {source_data}")
    if not Path(target_data).exists():
        raise FileNotFoundError(f"目标域数据配置文件不存在: {target_data}")
    
    # 验证配置内容
    with open(source_data, 'r', encoding='utf-8') as f:
        source_cfg = yaml.safe_load(f)
    with open(target_data, 'r', encoding='utf-8') as f:
        target_cfg = yaml.safe_load(f)
    
    # 检查必要字段
    required_fields = ['train', 'val', 'nc', 'names']
    for field in required_fields:
        if field not in source_cfg:
            raise ValueError(f"源域配置缺少必要字段: {field}")
        if field not in target_cfg:
            raise ValueError(f"目标域配置缺少必要字段: {field}")
    
    # 检查类别数量一致性
    if source_cfg['nc'] != target_cfg['nc']:
        raise ValueError(f"源域和目标域类别数量不一致: {source_cfg['nc']} vs {target_cfg['nc']}")
    
    LOGGER.info(f"{colorstr('green', '数据配置验证通过')}")
    LOGGER.info(f"  源域: {source_cfg['nc']} 类别, 训练集: {source_cfg.get('train', 'N/A')}")
    LOGGER.info(f"  目标域: {target_cfg['nc']} 类别, 训练集: {target_cfg.get('train', 'N/A')}")


def main():
    """主函数"""
    args = parse_args()
    
    # 显示配置信息
    LOGGER.info(f"{colorstr('cyan', 'bold', 'MFDAM Day→Rain 域适应训练')}")
    LOGGER.info(f"  模型: {args.model}")
    LOGGER.info(f"  源域数据: {args.source_data}")
    LOGGER.info(f"  目标域数据: {args.target_data}")
    LOGGER.info(f"  训练轮数: {args.epochs}")
    LOGGER.info(f"  批次大小: {args.batch_size}")
    
    # 验证数据配置
    validate_data_configs(args.source_data, args.target_data)
    
    # 创建MFDAM配置
    mfdam_config = load_mfdam_config(
        preset=args.config_preset,
        da_lr=args.da_lr,
        domain_weight=args.domain_weight,
        da_loss_weight=args.da_loss_weight,
        domain_balance=args.domain_balance
    )
    
    LOGGER.info(f"{colorstr('green', 'MFDAM配置')}: {args.config_preset}")
    LOGGER.info(f"  域适应学习率: {mfdam_config.da_lr}")
    LOGGER.info(f"  域平衡比例: {mfdam_config.domain_balance}")
    LOGGER.info(f"  域损失权重: {mfdam_config.domain_weight}")
    
    # 创建训练参数
    train_args = {
        'model': args.model,
        'data': args.source_data,  # 使用源域作为主数据
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.imgsz,
        'lr0': args.lr0,
        'workers': args.workers,
        'device': args.device,
        'project': args.project,
        'name': args.name,
        'save_period': args.save_period,
        'val_period': args.val_period,
        'exist_ok': True
    }
    
    if args.pretrained:
        train_args['pretrained'] = args.pretrained
    
    # 创建配置对象
    cfg = get_cfg(DEFAULT_CFG, {})
    
    # 创建MFDAM训练器
    LOGGER.info(f"{colorstr('blue', '创建MFDAM训练器...')}")
    trainer = MFDAMTrainer(cfg=cfg, overrides=train_args)
    
    # 设置MFDAM组件
    LOGGER.info(f"{colorstr('blue', '设置MFDAM组件...')}")
    trainer.setup_mfdam(target_data=args.target_data, mfdam_config=mfdam_config)
    
    # 开始训练
    LOGGER.info(f"{colorstr('green', 'bold', '开始MFDAM训练...')}")
    try:
        trainer.train()
        LOGGER.info(f"{colorstr('green', 'bold', 'MFDAM训练完成!')}")
    except Exception as e:
        LOGGER.error(f"{colorstr('red', 'bold', f'训练失败: {e}')}")
        raise
    
    return trainer


if __name__ == '__main__':
    main()