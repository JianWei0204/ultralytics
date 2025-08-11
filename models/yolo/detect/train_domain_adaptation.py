#!/usr/bin/env python3
"""
Domain Adaptation Training Script for YOLOv8
Usage:
    python train_domain_adaptation.py --source_data path/to/source.yaml --target_data path/to/target.yaml --model yolov8n.pt --epochs 100
This script implements domain adaptation training following the UDAT approach.
"""

import argparse
import yaml
from pathlib import Path

from ultralytics.models.yolo.detect.domain_adaptation_trainer import DomainAdaptationTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YOLOv8 Domain Adaptation Training')

    # Basic training arguments
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Model config (.yaml) or pretrained weights (.pt)')
    parser.add_argument('--pretrained', type=str, default='', help='Pretrained weights file (when using .yaml config)')
    parser.add_argument('--source_data', type=str, required=True, help='yolov8/ultralytics/cfg/datasets/source_dataset.yaml')
    parser.add_argument('--target_data', type=str, required=True, help='yolov8/ultralytics/cfg/datasets/target_dataset.yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='', help='0')

    # Domain adaptation specific arguments
    parser.add_argument('--discriminator_lr', type=float, default=1e-4, help='Discriminator learning rate')
    parser.add_argument('--adv_loss_weight', type=float, default=0.1, help='Adversarial loss weight')
    parser.add_argument('--discriminator_channels', type=int, default=512, help='Discriminator input channels (P4 feature channels)')

    # Training configuration
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=float, default=3.0, help='Warmup epochs')

    # Output and logging
    parser.add_argument('--project', type=str, default='runs/domain_adapt', help='Project directory')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--save_period', type=int, default=10, help='Save checkpoint every x epochs')
    parser.add_argument('--val', action='store_true', help='Validate during training')
    parser.add_argument('--plots', action='store_true', help='Generate training plots')

    # Resume training
    parser.add_argument('--resume', type=str, default='', help='Resume training from checkpoint')
    parser.add_argument('--resume_discriminator', type=str, default='', help='Resume discriminator from checkpoint')

    return parser.parse_args()


def create_domain_config(source_data, target_data):
    """
    Create a combined configuration for domain adaptation.
    """
    # Load source and target configurations
    with open(source_data, 'r') as f:
        source_config = yaml.safe_load(f)

    with open(target_data, 'r') as f:
        target_config = yaml.safe_load(f)

    # Create combined configuration
    domain_config = {
        'source': source_config,
        'target': target_config,
        'nc': source_config.get('nc', target_config.get('nc')),
        'names': source_config.get('names', target_config.get('names'))
    }

    # Use source domain configuration as base for training
    base_config = source_config.copy()
    base_config['domain_adaptation'] = domain_config

    return base_config


def main():
    """Main training function."""
    args = parse_args()

    # Create domain adaptation configuration
    domain_config = create_domain_config(args.source_data, args.target_data)

    # Prepare training arguments with proper structure
    training_args = {
        'model': args.model,
        'data': domain_config,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'lr0': args.lr0,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'project': args.project,
        'name': args.name,
        'save_period': args.save_period,
        'val': args.val,
        'plots': args.plots,
        'resume': args.resume,
        
        'seed': 0,  # 明确添加种子参数
        'deterministic': True,
        'verbose': True,
        'amp': True,
        'save': True,
        
        'discriminator_lr': args.discriminator_lr,
        'adv_loss_weight': args.adv_loss_weight,
        'discriminator_channels': args.discriminator_channels,
        'resume_discriminator': args.resume_discriminator,
    }

    # Initialize domain adaptation trainer with proper cfg parameter
    trainer = DomainAdaptationTrainer(cfg=training_args, overrides=None)

    # Set domain adaptation specific parameters
    trainer.discriminator_lr = args.discriminator_lr
    trainer.adv_loss_weight = args.adv_loss_weight

    print("Starting Domain Adaptation Training")
    print(f"Source data: {args.source_data}")
    print(f"Target data: {args.target_data}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Discriminator LR: {args.discriminator_lr}")
    print(f"Adversarial loss weight: {args.adv_loss_weight}")

    # Start training
    trainer.train()

    print("Domain Adaptation Training Complete!")


if __name__ == '__main__':
    main()