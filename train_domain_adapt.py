from ultralytics import YOLO
import torch
import argparse
import os
from domain_adapt_trainer import DomainAdaptTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8 Domain Adaptation Training')
    parser.add_argument('--cfg', type=str, default='ultralytics/cfg/models/yolov8-translayer.yaml', help='model.yaml path')
    parser.add_argument('--source-data', type=str, default='/root/autodl-tmp/v8-bytetrack/yolov8/ultralytics/cfg/datasets/source_dataset.yaml', help='source domain dataset.yaml path')
    parser.add_argument('--target-data', type=str, default='/root/autodl-tmp/v8-bytetrack/yolov8/ultralytics/cfg/datasets/target_dataset.yaml', help='target domain dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lr0-d', type=float, default=0.001, help='initial discriminator learning rate')
    parser.add_argument('--weights', type=str, default=None, help='initial weights path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-period', type=int, default=10, help='Save checkpoint every x epochs')
    return parser.parse_args()


def main():
    args = parse_args()

    # Create combined data dictionary with source and target domains
    data = {
        'source_domain': args.source_data,
        'target_domain': args.target_data
    }

    # Configure training arguments
    train_args = {
        'model': args.cfg,
        'data': data,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.imgsz,
        'lr0': args.lr0,
        'lr0_d': args.lr0_d,
        'device': args.device,
        'source_domain': 'source_domain',
        'target_domain': 'target_domain',
        'save_period': args.save_period
    }

    if args.weights:
        train_args['weights'] = args.weights

    # Initialize and run domain adaptation trainer
    trainer = DomainAdaptTrainer(overrides=train_args)
    trainer.train()


if __name__ == '__main__':
    main()