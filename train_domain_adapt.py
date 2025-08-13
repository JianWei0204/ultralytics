
import argparse

from domain_adapt_trainer import DomainAdaptTrainer
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8 Domain Adaptation Training')
    parser.add_argument('--cfg', type=str, default='ultralytics/cfg/models/yolov8-translayer.yaml', help='model.yaml path')
    parser.add_argument('--source-data', type=str, default='/root/autodl-tmp/ultralytics/cfg/datasets/source_dataset.yaml', help='source domain dataset.yaml path')
    parser.add_argument('--target-data', type=str, default='/root/autodl-tmp/ultralytics/cfg/datasets/target_dataset.yaml', help='target domain dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--disc-lr', type=float, default=0.001, help='initial discriminator learning rate')
    parser.add_argument('--weights', type=str, default=None, help='initial weights path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-period', type=int, default=10, help='Save checkpoint every x epochs')
    return parser.parse_args()


def main():
    args = parse_args()

    # 标准YOLOv8参数 - 只使用官方支持的参数
    train_args = {
        'model': args.cfg,
        'data': args.source_data,  # 使用源域数据作为主要训练数据
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.imgsz,
        'lr0': args.lr0,
        'device': args.device,
        'save_period': args.save_period
    }

    if args.weights:
        train_args['weights'] = args.weights
    # 创建cfg对象
    cfg = get_cfg(DEFAULT_CFG, {})  # 创建一个空的配置

    # 创建并初始化域适应训练器
    trainer = DomainAdaptTrainer(cfg=cfg,overrides=train_args)

    # 单独设置域适应特定参数
    trainer.setup_domain_adaptation(
        target_data=args.target_data,
        disc_lr=args.disc_lr
    )

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()