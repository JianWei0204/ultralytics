# Source domain dataset configuration (e.g., COCO format)
source_config = {
    'path': '/root/autodl-tmp/v8-bytetrack/yolov8/datasets/soda10m',
    'train': 'images/train',
    'val': 'images/val',
    'nc': 6,  # number of classes
    'names': ['Pedestrain','Cyclist', 'Car', 'Truck', 'Tram', 'Tricycle'],  # class names
}

# Target domain dataset configuration
target_config = {
    'path': '/root/autodl-tmp/v8-bytetrack/yolov8/datasets/soda10m',
    'train': 'images/train',
    'val': 'images/val',
    'nc': 6,  # should match source domain
    'names': ['Pedestrain','Cyclist', 'Car', 'Truck', 'Tram', 'Tricycle']  # should match source domain
}

# Domain adaptation training parameters
domain_adaptation_config = {
    # Basic training parameters
    'model': 'yolov8n.pt',
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'device': '0',

    # Domain adaptation specific parameters
    'discriminator_lr': 1e-4,
    'adv_loss_weight': 0.1,
    'discriminator_channels': 512,  # P4 feature channels

    # Learning rate and optimization
    'lr0': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,

    # Output configuration
    'project': 'runs/domain_adapt',
    'name': 'exp',
    'save_period': 10,
    'val': True,
    'plots': True,
}

# Example usage:
# python train_domain_adaptation.py \
#   --source_data source_dataset.yaml \
#   --target_data target_dataset.yaml \
#   --model yolov8n.pt \
#   --epochs 100 \
#   --batch 16 \
#   --discriminator_lr 1e-4 \
#   --adv_loss_weight 0.1