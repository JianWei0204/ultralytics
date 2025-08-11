from ultralytics import YOLO

# 加载模型（可以是 yaml 配置或预训练权重）
model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')

# 训练模型
results = model.train(
    data='ultralytics/cfg/datasets/soda10m.yaml',
    epochs=100,
    batch=16,
    lr0=0.01,
    pretrained='yolov8n.pt',  # 使用预训练权重
    resume=True                # 是否恢复训练
)