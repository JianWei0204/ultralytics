from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
# model = YOLO('yolov8/ultralytics/cfg/models/v8/yolov8.yaml')  # 使用YOLOv8n模型配置
# 训练模型
results = model.train(
    data='ultralytics/cfg/datasets/soda10m.yaml',
    epochs=100,
    batch=16,
    lr0=0.01,
    pretrained='yolov8n.pt',  # 使用预训练权重
    resume=True                # 是否恢复训练
)