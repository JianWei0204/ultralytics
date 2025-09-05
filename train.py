# from ultralytics import YOLO
#
# model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
# # model = YOLO('yolov8/ultralytics/cfg/models/v8/yolov8.yaml')  # 使用YOLOv8n模型配置
# # 训练模型
# results = model.train(
#     data='ultralytics/cfg/datasets/soda10m.yaml',
#     epochs=100,
#     batch=16,
#     lr0=0.01,
#     resume=True                # 是否恢复训练
# )
#
from ultralytics import YOLO

# 加载训练好的模型
# 方式1：加载最佳训练权重
model = YOLO('runs/detect/train/weights/best.pt')  # 使用训练得到的最佳权重

# 方式2：如果没有训练权重，可以使用预训练权重
# model = YOLO('yolov8n.pt')  # 使用预训练权重进行测试

# 在测试集上进行验证
results = model.val(
    data='ultralytics/cfg/datasets/target_dataset.yaml',  # 使用相同的数据集配置
    split='val',  # 使用测试集，也可以是 'val' 使用验证集
    imgsz=640,     # 图像大小
    batch=16,      # 批次大小
    conf=0.25,     # 置信度阈值
    iou=0.7,       # NMS IoU阈值
    save_json=True, # 保存COCO格式的结果
    save_txt=True,  # 保存txt格式的检测结果
    plots=True,     # 生成混淆矩阵等图表
    verbose=True    # 详细输出
)
