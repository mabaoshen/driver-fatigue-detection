from ultralytics import YOLO
import torch
import os

# 数据集配置文件路径（需确保 dataset/data.yaml 存在且配置正确）
data_yaml_path = "dataset/data.yaml"

# 检查数据集配置文件是否存在
if not os.path.exists(data_yaml_path):
    raise FileNotFoundError(f"数据集配置文件 {data_yaml_path} 不存在，请检查路径")

# 检查是否有可用的 GPU
device = 0 if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("警告：未检测到可用的 GPU，将使用 CPU 训练，训练速度可能较慢")
else:
    print(f"将使用 GPU 编号 {device} 进行训练")

# 加载预训练模型（这里以 yolov8n.pt 为例，可根据需求更换为 yolov8s.pt 等其他模型）
model = YOLO("yolov8n.pt")

# 开始训练
results = model.train(
    data=data_yaml_path,  # 数据集配置文件
    epochs=100,           # 训练轮数，可根据需求调整
    batch=16,             # 批次大小，根据 GPU 显存调整，显存小则调小
    device=device,        # 指定使用的设备，0 表示使用第 1 块 GPU，若有多个 GPU 可指定其他编号或用列表 [0,1]
    imgsz=640,            # 输入图像大小
    workers=4,            # 数据加载线程数
    project="runs/train", # 训练结果保存的项目目录
    name="fatigue_detection"  # 训练任务名称，结果会保存在 runs/train/fatigue_detection 下
)

print("训练完成！训练结果保存在 runs/train/fatigue_detection 目录下")