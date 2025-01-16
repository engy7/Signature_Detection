import torch
from clearml import Task
from ultralytics import YOLO

Task.init(project_name="Signature Detection", task_name="YOLOv9 m Experiment with Clean Data S")

output_dir = "Clean Data S"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("weights/yolov9s.pt").to(device)

dataset_config = "data.yaml"
model.train(
    data=dataset_config,
    epochs=100,
    batch=8,
    imgsz=640,
    optimizer="SGD",
    lr0=0.0001,
    momentum=0.9,
    patience=20,
    weight_decay=0.0005,
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0,
    flipud=0.5,
    fliplr=0.5,
    mosaic=True,
    mixup=True,
    workers=8,
    project=output_dir,
    name="Clean Data S",
    pretrained=True
)

metrics = model.val()
print("Validation Metrics:", metrics)

export_path = f"{output_dir}/best_model.pt"
model.export(format="torchscript", save_dir=export_path)
print(f"Model exported to {export_path}")
