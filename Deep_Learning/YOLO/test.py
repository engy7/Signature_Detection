import os
from clearml import Task
from ultralytics import YOLO

# Connect to ClearML
Task.init(project_name="YOLOv9 Training", task_name="YOLOv9 Test")

# Load the trained model
model_path = "/home/omar/Masters/Adv Image and Video Processiong/Project/Deep_Learning/YOLO/Round4/yolov9_training/weights/best.pt"  # Replace with the path to your trained model
model = YOLO(model_path)

# Set the dataset configuration (if not done previously)
dataset_config = "/home/omar/Masters/Adv Image and Video Processiong/Project/Deep_Learning/YOLO/data.yaml"

# Directory for saving results and visualizations
output_dir = "/home/omar/Masters/Adv Image and Video Processiong/Project/Deep_Learning/YOLO/Round4"
os.makedirs(output_dir, exist_ok=True)

# Run inference on the test dataset
results = model.predict(source="/home/omar/Masters/Adv Image and Video Processiong/Project/Dataset/split_data/test/images",  # Test image folder
                        save=True,  # Save the predictions
                        project=output_dir,  # Save to the output directory
                        name="test_predictions")

# Save additional metrics and results
metrics = model.val(data=dataset_config)  # Run validation on the test set
print("Test Metrics:", metrics)

# Save the metrics to a text file for later review
metrics_file = os.path.join(output_dir, "metrics.txt")
with open(metrics_file, 'w') as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value}\n")

# Visualizations and predictions will be saved in the project directory
print(f"Results and visualizations are saved in {output_dir}")
