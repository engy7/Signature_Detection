import os
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont
import json
import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets.coco import CocoDetection
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter, RandomAffine
from tqdm import tqdm
from clearml import Task
import sys
import logging
import warnings
import sys
import os
# Suppress PyTorch and other library prints
logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# Add the sibling directory to the sys.path
sibling_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs'))
sys.path.append(sibling_dir)
from faster_rcnn_config import CONFIG

# Load the trained model
def load_model(checkpoint_path, num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Perform inference on a single image
def infer_image(model, image_path, device, confidence_threshold):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor)[0]

    return image, predictions

# Draw bounding boxes and labels on the image
def draw_predictions(image, predictions, confidence_threshold):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= confidence_threshold:
            box = [int(coord) for coord in box]
            draw.rectangle(box, outline="red", width=2)
            text = f"{label}: {score:.2f}"
            draw.text((box[0], box[1] - 10), text, fill="red", font=font)

    return image

# Main inference script
def main():
    input_dir = "/home/omar/Masters/Adv Image and Video Processiong/Project/Dataset/fcrnn_split_data_new/test/images"  # Update this path
    output_dir = "/home/omar/Masters/Adv Image and Video Processiong/Project/Deep_Learning/FRCNN/inference_results"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint = "/home/omar/Masters/Adv Image and Video Processiong/Project/Deep_Learning/FRCNN/scripts/weights/best_model_epoch.pth" 
    model = load_model(model_checkpoint, CONFIG["num_classes"])
    model.to(device)

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)

        try:
            image, predictions = infer_image(model, image_path, device, CONFIG["confidence_threshold"])
            output_image = draw_predictions(image, predictions, CONFIG["confidence_threshold"])

            output_path = os.path.join(output_dir, image_name)
            output_image.save(output_path)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    main()
