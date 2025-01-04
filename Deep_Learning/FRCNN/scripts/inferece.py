import os
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont
import json
import logging
import warnings
import sys
import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from clearml import Task
import numpy as np

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

# Compute Intersection over Union (IoU)
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1_area + box2_area - intersection

    if union == 0:
        return 0.0

    return intersection / union

# Filter predictions using IoU to keep only the highest confidence box per class in overlapping regions
def filter_predictions(predictions, iou_threshold):
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']

    # Convert to NumPy arrays for easier manipulation
    boxes = boxes.cpu().numpy()
    labels = labels.cpu().numpy()
    scores = scores.cpu().numpy()

    # Sort indices by scores in descending order
    sorted_indices = np.argsort(-scores)
    boxes = boxes[sorted_indices]
    labels = labels[sorted_indices]
    scores = scores[sorted_indices]

    # Keep track of selected boxes
    keep = []
    selected_classes = {}

    for i, box in enumerate(boxes):
        label = labels[i]
        if label not in selected_classes:
            selected_classes[label] = []

        # Check if this box overlaps significantly with any selected box of the same class
        overlaps = [
            compute_iou(box, selected_box) > iou_threshold
            for selected_box in selected_classes[label]
        ]

        # If no significant overlap, keep the box
        if not any(overlaps):
            keep.append(i)
            selected_classes[label].append(box)
        else:
            # If there's an overlap, find the box with the highest score
            max_score = max(scores[sorted_indices][np.where(np.array(overlaps))[0]])
            if scores[i] > max_score:
                # If the current box has a higher score, replace the box with the highest score
                idx = np.where(np.array(overlaps))[0][np.argmax(scores[sorted_indices][np.where(np.array(overlaps))[0]])]
                selected_classes[label][idx] = box
                keep.append(i)

    # Prepare filtered predictions
    filtered_predictions = {
        'boxes': torch.tensor(boxes[keep]),
        'labels': torch.tensor(labels[keep]),
        'scores': torch.tensor(scores[keep])
    }

    return filtered_predictions



# Draw bounding boxes and labels on the image for a specific class
def draw_predictions(image, predictions, confidence_threshold, target_class_id):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= confidence_threshold :
            box = [int(coord) for coord in box]
            draw.rectangle(box, outline="red", width=2)
            text = f"{label}: {score:.2f}"
            draw.text((box[0], box[1] - 10), text, fill="red", font=font)

    return image

import json

# Save results in COCO format
def save_coco_results(predictions, image_id, results_list):
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']
    # print(predictions)
    n = 0
    for box, label, score in zip(boxes, labels, scores):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        if float(score) >=  CONFIG["confidence_threshold"]:
            n+=1
            results_list.append({
                "image_id": image_id,
                "category_id": int(label),  # COCO uses integers for category IDs
                "bbox": [float(x_min), float(y_min), float(width), float(height)],
                "score": float(score),
            })
    # print(f'preds: {n}')
    

# Main inference script
def main():
    input_dir = "/home/omar/Masters/Adv Image and Video Processiong/Project/Dataset/fcrnn_split_data_new/test/images"
    output_dir = "/home/omar/Masters/Adv Image and Video Processiong/Project/Deep_Learning/FRCNN/inference_results4"
    results_file = os.path.join(output_dir, "results.json")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint = "/home/omar/Masters/Adv Image and Video Processiong/Project/Deep_Learning/FRCNN/weights/best_model.pth"
    model = load_model(model_checkpoint, CONFIG["num_classes"])
    model.to(device)

    iou_threshold = 0  # Adjust as needed
    results_list = []  # Store predictions in COCO format

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        image_id = image_name # Assuming the image name is its ID

        try:
            image, predictions = infer_image(model, image_path, device, CONFIG["confidence_threshold"])
            # print(f'preds before: {len(predictions["boxes"])}')
            predictions = filter_predictions(predictions, iou_threshold)
            # print(f'preds after: {len(predictions["boxes"])}')

            # Save predictions in COCO format
            # print(f'preds: {len(predictions["boxes"])}')
            # print(f'res before: {len(results_list)}')
            # print(len(results_list))
            save_coco_results(predictions, image_id, results_list)
            # print(f'res after: {len(results_list)}')
            # print(len(results_list))

            # Draw predictions (optional)
            output_image = draw_predictions(image, predictions, CONFIG["confidence_threshold"], target_class_id=3)
            output_path = os.path.join(output_dir, image_name)
            output_image.save(output_path)

            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # Save results to JSON file
    with open(results_file, "w") as f:
        json.dump(results_list, f, indent=4)
    print(f"Results saved in COCO format to: {results_file}")

if __name__ == "__main__":
    main()
