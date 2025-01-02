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

# Suppress prints by redirecting stdout to null device
class suppress_stdout_stderr(object):
    def __init__(self):
        self.nullfile = open(os.devnull, 'w')
    def __enter__(self):
        self.saved_stdout = sys.stdout
        sys.stdout = self.nullfile
        self.saved_stderr = sys.stderr
        sys.stderr = self.nullfile
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.saved_stdout
        sys.stderr = self.saved_stderr


# Suppress PyTorch and other library prints
logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# Add the sibling directory to the sys.path
sibling_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs'))
sys.path.append(sibling_dir)
from faster_rcnn_config import CONFIG

# Define the data transformation pipeline with augmentation
def get_transforms():
    return Compose([
        # RandomHorizontalFlip(0.5),
        # RandomVerticalFlip(0.5),
        # RandomRotation(30),
        # RandomAffine(15, shear=10),
        # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        ToTensor()
    ])

def get_data_loader(images_dir, annotations_file, batch_size, shuffle=False):
    # Apply transformations to images
    dataset = CocoDetection(images_dir, annotations_file, transform=get_transforms())
    
    def collate_fn(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=16)
    return data_loader

def prepare_target(target):
    # Extract bounding boxes, categories, image_ids, and other necessary information
    boxes = [torch.tensor(t['bbox'], dtype=torch.float32) for t in target]
    labels = [t['category_id'] for t in target]
    
    # Ensure labels are within the range of classes
    labels = [min(max(label, 0), CONFIG["num_classes"] - 1) for label in labels]
    
    # Proceed with the rest of the processing
    boxes = torch.stack(boxes)
    labels = torch.tensor(labels, dtype=torch.int64)
    
    # Validate and adjust boxes if needed
    valid_boxes = []
    for box in boxes:
        x_min, y_min, width, height = box
        if width > 0 and height > 0:
            valid_boxes.append(box)
        else:
            # Correct invalid bounding boxes
            corrected_box = torch.tensor([x_min, y_min, max(width, 0.01), max(height, 0.01)], dtype=torch.float32)
            valid_boxes.append(corrected_box)
    boxes = torch.stack(valid_boxes)
    
    area = [t['area'] for t in target]
    iscrowd = [t['iscrowd'] for t in target]

    area = torch.tensor(area, dtype=torch.float32)
    iscrowd = torch.tensor(iscrowd, dtype=torch.uint8)

    return {
        "boxes": boxes,
        "labels": labels,
        "image_id": torch.tensor([target[0]['image_id']], dtype=torch.int64),  # Single image_id for the batch
        "area": area,
        "iscrowd": iscrowd
    }

def validate(model, data_loader, device):
    model.train()  # Temporarily switch to training mode for loss calculation
    total_loss = 0.0

    with tqdm(data_loader, desc=f"Validation", unit="batch") as pbar:
        for images, targets in pbar:
            with torch.no_grad():  # Disable gradient calculation during inference
                targets = [prepare_target(t) for t in targets]

                images = [image.to(device) for image in images]
                for i in range(len(targets)):
                    targets[i] = {k: v.to(device) for k, v in targets[i].items()}

                with suppress_stdout_stderr():
                    loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                total_loss += losses.item()
                pbar.set_postfix(loss=losses.item())

    average_loss = total_loss / len(data_loader)
    return average_loss

def main():
    # Initialize ClearML task
    task = Task.init(project_name="FasterRCNN Training", task_name="Training FasterRCNN with COCO Format")

    # Load configuration
    train_loader = get_data_loader(CONFIG['train_images_dir'], CONFIG['train_annotations'], CONFIG['batch_size'], shuffle=True)
    val_loader = get_data_loader(CONFIG['val_images_dir'], CONFIG['val_annotations'], CONFIG['batch_val'], shuffle=False)

    # Load model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, CONFIG["num_classes"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
    
    # Use ReduceLROnPlateau scheduler to reduce the learning rate when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)


    best_val_loss = float("inf")
    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        total_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['num_epochs']}", unit="batch") as pbar:
            for images, targets in pbar:
                targets = [prepare_target(t) for t in targets]

                images = [image.to(device) for image in images]

                for i in range(len(targets)):
                    targets[i] = {k: v.to(device) for k, v in targets[i].items()}

                with suppress_stdout_stderr():
                    loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                total_loss += losses.item()
                pbar.set_postfix(loss=losses.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}], Training Loss: {avg_loss:.4f}")
        task.get_logger().report_scalar("Training Loss", "Training", iteration=epoch + 1, value=avg_loss)

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        task.get_logger().report_scalar("Learning Rate", "Training", iteration=epoch + 1, value=current_lr)

        # Validate the model
        val_loss = validate(model, val_loader, device)
        print(f'Validation loss: {val_loss}')
        if val_loss is not None:
            task.get_logger().report_scalar("Validation Loss", "Validation", iteration=epoch + 1, value=val_loss)

        # Use the ReduceLROnPlateau scheduler to adjust the learning rate
        scheduler.step(val_loss)

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(CONFIG["output_dir"], exist_ok=True)
            try:
                checkpoint_path = os.path.join(CONFIG["output_dir"], f"best_model_epoch.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss,
                }, checkpoint_path)
                print(f"New best model saved with validation loss: {val_loss:.4f}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")


if __name__ == "__main__":
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    main()