import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets.coco import CocoDetection
from torchvision.transforms import ToTensor, Compose
from tqdm import tqdm
from clearml import Task
import sys
import logging
import warnings

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
        ToTensor()
    ])

def get_data_loader(images_dir, annotations_file, batch_size, shuffle=False):
    dataset = CocoDetection(images_dir, annotations_file, transform=get_transforms())
    
    def collate_fn(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=4)
    return data_loader

def prepare_target(target):
    boxes = [torch.tensor(t['bbox'], dtype=torch.float32) for t in target]
    labels = [t['category_id'] for t in target]
    labels = [min(max(label, 0), CONFIG["num_classes"] - 1) for label in labels]
    boxes = torch.stack([box for box in boxes if box[2] > 0 and box[3] > 0])
    labels = torch.tensor(labels, dtype=torch.int64)

    area = torch.tensor([t['area'] for t in target], dtype=torch.float32)
    iscrowd = torch.tensor([t['iscrowd'] for t in target], dtype=torch.uint8)

    return {
        "boxes": boxes,
        "labels": labels,
        "image_id": torch.tensor([target[0]['image_id']], dtype=torch.int64),
        "area": area,
        "iscrowd": iscrowd
    }

def validate(model, data_loader, device):
    model.train()
    total_loss = 0.0
    with tqdm(data_loader, desc="Validation", unit="batch") as pbar:
        for images, targets in pbar:
            with torch.no_grad():
                targets = [prepare_target(t) for t in targets]
                images = [image.to(device) for image in images]
                for i in range(len(targets)):
                    targets[i] = {k: v.to(device) for k, v in targets[i].items()}
                with suppress_stdout_stderr():
                    loss_dict = model(images, targets)
                total_loss += sum(loss.item() for loss in loss_dict.values())
                pbar.set_postfix(loss=sum(loss.item() for loss in loss_dict.values()))
    return total_loss / len(data_loader)

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resuming training from epoch {start_epoch} with best validation loss: {best_val_loss}")
        return model, optimizer, start_epoch, best_val_loss
    else:
        print("No checkpoint found, starting training from scratch.")
        return model, optimizer, 0, float('inf')

def main():
    task = Task.init(project_name="FasterRCNN Training", task_name="Training FasterRCNN with COCO Format")
    train_loader = get_data_loader(CONFIG['train_images_dir'], CONFIG['train_annotations'], CONFIG['batch_size'], shuffle=True)
    val_loader = get_data_loader(CONFIG['val_images_dir'], CONFIG['val_annotations'], CONFIG['batch_val'], shuffle=False)

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, CONFIG["num_classes"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    checkpoint_path = os.path.join(CONFIG["output_dir"], "best_model.pth")
    model, optimizer, start_epoch, best_val_loss = load_checkpoint(model, optimizer, checkpoint_path)

    for epoch in range(start_epoch, CONFIG["num_epochs"]):
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

        val_loss = validate(model, val_loader, device)
        print(f'Validation loss: {val_loss}')
        if val_loss is not None:
            task.get_logger().report_scalar("Validation Loss", "Validation", iteration=epoch + 1, value=val_loss)

        scheduler.step(val_loss)

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(CONFIG["output_dir"], exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}")

if __name__ == "__main__":
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    main()
