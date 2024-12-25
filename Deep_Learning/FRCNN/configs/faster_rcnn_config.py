# configs/faster_rcnn_config.py

import os

BASE_DIR = "/home/omar/Masters/Adv Image and Video Processiong/Project/Dataset/fcrnn_split_data_new"
CONFIG = {
    "train_images_dir": os.path.join(BASE_DIR, "train", "images"),
    "train_annotations": os.path.join(BASE_DIR, "train", "train_annotations.json"),
    "val_images_dir": os.path.join(BASE_DIR, "val", "images"),
    "val_annotations": os.path.join(BASE_DIR, "val", "val_annotations.json"),
    "num_classes": 4, 
    "batch_size": 4,
    "batch_val": 4,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "output_dir": "weights",
}
