import os
import cv2
import numpy as np
from time import time

def extract_signatures(image, min_area=100, max_area=2000, min_aspect_ratio=1.5, max_aspect_ratio=6.0):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    extracted_boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect_ratio = w / h
        if min_area < area < max_area and min_aspect_ratio < aspect_ratio < max_aspect_ratio:
            extracted_boxes.append((x, y, w, h))
    return extracted_boxes

input_directory = "Dataset/40gigs_signatures_imgs"
output_directory = "Outputs/BasicCCA"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

min_area = 500
max_area = 2000
min_aspect_ratio = 2.0
max_aspect_ratio = 8.0

for filename in os.listdir(input_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        t = time()
        signature_boxes = extract_signatures(binary_image, min_area, max_area, min_aspect_ratio, max_aspect_ratio)
        print(f"Processed {filename} in {1000 * (time() - t):.2f} ms")

        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in signature_boxes:
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imwrite(output_path, output_image)
