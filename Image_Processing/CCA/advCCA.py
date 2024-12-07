import os
import cv2
from utils.loader import Loader
from utils.extractor import Extractor
from utils.cropper import Cropper

def process_and_save_images_with_bounding_boxes(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    loader = Loader()
    extractor = Extractor(amplfier=15)
    cropper = Cropper()

    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)

            image = cv2.imread(input_path)
            if image is None:
                print(f"Error: Could not read image from path: {input_path}")
                continue

            masks = loader.get_masks(input_path)
            if not masks:
                print(f"Error: No masks generated for {filename}")
                continue

            mask = masks[0]
            labeled_mask = extractor.extract(mask)
            results = cropper.run(labeled_mask)

            if not results:
                print(f"Error: No signatures extracted for {filename}")
                continue

            for idx, result in results.items():
                x, y, w, h = result["cropped_region"]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imwrite(output_path, image)
            print(f"Saved: {output_path}")

input_directory = "Dataset/40gigs_signatures_imgs"
output_directory = "Outputs/AdvCCA"

process_and_save_images_with_bounding_boxes(input_directory, output_directory)

