import os
import cv2
import numpy as np

def load_image(image_path: str) -> np.ndarray:
    """Loads an image in grayscale mode."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Error loading image from path: {image_path}")
    return image

def preprocess_image(image: np.ndarray, threshold: int = 150, blur_kernel: tuple = (5, 5)) -> np.ndarray:
    """Applies thresholding and Gaussian blur to an image."""
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    blurred_image = cv2.GaussianBlur(binary_image, blur_kernel, 0)
    return blurred_image

def detect_contours(image: np.ndarray) -> list:
    """Detects contours in a binary image."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_contours(contours: list, min_area: int = 500, max_area: int = 5000, min_aspect_ratio: float = 1.0, max_aspect_ratio: float = 10.0) -> list:
    """
    Filters contours based on area and aspect ratio criteria.

    Args:
        contours (list): List of contours.
        min_area (int): Minimum area of the bounding box.
        max_area (int): Maximum area of the bounding box.
        min_aspect_ratio (float): Minimum aspect ratio (width/height).
        max_aspect_ratio (float): Maximum aspect ratio (width/height).

    Returns:
        list: Filtered contours meeting the criteria.
    """
    filtered = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / h
        if min_area <= area <= max_area and min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
            filtered.append(contour)
    return filtered

def draw_bounding_boxes(image: np.ndarray, contours: list, color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draws bounding boxes on the original image."""
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, thickness)
    return output_image

def process_image(image_path: str, output_dir: str, min_area: int = 500, max_area: int = 5000, min_aspect_ratio: float = 1.0, max_aspect_ratio: float = 10.0):
    """Processes a single image to detect and draw bounding boxes around signature regions."""
    # Step 1: Load the image
    image = load_image(image_path)

    # Step 2: Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Step 3: Detect contours
    contours = detect_contours(preprocessed_image)

    # Step 4: Filter contours based on area and aspect ratio
    filtered_contours = filter_contours(contours, min_area, max_area, min_aspect_ratio, max_aspect_ratio)

    # Step 5: Draw bounding boxes on the original image
    output_image_with_boxes = draw_bounding_boxes(image, filtered_contours)

    # Save results
    base_name = os.path.basename(image_path)
    cv2.imwrite(os.path.join(output_dir, f"bbox_{base_name}"), output_image_with_boxes)

def process_directory(input_dir: str, output_dir: str, min_area: int = 500, max_area: int = 5000, min_aspect_ratio: float = 1.0, max_aspect_ratio: float = 10.0):
    """
    Loops through all images in the input directory and applies the signature detection pipeline.

    Args:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory for saving output images.
        min_area (int): Minimum area for bounding boxes.
        max_area (int): Maximum area for bounding boxes.
        min_aspect_ratio (float): Minimum aspect ratio.
        max_aspect_ratio (float): Maximum aspect ratio.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            try:
                print(f"Processing: {filename}")
                process_image(file_path, output_dir, min_area, max_area, min_aspect_ratio, max_aspect_ratio)
            except Exception as e:
                print(f"Error processing {filename}: {e}")


input_directory = "Dataset/40gigs_signatures_imgs"
output_directory = "Outputs/Contours"


# Define area and aspect ratio ranges for filtering bounding boxes
min_area = 1000
max_area = 3000
min_aspect_ratio = 0.5
max_aspect_ratio = 2.0

process_directory(input_directory, output_directory, min_area, max_area, min_aspect_ratio, max_aspect_ratio)

