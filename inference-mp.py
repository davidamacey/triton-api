import os
from ultralytics import YOLO
import concurrent.futures
import cv2
from itertools import repeat

def detect_image(img_path):
    """
    Detect objects in an image using the provided YOLO model.
    
    Args:
        model (YOLO): The YOLO model to use for detection.
        img_path (str): The path to the image file.

    Returns:
        list: A list of detected objects with their bounding box coordinates and class labels.
    """
    
    # Load the Triton Server model
    model = YOLO("http://127.0.0.1:8010/yolo", task="detect")
    # Run inference on the server
    return model(img_path)

def process_images(images_folder):
    """
    Process all images in a folder using the provided YOLO model.

    Args:
        model (YOLO): The YOLO model to use for detection.
        images_folder (str): The path to the folder containing the image files.

    Returns:
        list: A list of lists, where each sublist contains the detected objects and their bounding box coordinates and class labels for a single image file.
    """
    # Get all image file paths in the folder
    img_paths = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')]
    
    # Process each image in parallel using multiprocessing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results_list = list(executor.map(detect_image, img_paths[:50]))
    
    return results_list

if __name__ == "__main__":
    
    # Define the path to the images folder
    images_folder = "/mnt/nvm/killboy_sm_ai/"

    # Process all images in the folder using multiprocessing
    results_list = process_images(images_folder)

    # Print the detected objects for each image
    # for i, result in enumerate(results_list):
    #     print(f"Image {i+1}:")
    #     for obj in result:
    #         print(obj)