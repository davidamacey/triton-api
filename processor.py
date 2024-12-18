import os
import cv2
import numpy as np
import asyncio
import aiohttp
from tqdm import tqdm

FASTAPI_URL = "http://localhost:8200/predict/nano"  # Replace 'nano' with desired model name
IMAGE_DIRECTORY = "/mnt/nvm/killboy_sm_ai"  # Path to the folder containing images

def resize_image(image, target_size=640):
    """
    Resize image to the target size while maintaining aspect ratio.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        target_size (int): The target size for the largest dimension of the image.

    Returns:
        numpy.ndarray: The resized image.
    """
    
    height, width = image.shape[:2]
    scale = target_size / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

async def process_image(session, file_path, target_size=640):
    """
    Process a single image by resizing it and sending it to a FastAPI endpoint for inference.

    Args:
        session (aiohttp.ClientSession): The aiohttp session used to send HTTP requests.
        file_path (str): The path to the image file to process.
        target_size (int): The target size for resizing the image. Defaults to 640.

    Returns:
        tuple: A tuple containing the file path and the result or error message from the API.
    """
    
    try:
        # Read and resize image
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Failed to read image file: {file_path}")
        resized_image = resize_image(image, target_size)

        # Encode resized image
        _, img_encoded = cv2.imencode('.jpg', resized_image)

        # Prepare payload with additional parameters
        payload = aiohttp.FormData()
        payload.add_field('image', img_encoded.tobytes(), filename=os.path.basename(file_path), content_type='image/jpeg')
        payload.add_field('model_name', str('small'))

        # Send image to FastAPI endpoint
        async with session.post(FASTAPI_URL, data=payload) as response:
            if response.status == 200:
                result = await response.json()
                return file_path, result
            else:
                return file_path, f"Error: {response.status}"
    except Exception as e:
        return file_path, str(e)


async def process_images_concurrently():
    """
    Process all images in a specified directory concurrently by sending them to a FastAPI endpoint.

    Returns:
        list: A list of results containing the file path and the corresponding inference result or error message.
    """
    
    tasks = []
    async with aiohttp.ClientSession() as session:
        for file_name in os.listdir(IMAGE_DIRECTORY):
            file_path = os.path.join(IMAGE_DIRECTORY, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                tasks.append(process_image(session, file_path))

        results = []
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Images"):
            result = await future
            results.append(result)

        return results

if __name__ == "__main__":
    results = asyncio.run(process_images_concurrently())

    for file_path, result in results:
        print(f"Image: {file_path} -> Result: {result}")
