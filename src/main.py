from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()

# Dictionary to store model endpoints for different YOLO model versions
MODEL_ENDPOINTS = {
    "nano": YOLO("http://triton-api:8000/yolov11_nano", task="detect"),
    "small": YOLO("http://triton-api:8000/yolov11_small", task="detect"),
    "medium": YOLO("http://triton-api:8000/yolov11_medium", task="detect"),
}

class InferenceResult(BaseModel):
    """
    Schema for the response returned by the endpoint.

    Attributes:
        detections (list): List of detections with bounding box coordinates, confidence, and class.
        status (str): Status of the inference process.
    """
    
    detections: list
    status: str

@app.post("/predict/{model_name}", response_model=InferenceResult)
async def predict(
    model_name: str = 'nano',
    image: UploadFile = File(...),
):
    """
    Endpoint to perform object detection on an uploaded image using the specified YOLO model.

    Args:
        model_name (str): The name of the model to use for inference. Defaults to 'nano'.
        image (UploadFile): The image file uploaded by the user.

    Returns:
        dict: A dictionary containing detection results and status.

    Raises:
        HTTPException: If the model name is invalid or an error occurs during inference.
    """
    
    if model_name not in MODEL_ENDPOINTS:
        raise HTTPException(status_code=400, detail="Invalid model name.")
    
    model = MODEL_ENDPOINTS[model_name]

    try:
        # Read and decode the image
        image_data = await image.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image.")

        # Perform object detection with YOLOv8
        detections = model(img)
        
        # Extract bounding box data
        boxes = detections[0].boxes.xyxy.cpu().numpy()
        scores = detections[0].boxes.conf.cpu().numpy()
        classes = detections[0].boxes.cls.cpu().numpy()

        # Format the results as a list of dictionaries
        results = []
        for box, score, cls in zip(boxes, scores, classes):
            results.append({
                'x1': float(box[0]),
                'y1': float(box[1]),
                'x2': float(box[2]),
                'y2': float(box[3]),
                'confidence': float(score),
                'class': int(cls)
            })

        return {
            "detections": results,
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))