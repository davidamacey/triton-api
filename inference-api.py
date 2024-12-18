from ultralytics import YOLO
import cv2
import numpy as np

# Load the Triton Server model
model = YOLO("http://127.0.0.1:8010/yolov11_nano", task="detect")

img_path = "./IMGS_TEST/img_0248.jpg"

def resize_image(image, target_size=640):
    """Resize image to the target size while maintaining aspect ratio."""
    height, width = image.shape[:2]
    scale = target_size / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

# Read and resize image
image = cv2.imread(img_path)
if image is None:
    raise ValueError(f"Failed to read image file: {img_path}")
resized_image = resize_image(image, 640)

# Encode resized image
_, img_encoded = cv2.imencode('.jpg', resized_image)

img_bytes = img_encoded.tobytes()

# Read and decode the image
nparr = np.frombuffer(img_bytes, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
print(img.shape)
if img is None:
    raise ValueError("Failed to decode image.")

print(type(img))

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
        'x1': box[0],
        'y1': box[1],
        'x2': box[2],
        'y2': box[3],
        'confidence': score,
        'class': int(cls)
    })


print(results)