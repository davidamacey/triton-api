from ultralytics import YOLO

# Load the Triton Server model
model = YOLO("http://127.0.0.1:8010/yolov11_nano", task="detect")

img_path = "/mnt/nvm/killboy_sm_ai/img_0248.jpg"

# Run inference on the server
for _ in range(5):
    results = model(img_path)

# print(results)