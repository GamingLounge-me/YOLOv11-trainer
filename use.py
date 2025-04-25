from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('models/best.pt')  # Path to the trained model

# folder and image supported
results = model.predict(source='input/images/', save=True, imgsz=640)
