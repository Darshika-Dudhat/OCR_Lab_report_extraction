from ultralytics import YOLO

model = YOLO('yolov3.pt')

model.train(
    data = "data.yaml",
    epochs = 100,
    batch = 8,
    imgsz = 640,
    device = "cpu",
    project = "models",
    name =  "lab_table_ocr",
    exist_ok=True               # Optional: avoids error if folder already exists
)
