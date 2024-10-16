from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')
results = model.train(data='/home/robot/MlSoft/yolov7/data', epochs=25, imgsz=224, batch=8)
model.save('yolov8n-cls-trained.pt')
