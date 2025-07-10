from ultralytics import YOLO

YAML_PATH = 'furniture.v2-release.yolov8/data.yaml'
model = YOLO('yolov8n.pt')
metrics = model.val(data=YAML_PATH, split='val')
print(metrics) 