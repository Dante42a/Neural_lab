from ultralytics import YOLO
import cv2
import os
def get_chair_class_id(yaml_path):
    with open(yaml_path, 'r') as f:
        for line in f:
            if 'names:' in line:
                break
        for line in f:
            if 'chair' in line:
                return int(line.split(':')[0].strip())
    return 0  # fallback

IMAGES_DIR = 'furniture.v2-release.yolov8/valid/images'
RESULTS_DIR = 'results_valid'
YAML_PATH = 'furniture.v2-release.yolov8/data.yaml'
os.makedirs(RESULTS_DIR, exist_ok=True)

chair_class_id = get_chair_class_id(YAML_PATH)
model = YOLO('yolov8n.pt')

for img_name in os.listdir(IMAGES_DIR):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    img_path = os.path.join(IMAGES_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f'Не удалось загрузить изображение: {img_path}')
        continue
    results = model(img)[0]
    chairs = [b for b in results.boxes if int(b.cls) == chair_class_id]
    for b in chairs:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, 'Chair', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.imwrite(os.path.join(RESULTS_DIR, img_name), img) 