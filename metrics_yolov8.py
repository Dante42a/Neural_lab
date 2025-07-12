from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8n.pt')

# Классы COCO
CHAIR_CLASS_ID = 56
PERSON_CLASS_ID = 0

def analyze_hall_occupancy(image_path):
    img = cv2.imread(image_path)
    results = model(img)[0]
    
    chairs = []
    people = []
    
    for box in results.boxes:
        cls = int(box.cls)
        if cls == CHAIR_CLASS_ID:
            chairs.append(box.xyxy[0].cpu().numpy())
        elif cls == PERSON_CLASS_ID:
            people.append(box.xyxy[0].cpu().numpy())
    
    # Простая логика: если человек близко к стулу, стул занят
    occupied_chairs = 0
    for chair in chairs:
        chair_center = [(chair[0] + chair[2])/2, (chair[1] + chair[3])/2]
        for person in people:
            person_center = [(person[0] + person[2])/2, (person[1] + person[3])/2]
            distance = np.sqrt((chair_center[0] - person_center[0])**2 + 
                             (chair_center[1] - person_center[1])**2)
            if distance < 100:  # порог расстояния
                occupied_chairs += 1
                break
    
    return len(chairs), occupied_chairs, len(chairs) - occupied_chairs 