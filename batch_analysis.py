from ultralytics import YOLO
import cv2
import numpy as np
import os

# Модели
pose_model = YOLO('yolov8n-pose.pt')  # для детекции людей и их позы
chair_model = YOLO('yolov8n.pt')      # для детекции стульев

# Классы COCO
CHAIR_CLASS_ID = 56
PERSON_CLASS_ID = 0

def analyze_hall_occupancy_simple(image_path, save_result=True):
    """Упрощённый анализ заполненности зала"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return None
    
    # Детекция стульев
    chair_results = chair_model(img)[0]
    chairs = []
    for box in chair_results.boxes:
        if int(box.cls) == CHAIR_CLASS_ID:
            chairs.append(box.xyxy[0].cpu().numpy())
    
    # Детекция людей (с позой)
    pose_results = pose_model(img)[0]
    people = []
    
    for box in pose_results.boxes:
        if int(box.cls) == PERSON_CLASS_ID:
            person_bbox = box.xyxy[0].cpu().numpy()
            people.append(person_bbox)
    
    # Простая логика: если человек близко к стулу, стул занят
    occupied_chairs = 0
    for chair in chairs:
        chair_center = [(chair[0] + chair[2])/2, (chair[1] + chair[3])/2]
        
        for person in people:
            person_center = [(person[0] + person[2])/2, (person[1] + person[3])/2]
            distance = np.sqrt((chair_center[0] - person_center[0])**2 + 
                             (chair_center[1] - person_center[1])**2)
            
            if distance < 150:  # порог расстояния
                occupied_chairs += 1
                break
    
    # Визуализация результатов
    if save_result:
        result_img = img.copy()
        
        # Рисуем стулья
        for i, chair in enumerate(chairs):
            x1, y1, x2, y2 = map(int, chair)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_img, f'Chair {i+1}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Рисуем людей
        for i, person in enumerate(people):
            x1, y1, x2, y2 = map(int, person)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(result_img, f'Person {i+1}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Добавляем статистику
        stats_text = f'Chairs: {len(chairs)}, Occupied: {occupied_chairs}, Free: {len(chairs) - occupied_chairs}'
        cv2.putText(result_img, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Сохраняем результат
        result_path = f'results_occupancy/{os.path.basename(image_path)}'
        os.makedirs('results_occupancy', exist_ok=True)
        cv2.imwrite(result_path, result_img)
    
    return {
        'total_chairs': len(chairs),
        'occupied_chairs': occupied_chairs,
        'free_chairs': len(chairs) - occupied_chairs,
        'total_people': len(people),
        'occupancy_rate': occupied_chairs / len(chairs) if len(chairs) > 0 else 0
    }

def analyze_batch(images_dir):
    """Анализ всех изображений в папке"""
    results = []
    
    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(images_dir, img_name)
        print(f"Анализирую: {img_name}")
        
        result = analyze_hall_occupancy_simple(img_path)
        if result:
            result['image_name'] = img_name
            results.append(result)
    
    return results

# Пример использования
if __name__ == "__main__":
    # Анализ всех изображений в папке
    images_dir = "furniture.v2-release.yolov8/valid/images"  # замените на путь к вашим изображениям
    results = analyze_batch(images_dir)
    
    print("\nРезультаты анализа:")
    print("-" * 50)
    for result in results:
        print(f"Изображение: {result['image_name']}")
        print(f"  Всего стульев: {result['total_chairs']}")
        print(f"  Занятых стульев: {result['occupied_chairs']}")
        print(f"  Свободных стульев: {result['free_chairs']}")
        print(f"  Всего людей: {result['total_people']}")
        print(f"  Заполненность: {result['occupancy_rate']:.2%}")
        print("-" * 30) 