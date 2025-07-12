from ultralytics import YOLO
import cv2
import numpy as np
import os

# Модели
pose_model = YOLO('yolov8n-pose.pt')  # для детекции позы людей
chair_model = YOLO('yolov8n.pt')      # для детекции стульев

# Классы COCO
CHAIR_CLASS_ID = 56
PERSON_CLASS_ID = 0

def is_sitting(pose_keypoints):
    """Определяет, сидит ли человек, анализируя ключевые точки позы"""
    if pose_keypoints is None or len(pose_keypoints) == 0:
        return False
    
    # Ключевые точки: 0-нос, 11-левое плечо, 12-правое плечо, 23-левое бедро, 24-правое бедро
    if len(pose_keypoints) < 25:  # проверяем, что есть достаточно ключевых точек
        return False
        
    nose = pose_keypoints[0] if pose_keypoints[0] is not None and pose_keypoints[0][2] > 0.5 else None
    left_hip = pose_keypoints[23] if pose_keypoints[23] is not None and pose_keypoints[23][2] > 0.5 else None
    right_hip = pose_keypoints[24] if pose_keypoints[24] is not None and pose_keypoints[24][2] > 0.5 else None
    
    if left_hip is None and right_hip is None:
        return False
    
    # Если бедра ниже носа (по Y координате), человек скорее всего сидит
    if nose and (left_hip or right_hip):
        hip_y = left_hip[1] if left_hip else right_hip[1]
        return hip_y > nose[1]
    
    return False

def analyze_hall_occupancy(image_path, save_result=True):
    """Анализирует заполненность зала по изображению"""
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
    
    # Детекция позы людей
    pose_results = pose_model(img)[0]
    sitting_people = []
    standing_people = []
    
    for box in pose_results.boxes:
        if int(box.cls) == PERSON_CLASS_ID:
            person_bbox = box.xyxy[0].cpu().numpy()
            keypoints = box.keypoints.cpu().numpy()[0] if box.keypoints is not None else None
            
            if is_sitting(keypoints):
                sitting_people.append(person_bbox)
            else:
                standing_people.append(person_bbox)
    
    # Определение занятых стульев
    occupied_chairs = 0
    for chair in chairs:
        chair_center = [(chair[0] + chair[2])/2, (chair[1] + chair[3])/2]
        
        # Проверяем сидящих людей
        for person in sitting_people:
            person_center = [(person[0] + person[2])/2, (person[1] + person[3])/2]
            distance = np.sqrt((chair_center[0] - person_center[0])**2 + 
                             (chair_center[1] - person_center[1])**2)
            
            # Если сидящий человек близко к стулу, считаем стул занятым
            if distance < 150:  # увеличенный порог для учёта позы
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
        
        # Рисуем сидящих людей
        for person in sitting_people:
            x1, y1, x2, y2 = map(int, person)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(result_img, 'Sitting', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Рисуем стоящих людей
        for person in standing_people:
            x1, y1, x2, y2 = map(int, person)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(result_img, 'Standing', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
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
        'sitting_people': len(sitting_people),
        'standing_people': len(standing_people),
        'occupancy_rate': occupied_chairs / len(chairs) if len(chairs) > 0 else 0
    }

# Пример использования
if __name__ == "__main__":
    # Анализ одного изображения
    image_path = "path/to/your/hall_image.jpg"  # замените на путь к вашему изображению
    result = analyze_hall_occupancy(image_path)
    
    if result:
        print(f"Результаты анализа:")
        print(f"Всего стульев: {result['total_chairs']}")
        print(f"Занятых стульев: {result['occupied_chairs']}")
        print(f"Свободных стульев: {result['free_chairs']}")
        print(f"Сидит людей: {result['sitting_people']}")
        print(f"Стоит людей: {result['standing_people']}")
        print(f"Заполненность: {result['occupancy_rate']:.2%}") 