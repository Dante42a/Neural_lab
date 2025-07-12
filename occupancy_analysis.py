from ultralytics import YOLO
import cv2
import os
import numpy as np

# Модель для детекции занятости
model = YOLO('yolov8n.pt')

# Классы из нового датасета
OCCUPIED_CLASS_ID = 0  # "Non-empty"
EMPTY_CLASS_ID = 1     # "empty"

def analyze_occupancy(image_path, save_result=True):
    """Анализ заполненности зала по новому датасету"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return None
    
    # Детекция занятых и свободных мест
    results = model(img)[0]
    occupied = []
    empty = []
    
    for box in results.boxes:
        cls = int(box.cls)
        bbox = box.xyxy[0].cpu().numpy()
        
        if cls == OCCUPIED_CLASS_ID:
            occupied.append(bbox)
        elif cls == EMPTY_CLASS_ID:
            empty.append(bbox)
    
    # Визуализация результатов
    if save_result:
        result_img = img.copy()
        
        # Рисуем занятые места (красные)
        for i, bbox in enumerate(occupied):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(result_img, f'Occupied {i+1}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Рисуем свободные места (зелёные)
        for i, bbox in enumerate(empty):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_img, f'Empty {i+1}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Добавляем статистику
        total_seats = len(occupied) + len(empty)
        occupancy_rate = len(occupied) / total_seats if total_seats > 0 else 0
        
        stats_text = f'Total: {total_seats}, Occupied: {len(occupied)}, Free: {len(empty)}, Rate: {occupancy_rate:.1%}'
        cv2.putText(result_img, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Сохраняем результат
        result_path = f'results_occupancy/{os.path.basename(image_path)}'
        os.makedirs('results_occupancy', exist_ok=True)
        cv2.imwrite(result_path, result_img)
    
    return {
        'total_seats': len(occupied) + len(empty),
        'occupied_seats': len(occupied),
        'empty_seats': len(empty),
        'occupancy_rate': len(occupied) / (len(occupied) + len(empty)) if (len(occupied) + len(empty)) > 0 else 0
    }

def analyze_batch_occupancy(images_dir):
    """Анализ всех изображений в папке"""
    results = []
    
    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(images_dir, img_name)
        print(f"Анализирую: {img_name}")
        
        result = analyze_occupancy(img_path)
        if result:
            result['image_name'] = img_name
            results.append(result)
    
    return results

# Пример использования
if __name__ == "__main__":
    # Анализ изображений из нового датасета
    images_dir = "test/images"  # используем новый датасет
    results = analyze_batch_occupancy(images_dir)
    
    print("\nРезультаты анализа заполненности зала:")
    print("-" * 60)
    for result in results:
        print(f"Изображение: {result['image_name']}")
        print(f"  Всего мест: {result['total_seats']}")
        print(f"  Занятых мест: {result['occupied_seats']}")
        print(f"  Свободных мест: {result['empty_seats']}")
        print(f"  Заполненность: {result['occupancy_rate']:.2%}")
        print("-" * 40) 