from ultralytics import YOLO
import cv2
import os
import numpy as np

# Модель для детекции
model = YOLO('yolov8n.pt')

# Классы из датасета
OCCUPIED_CLASS_ID = 0  # "Non-empty"
EMPTY_CLASS_ID = 1     # "empty"

def load_annotations(label_path):
    """Загружает аннотации из YOLO формата"""
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append({
                        'class': cls,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
    return annotations

def convert_yolo_to_pixel(yolo_coords, img_width, img_height):
    """Конвертирует YOLO координаты в пиксели"""
    x_center = yolo_coords['x_center'] * img_width
    y_center = yolo_coords['y_center'] * img_height
    width = yolo_coords['width'] * img_width
    height = yolo_coords['height'] * img_height
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    return [x1, y1, x2, y2]

def calculate_iou(box1, box2):
    """Вычисляет IoU между двумя bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def evaluate_image(image_path, label_path):
    """Оценивает качество детекции на одном изображении"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_height, img_width = img.shape[:2]
    
    # Загружаем ground truth
    gt_annotations = load_annotations(label_path)
    gt_occupied = [ann for ann in gt_annotations if ann['class'] == OCCUPIED_CLASS_ID]
    gt_empty = [ann for ann in gt_annotations if ann['class'] == EMPTY_CLASS_ID]
    
    # Получаем предсказания модели
    results = model(img)[0]
    pred_occupied = []
    pred_empty = []
    
    for box in results.boxes:
        cls = int(box.cls)
        bbox = box.xyxy[0].cpu().numpy()
        
        # Простая логика: если детектирован стул, считаем его занятым
        if cls == 56:  # chair class in COCO
            pred_occupied.append(bbox)
    
    # Конвертируем GT в пиксели
    gt_occupied_pixels = [convert_yolo_to_pixel(ann, img_width, img_height) for ann in gt_occupied]
    gt_empty_pixels = [convert_yolo_to_pixel(ann, img_width, img_height) for ann in gt_empty]
    
    # Вычисляем метрики
    tp_occupied = 0
    fp_occupied = len(pred_occupied)
    fn_occupied = len(gt_occupied_pixels)
    
    # Простая оценка: считаем совпадения по IoU
    for pred in pred_occupied:
        for gt in gt_occupied_pixels:
            if calculate_iou(pred, gt) > 0.5:  # IoU threshold
                tp_occupied += 1
                fp_occupied -= 1
                fn_occupied -= 1
                break
    
    precision = tp_occupied / (tp_occupied + fp_occupied) if (tp_occupied + fp_occupied) > 0 else 0
    recall = tp_occupied / (tp_occupied + fn_occupied) if (tp_occupied + fn_occupied) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'gt_occupied': len(gt_occupied),
        'gt_empty': len(gt_empty),
        'pred_occupied': len(pred_occupied),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_dataset(images_dir, labels_dir):
    """Оценивает качество на всём датасете"""
    results = []
    
    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(images_dir, img_name)
        label_name = img_name.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        
        print(f"Оцениваю: {img_name}")
        result = evaluate_image(img_path, label_path)
        if result:
            result['image_name'] = img_name
            results.append(result)
    
    return results

# Пример использования
if __name__ == "__main__":
    images_dir = "test/images"
    labels_dir = "test/labels"
    
    results = evaluate_dataset(images_dir, labels_dir)
    
    if results:
        avg_precision = np.mean([r['precision'] for r in results])
        avg_recall = np.mean([r['recall'] for r in results])
        avg_f1 = np.mean([r['f1'] for r in results])
        
        print("\nРезультаты оценки качества:")
        print("-" * 60)
        print(f"Средняя точность (Precision): {avg_precision:.3f}")
        print(f"Средняя полнота (Recall): {avg_recall:.3f}")
        print(f"Средний F1-score: {avg_f1:.3f}")
        print("-" * 60)
        
        for result in results:
            print(f"Изображение: {result['image_name']}")
            print(f"  GT занятых: {result['gt_occupied']}, GT свободных: {result['gt_empty']}")
            print(f"  Предсказано занятых: {result['pred_occupied']}")
            print(f"  Precision: {result['precision']:.3f}, Recall: {result['recall']:.3f}, F1: {result['f1']:.3f}")
            print("-" * 40) 