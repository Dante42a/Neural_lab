from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
import os
import sqlite3
from datetime import datetime
from ultralytics import YOLO
import json

app = Flask(__name__)

# Создаём папки для статических файлов
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Загружаем модель
print("Загружаем модель YOLOv8...")
model = YOLO('yolov8n.pt')
print("Модель загружена успешно!")

# Классы в COCO
CHAIR_CLASS_ID = 56
PERSON_CLASS_ID = 0

def init_database():
    """Инициализация базы данных для истории запросов"""
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            chairs_detected INTEGER,
            people_detected INTEGER,
            occupied_seats INTEGER,
            free_seats INTEGER,
            result_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def analyze_hall_occupancy(image_path):
    """Анализ заполненности зала"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Детекция объектов
    results = model(img)[0]
    chairs = []
    people = []
    
    for box in results.boxes:
        class_id = int(box.cls)
        if class_id == CHAIR_CLASS_ID:
            chairs.append(box.xyxy[0].cpu().numpy())
        elif class_id == PERSON_CLASS_ID:
            people.append(box.xyxy[0].cpu().numpy())
    
    # Определение занятых мест (простая логика: количество людей = занятые места)
    occupied_seats = len(people)
    free_seats = max(0, len(chairs) - occupied_seats)
    
    # Визуализация результатов
    result_img = img.copy()
    
    # Отрисовка стульев
    for i, chair in enumerate(chairs):
        x1, y1, x2, y2 = map(int, chair)
        # Зеленый для свободных мест, красный для занятых
        color = (0, 255, 0) if i < free_seats else (0, 0, 255)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        status = "Free" if i < free_seats else "Occupied"
        cv2.putText(result_img, f'{status} {i+1}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Отрисовка людей
    for i, person in enumerate(people):
        x1, y1, x2, y2 = map(int, person)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(result_img, f'Person {i+1}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Добавляем статистику
    stats_text = f'Chairs: {len(chairs)} | People: {len(people)} | Free: {free_seats} | Occupied: {occupied_seats}'
    cv2.putText(result_img, stats_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return result_img, len(chairs), len(people), occupied_seats, free_seats

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    """Обработка загруженного изображения"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Сохраняем загруженное изображение
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{file.filename}"
    upload_path = os.path.join('static/uploads', filename)
    file.save(upload_path)
    
    # Анализируем изображение
    result = analyze_hall_occupancy(upload_path)
    
    if result is None:
        return jsonify({'error': 'Failed to process image'}), 500
    
    result_img, chairs_count, people_count, occupied_seats, free_seats = result
    
    # Сохраняем результат
    result_filename = f"result_{filename}"
    result_path = os.path.join('static/results', result_filename)
    cv2.imwrite(result_path, result_img)
    
    # Сохраняем в базу данных
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO requests (timestamp, filename, chairs_detected, people_detected, occupied_seats, free_seats, result_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), filename, chairs_count, people_count, occupied_seats, free_seats, result_path))
    conn.commit()
    conn.close()
    
    return jsonify({
        'success': True,
        'chairs_detected': chairs_count,
        'people_detected': people_count,
        'occupied_seats': occupied_seats,
        'free_seats': free_seats,
        'result_image': f'/static/results/{result_filename}',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/history')
def history():
    """Страница с историей запросов"""
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM requests ORDER BY timestamp DESC LIMIT 10')
    requests = cursor.fetchall()
    conn.close()
    
    history_data = []
    for req in requests:
        history_data.append({
            'id': req[0],
            'timestamp': req[1],
            'filename': req[2],
            'chairs_detected': req[3],
            'people_detected': req[4],
            'occupied_seats': req[5],
            'free_seats': req[6],
            'result_path': req[7]
        })
    
    return render_template('history.html', requests=history_data)

@app.route('/download_report')
def download_report():
    """Генерация отчёта в формате JSON"""
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM requests ORDER BY timestamp DESC')
    requests = cursor.fetchall()
    conn.close()
    
    report_data = {
        'generated_at': datetime.now().isoformat(),
        'total_requests': len(requests),
        'total_chairs': sum(req[3] for req in requests),
        'total_people': sum(req[4] for req in requests),
        'total_occupied': sum(req[5] for req in requests),
        'total_free': sum(req[6] for req in requests),
        'requests': []
    }
    
    for req in requests:
        report_data['requests'].append({
            'timestamp': req[1],
            'filename': req[2],
            'chairs_detected': req[3],
            'people_detected': req[4],
            'occupied_seats': req[5],
            'free_seats': req[6]
        })
    
    return jsonify(report_data)

if __name__ == '__main__':
    init_database()
    app.run(debug=False, host='0.0.0.0', port=5001) 