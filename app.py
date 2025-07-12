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
model = YOLO('yolov8n.pt')

# Класс стула в COCO
CHAIR_CLASS_ID = 56

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
    
    # Детекция стульев
    results = model(img)[0]
    chairs = []
    
    for box in results.boxes:
        if int(box.cls) == CHAIR_CLASS_ID:
            chairs.append(box.xyxy[0].cpu().numpy())
    
    # Визуализация результатов
    result_img = img.copy()
    
    for i, chair in enumerate(chairs):
        x1, y1, x2, y2 = map(int, chair)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result_img, f'Chair {i+1}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Добавляем статистику
    stats_text = f'Detected chairs: {len(chairs)}'
    cv2.putText(result_img, stats_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return result_img, len(chairs)

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
    result_img, chairs_count = analyze_hall_occupancy(upload_path)
    
    if result_img is None:
        return jsonify({'error': 'Failed to process image'}), 500
    
    # Сохраняем результат
    result_filename = f"result_{filename}"
    result_path = os.path.join('static/results', result_filename)
    cv2.imwrite(result_path, result_img)
    
    # Сохраняем в базу данных
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO requests (timestamp, filename, chairs_detected, result_path)
        VALUES (?, ?, ?, ?)
    ''', (datetime.now().isoformat(), filename, chairs_count, result_path))
    conn.commit()
    conn.close()
    
    return jsonify({
        'success': True,
        'chairs_detected': chairs_count,
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
            'result_path': req[4]
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
        'requests': []
    }
    
    for req in requests:
        report_data['requests'].append({
            'timestamp': req[1],
            'filename': req[2],
            'chairs_detected': req[3]
        })
    
    return jsonify(report_data)

if __name__ == '__main__':
    init_database()
    app.run(debug=True, host='0.0.0.0', port=5000) 