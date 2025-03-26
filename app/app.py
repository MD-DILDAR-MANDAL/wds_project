import os
import cv2
import torch
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLOv8 model
MODEL_PATH = "model/best.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH).to(device)

# Webcam resolution
WIDTH, HEIGHT = 640, 480  # Change this if needed

# YOLO Input Size
MODEL_SIZE = 416  

# Define colors for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(model.names), 3), dtype="uint8")

def process_frame(frame):
    """Runs YOLOv8 on a frame and returns a frame with bounding boxes."""
    h, w, _ = frame.shape
    frame_resized = cv2.resize(frame, (MODEL_SIZE, MODEL_SIZE))
    results = model(frame_resized)[0]  
    
    if results.boxes:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            class_id = int(box.cls[0])
            class_name = model.names.get(class_id, "Unknown")
            color = tuple(map(int, colors[class_id]))
            
            x1 = int(x1 * (w / MODEL_SIZE))
            y1 = int(y1 * (h / MODEL_SIZE))
            x2 = int(x2 * (w / MODEL_SIZE))
            y2 = int(y2 * (h / MODEL_SIZE))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return redirect(url_for('process_image', filename=file.filename))
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        return redirect(url_for('process_video', filename=file.filename))
    
    return "Unsupported file format", 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/process_image/<filename>')
def process_image(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    frame = cv2.imread(filepath)
    processed_frame = process_frame(frame)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], "processed_" + filename)
    cv2.imwrite(output_path, processed_frame)
    return redirect(url_for('uploaded_file', filename="processed_" + filename))

@app.route('/process_video/<filename>')
def process_video(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cap = cv2.VideoCapture(filepath)
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], "processed_" + filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        processed_frame = process_frame(frame)
        out.write(processed_frame)
    
    cap.release()
    out.release()
    return redirect(url_for('uploaded_file', filename="processed_" + filename))

if __name__ == '__main__':
    app.run(debug=True)
