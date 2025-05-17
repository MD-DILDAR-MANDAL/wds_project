import os
import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory
import time

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = "model/saved_model"  # Path to your trained VGG16 model
model = tf.saved_model.load(MODEL_PATH)

# Webcam resolution
WIDTH, HEIGHT = 640, 480  # Change this if needed

MODEL_SIZE = 416  

# Class labels for weapon detection
class_labels = ['Knife', 'Gun', 'LongGun']

# Confidence Threshold
THRESHOLD = 0.9  # <<< You can change this value easily now

def preprocess_frame(frame):
    h, w, _ = frame.shape
    frame_resized = cv2.resize(frame, (MODEL_SIZE, MODEL_SIZE))
    
    # Normalize the frame
    frame_resized = frame_resized.astype(np.float32) / 255.0
    
    # Expand dimensions to match the batch size expected by the model
    frame_array = np.expand_dims(frame_resized, axis=0)
    
    return frame_array

def process_frame(frame):
    """Runs the model on a frame and returns a frame with bounding boxes."""
    # Keep original frame
    original_frame = frame.copy()
    h, w, _ = original_frame.shape
    
    # Preprocess the frame for the model
    frame_array = preprocess_frame(frame)
    
    # Make prediction using the SavedModel
    predictions = model(tf.convert_to_tensor(frame_array))
    
    # Handle the predictions
    if isinstance(predictions, dict):
        class_probs = predictions['output_1'].numpy()[0]
        boxes = predictions['output_2'].numpy()[0]
    else:
        class_probs = predictions[0].numpy()[0]
        boxes = predictions[1].numpy()[0]
    
    # Get the class with the highest probability
    max_class_idx = np.argmax(class_probs)
    max_score = class_probs[max_class_idx]
    
    # If the confidence score is above threshold, draw the bounding box
    if max_score > THRESHOLD:
        # Convert normalized bounding box coordinates to pixel values
        # for the original frame size
        xmin, ymin, xmax, ymax = boxes
        xmin = int(xmin * w)
        ymin = int(ymin * h)
        xmax = int(xmax * w)
        ymax = int(ymax * h)
        
        label = f"{class_labels[max_class_idx]} {max_score:.2f}"
        
        # Draw bounding box and label on the original frame
        cv2.rectangle(original_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(original_frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return original_frame

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
