from flask import Flask, render_template, request, url_for, jsonify, send_file
from ultralytics import YOLO
from PIL import Image
import os
import cv2
import numpy as np
import time
import json
import logging
from datetime import datetime
import threading
from pathlib import Path

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detection.log'),
        logging.StreamHandler()
    ]
)

# Load multiple YOLO models for different use cases
models = {
    'yolov8n': None,
    'yolov8s': None,
    'best': None
}

def load_models():
    """Load all available models"""
    try:
        # Load YOLOv8 nano model (fast detection)
        if os.path.exists('model/yolov8n.pt'):
            models['yolov8n'] = YOLO('model/yolov8n.pt')
            logging.info("Loaded YOLOv8 Nano model")
        
        # Load YOLOv8 small model (balanced)
        if os.path.exists('model/yolov8s.pt'):
            models['yolov8s'] = YOLO('model/yolov8s.pt')
            logging.info("Loaded YOLOv8 Small model")
        
        # Load custom trained model (most accurate for fire/smoke)
        if os.path.exists('model/best.pt'):
            try:
                models['best'] = YOLO('model/best.pt')
                logging.info("Loaded custom trained model")
            except Exception as e:
                logging.warning(f"Failed to load custom model: {e}")
        
        # Fallback to nano if no models loaded
        if not any(models.values()):
            models['yolov8n'] = YOLO('yolov8n.pt')  # Download default
            logging.info("Downloaded and loaded default YOLOv8 Nano model")
            
    except Exception as e:
        logging.error(f"Error loading models: {e}")

# Load models on startup
load_models()

# Default model selection
current_model = models['yolov8n'] or models['yolov8s'] or models['best']
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
HISTORY_FILE = 'detection_history.json'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Detection history storage
detection_history = []

def load_detection_history():
    """Load detection history from file"""
    global detection_history
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                detection_history = json.load(f)
    except Exception as e:
        logging.error(f"Error loading history: {e}")
        detection_history = []

def save_detection_history():
    """Save detection history to file"""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(detection_history[-100:], f, indent=2)  # Keep last 100 records
    except Exception as e:
        logging.error(f"Error saving history: {e}")

def get_model(model_type='yolov8n'):
    """Get the specified model"""
    model = models.get(model_type)
    if model is None:
        # Fallback to any available model
        model = next((m for m in models.values() if m is not None), None)
        if model is None:
            raise ValueError("No models available")
    return model

def analyze_detections(results, confidence_threshold=0.5):
    """Analyze detection results and provide insights"""
    detections = []
    max_confidence = 0
    fire_count = 0
    smoke_count = 0
    
    if results and len(results) > 0:
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    confidence = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) > 0 else 0
                    if confidence >= confidence_threshold:
                        # Get class name
                        class_id = int(box.cls[0]) if hasattr(box, 'cls') and len(box.cls) > 0 else 0
                        class_name = result.names[class_id] if hasattr(result, 'names') and class_id in result.names else 'unknown'
                        
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': box.xyxy[0].tolist() if hasattr(box, 'xyxy') else []
                        })
                        
                        max_confidence = max(max_confidence, confidence)
                        if 'fire' in class_name.lower():
                            fire_count += 1
                        elif 'smoke' in class_name.lower():
                            smoke_count += 1
    
    # Determine risk level
    risk_level = 'Low'
    if fire_count > 0:
        risk_level = 'Critical' if fire_count > 2 else 'High'
    elif smoke_count > 0:
        risk_level = 'Medium' if smoke_count > 1 else 'Low'
    elif max_confidence > 0.8:
        risk_level = 'Medium'
    
    return {
        'detections': detections,
        'detection_count': len(detections),
        'max_confidence': round(max_confidence, 3) if max_confidence > 0 else 0,
        'fire_count': fire_count,
        'smoke_count': smoke_count,
        'risk_level': risk_level
    }

# Load history on startup
load_detection_history()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    start_time = time.time()
    
    try:
        # Debug logging
        logging.info(f"Received POST request to /detect")
        logging.info(f"Files in request: {list(request.files.keys())}")
        logging.info(f"Form data: {dict(request.form)}")
        
        if 'file' not in request.files:
            logging.warning("No file in request")
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            logging.warning("Empty filename")
            return jsonify({'success': False, 'message': 'No selected file'}), 400

        # Get request parameters with better error handling
        try:
            model_type = request.form.get('model_type', 'yolov8n')
            confidence = float(request.form.get('confidence', 0.5))
            save_results = request.form.get('save_results', 'false').lower() == 'true'
        except ValueError as e:
            logging.error(f"Parameter parsing error: {e}")
            return jsonify({'success': False, 'message': 'Invalid parameters'}), 400

        # Get the specified model
        model = get_model(model_type)
        if model is None:
            logging.error(f"Model {model_type} not available")
            return jsonify({'success': False, 'message': f'Model {model_type} not available'}), 400
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)

        # Process based on file type
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Process Image
            processed_file_path, analysis = process_image_advanced(file_path, model, confidence)
            file_type = 'image'
        elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            # Process Video
            processed_file_path, analysis = process_video_advanced(file_path, model, confidence)
            file_type = 'video'
        else:
            return jsonify({'success': False, 'message': 'Unsupported file format'}), 400

        processing_time = round(time.time() - start_time, 2)
        
        # Create detection record
        detection_record = {
            'timestamp': datetime.now().isoformat(),
            'filename': file.filename,
            'model_used': model_type,
            'processing_time': processing_time,
            'file_type': file_type,
            'confidence_threshold': confidence,
            **analysis
        }
        
        # Save to history
        detection_history.append(detection_record)
        if save_results:
            save_detection_history()
        
        # Log detection
        logging.info(f"Detection completed: {analysis['detection_count']} detections, {processing_time}s")
        
        # Prepare response
        response_data = {
            'success': True,
            'uploaded_file': url_for('static', filename='uploads/' + unique_filename),
            'processed_file': url_for('static', filename='processed/' + os.path.basename(processed_file_path)),
            'file_type': file_type,
            'processing_time': processing_time,
            **analysis
        }
        
        # Add alert message
        if analysis['detection_count'] > 0:
            if analysis['fire_count'] > 0:
                response_data['alert_message'] = f"🔥 FIRE DETECTED! {analysis['fire_count']} fire detection(s) found!"
                response_data['alert_type'] = 'error'
            elif analysis['smoke_count'] > 0:
                response_data['alert_message'] = f"💨 SMOKE DETECTED! {analysis['smoke_count']} smoke detection(s) found!"
                response_data['alert_type'] = 'warning'
            else:
                response_data['alert_message'] = f"⚠️ {analysis['detection_count']} detection(s) found!"
                response_data['alert_type'] = 'warning'
        else:
            response_data['alert_message'] = "✅ No fire or smoke detected."
            response_data['alert_type'] = 'success'
        
        # Return JSON response for AJAX or render template for form submission
        if request.is_json or 'application/json' in request.headers.get('Accept', ''):
            return jsonify(response_data)
        else:
            return render_template('index.html', **response_data)
        
    except Exception as e:
        logging.error(f"Detection error: {str(e)}")
        logging.error(f"Exception type: {type(e).__name__}")
        logging.error(f"Traceback: ", exc_info=True)
        
        error_response = {
            'success': False, 
            'message': f'Detection failed: {str(e)}',
            'error_type': type(e).__name__
        }
        
        # Always return JSON for fetch requests
        return jsonify(error_response), 500

def process_image_advanced(file_path, model, confidence_threshold=0.5):
    """Advanced image processing with detailed analysis"""
    try:
        # Run detection
        results = model(file_path, conf=confidence_threshold)
        
        # Analyze results
        analysis = analyze_detections(results, confidence_threshold)
        
        # Create annotated image
        processed_image_path = os.path.join(PROCESSED_FOLDER, 'processed_' + os.path.basename(file_path))
        
        if results and len(results) > 0:
            # Get annotated image
            annotated_image = results[0].plot(
                conf=True,  # Show confidence scores
                labels=True,  # Show labels
                boxes=True,  # Show bounding boxes
                line_width=3  # Thicker lines for better visibility
            )
            cv2.imwrite(processed_image_path, annotated_image)
        else:
            # No detections, copy original
            import shutil
            shutil.copy2(file_path, processed_image_path)
        
        return processed_image_path, analysis
        
    except Exception as e:
        logging.error(f"Image processing error: {e}")
        raise

def process_video_advanced(file_path, model, confidence_threshold=0.5):
    """Advanced video processing with frame-by-frame analysis"""
    try:
        cap = cv2.VideoCapture(file_path)
        output_file = os.path.join(PROCESSED_FOLDER, 'processed_' + os.path.basename(file_path))
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # Analysis tracking
        all_detections = []
        frame_count = 0
        
        logging.info(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection on frame
            results = model(frame, conf=confidence_threshold, verbose=False)
            
            # Analyze frame detections
            frame_analysis = analyze_detections(results, confidence_threshold)
            if frame_analysis['detection_count'] > 0:
                frame_analysis['frame_number'] = frame_count
                all_detections.append(frame_analysis)
            
            # Get annotated frame
            if results and len(results) > 0:
                annotated_frame = results[0].plot(
                    conf=True,
                    labels=True,
                    boxes=True,
                    line_width=2
                )
                out.write(annotated_frame)
            else:
                out.write(frame)
            
            # Progress logging
            if frame_count % max(1, total_frames // 10) == 0:
                progress = (frame_count / total_frames) * 100
                logging.info(f"Video processing progress: {progress:.1f}%")
        
        cap.release()
        out.release()
        
        # Compile overall video analysis
        total_detections = sum(d['detection_count'] for d in all_detections)
        max_confidence = max((d['max_confidence'] for d in all_detections), default=0)
        total_fire = sum(d['fire_count'] for d in all_detections)
        total_smoke = sum(d['smoke_count'] for d in all_detections)
        
        # Determine overall risk level
        risk_level = 'Low'
        if total_fire > 5:
            risk_level = 'Critical'
        elif total_fire > 0:
            risk_level = 'High'
        elif total_smoke > 10:
            risk_level = 'High'
        elif total_smoke > 0:
            risk_level = 'Medium'
        elif max_confidence > 0.8:
            risk_level = 'Medium'
        
        analysis = {
            'detection_count': total_detections,
            'max_confidence': round(max_confidence, 3),
            'fire_count': total_fire,
            'smoke_count': total_smoke,
            'risk_level': risk_level,
            'frames_with_detections': len(all_detections),
            'total_frames': total_frames,
            'detection_density': round(len(all_detections) / total_frames * 100, 2) if total_frames > 0 else 0
        }
        
        return output_file, analysis
        
    except Exception as e:
        logging.error(f"Video processing error: {e}")
        raise

# API Endpoints for advanced features
@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available models"""
    available_models = []
    for name, model in models.items():
        if model is not None:
            available_models.append({
                'name': name,
                'description': get_model_description(name),
                'loaded': True
            })
    return jsonify({'models': available_models})

@app.route('/api/history', methods=['GET'])
def get_detection_history():
    """Get detection history"""
    limit = request.args.get('limit', 50, type=int)
    return jsonify({'history': detection_history[-limit:]})

@app.route('/api/stats', methods=['GET'])
def get_detection_stats():
    """Get detection statistics"""
    if not detection_history:
        return jsonify({'stats': {}})
    
    total_detections = len(detection_history)
    fire_detections = sum(1 for d in detection_history if d.get('fire_count', 0) > 0)
    smoke_detections = sum(1 for d in detection_history if d.get('smoke_count', 0) > 0)
    avg_processing_time = sum(d.get('processing_time', 0) for d in detection_history) / total_detections
    
    stats = {
        'total_detections': total_detections,
        'fire_detections': fire_detections,
        'smoke_detections': smoke_detections,
        'avg_processing_time': round(avg_processing_time, 2),
        'risk_levels': {
            'critical': sum(1 for d in detection_history if d.get('risk_level') == 'Critical'),
            'high': sum(1 for d in detection_history if d.get('risk_level') == 'High'),
            'medium': sum(1 for d in detection_history if d.get('risk_level') == 'Medium'),
            'low': sum(1 for d in detection_history if d.get('risk_level') == 'Low')
        }
    }
    
    return jsonify({'stats': stats})

def get_model_description(model_name):
    """Get description for model"""
    descriptions = {
        'yolov8n': 'YOLOv8 Nano - Fast detection with good accuracy',
        'yolov8s': 'YOLOv8 Small - Balanced speed and accuracy',
        'best': 'Custom Trained - Highest accuracy for fire/smoke detection'
    }
    return descriptions.get(model_name, 'Unknown model')

# Legacy functions for compatibility
def process_image(file_path):
    """Legacy image processing function"""
    model = get_model('yolov8n')
    processed_path, _ = process_image_advanced(file_path, model, 0.5)
    return processed_path

def process_video(file_path):
    """Legacy video processing function"""
    model = get_model('yolov8n')
    processed_path, _ = process_video_advanced(file_path, model, 0.5)
    return processed_path

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080,debug=False)  # Updated for production