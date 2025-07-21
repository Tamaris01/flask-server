from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import base64
import cv2
import os
import threading
import time
import numpy as np
import logging
from detect_plate import detect_plate_image

app = Flask(__name__)
CORS(app)

# Logging agar lebih rapi
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at: {MODEL_PATH}")

# Global states
raw_frame = None
display_frame = None
result_text = "-"
lock = threading.Lock()
last_track_ids = set()

# Thread loop for real-time detection
def detect_loop():
    global raw_frame, display_frame, result_text, last_track_ids
    logging.info("Detection loop started with YOLO tracking + PaddleOCR.")
    while True:
        with lock:
            frame_copy = raw_frame.copy() if raw_frame is not None else None

        if frame_copy is not None:
            try:
                det_frame, ocr_text, new_track_ids = detect_plate_image(
                    frame_copy, MODEL_PATH, last_track_ids
                )
                with lock:
                    display_frame = det_frame
                    if ocr_text != "-":
                        result_text = ocr_text
                        logging.info(f"Detected plate: {ocr_text}")
                    last_track_ids.update(new_track_ids)
            except Exception as e:
                logging.error(f"Detection failed: {e}")
        time.sleep(0.1)  # lebih stabil

def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_frame = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_frame}"

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global raw_frame
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # ✅ Perbaikan: Cek apakah ada koma sebelum split
        image_base64 = data['image']
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]

        img_array = np.frombuffer(base64.b64decode(image_base64), np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        with lock:
            raw_frame = frame

        return jsonify({'message': 'Frame received and processed successfully'}), 200

    except Exception as e:
        logging.error(f"upload_frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_processed_frame', methods=['GET'])
def get_processed_frame():
    with lock:
        if display_frame is None:
            return jsonify({'error': 'No frame to send'}), 400
        processed_frame_base64 = frame_to_base64(display_frame)
        return jsonify({'frame': processed_frame_base64}), 200

@app.route('/result', methods=['GET'])
def result():
    with lock:
        return jsonify({'plat_nomor': result_text}), 200

@app.route('/check_plate/<plat_nomor>', methods=['GET'])
def check_plate(plat_nomor):
    try:
        response = requests.get(f'https://alpu.web.id/api/check_plate/{plat_nomor}')
        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            return jsonify({"error": "Failed to connect to Laravel", "exists": False}), 500
    except requests.exceptions.RequestException as e:
        logging.error(f"check_plate: {e}")
        return jsonify({"error": str(e), "exists": False}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    logging.info("Starting Flask YOLO + PaddleOCR server...")
    threading.Thread(target=detect_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
