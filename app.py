from flask import Flask, jsonify, request
from flask_cors import CORS
import base64
import cv2
import numpy as np
import threading
import time
import logging
from detect_plate import detect_plate_image  # pastikan sudah pakai YOLO+PaddleOCR

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

raw_frame = None
display_frame = None
result_text = "-"
lock = threading.Lock()
last_track_ids = set()

MODEL_PATH = "best.pt"

def detect_loop():
    global raw_frame, display_frame, result_text, last_track_ids
    logging.info("🚀 Detection loop started...")
    while True:
        frame_copy = None
        with lock:
            if raw_frame is not None:
                frame_copy = raw_frame.copy()

        if frame_copy is not None:
            try:
                det_frame, ocr_text, new_track_ids = detect_plate_image(
                    frame_copy, MODEL_PATH, last_track_ids
                )
                with lock:
                    display_frame = det_frame
                    if ocr_text and ocr_text != "-":
                        result_text = ocr_text
                        logging.info(f"✅ Detected plate: {ocr_text}")
                    last_track_ids.update(new_track_ids)
            except Exception as e:
                logging.error(f"Detection failed: {e}")
        else:
            logging.debug("⏳ Waiting for frame...")
        time.sleep(0.1)

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

        image_base64 = data['image']
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]

        img_array = np.frombuffer(base64.b64decode(image_base64), np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        with lock:
            raw_frame = frame

        return jsonify({'message': 'Frame received successfully'}), 200

    except Exception as e:
        logging.error(f"upload_frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_processed_frame', methods=['GET'])
def get_processed_frame():
    with lock:
        if display_frame is None:
            return jsonify({'error': 'No frame to send'}), 400
        return jsonify({'frame': frame_to_base64(display_frame)}), 200

@app.route('/result', methods=['GET'])
def result():
    with lock:
        return jsonify({'plat_nomor': result_text}), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    threading.Thread(target=detect_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
