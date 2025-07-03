from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import base64
import cv2
import os
import threading
import time
import numpy as np
from detect_plate import detect_plate_image

app = Flask(__name__)
CORS(app)

# === CONFIG ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model YOLOv8 tidak ditemukan di {MODEL_PATH}")
else:
    print(f"✅ Model YOLOv8 ditemukan di {MODEL_PATH}. Siap digunakan.")

# === GLOBAL STATE ===
raw_frame = None
display_frame = None
result_text = "-"
lock = threading.Lock()

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "✅ Flask YOLOv8 Server Running",
        "model_path": MODEL_PATH,
        "endpoints": ["/upload_frame [POST]", "/get_processed_frame [GET]", "/result [GET]", "/check_plate/<plat_nomor> [GET]"]
    })

# === YOLOv8 Detection Loop ===
def detect_loop():
    global raw_frame, display_frame, result_text
    last_result = "-"
    first_detection_logged = False
    print("[INFO] detect_loop berjalan...")

    while True:
        frame_copy = None
        with lock:
            if raw_frame is not None:
                frame_copy = raw_frame.copy()
                raw_frame = None  # Clear after taking

        if frame_copy is not None:
            try:
                print("[INFO] Memulai deteksi YOLOv8 pada frame baru...")
                det_frame, ocr_text = detect_plate_image(frame_copy, MODEL_PATH)

                with lock:
                    display_frame = det_frame
                    if ocr_text != "-" and ocr_text != last_result:
                        result_text = ocr_text
                        last_result = ocr_text
                        print(f"[INFO] Plat terdeteksi: {ocr_text}")

                if not first_detection_logged:
                    print("✅ YOLOv8 model berjalan dan deteksi aktif.")
                    first_detection_logged = True

            except Exception as e:
                print(f"[ERROR] YOLOv8 gagal memproses: {e}")
        else:
            time.sleep(0.5)  # Sleep lebih pendek agar responsif

# === Utility ===
def frame_to_base64(frame):
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        encoded = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        print(f"[ERROR] Gagal encode frame ke base64: {e}")
        return None

# === Endpoints ===
@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    global raw_frame
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        image_data = data['image'].split(",")[-1]
        img_array = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        with lock:
            raw_frame = frame

        print("[INFO] Frame diterima dan siap diproses YOLOv8.")
        return jsonify({"message": "Frame received successfully"}), 200

    except Exception as e:
        print(f"[ERROR] upload_frame: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_processed_frame', methods=['GET'])
def get_processed_frame():
    with lock:
        if display_frame is None:
            return jsonify({'error': 'No frame processed yet'}), 400

        encoded_frame = frame_to_base64(display_frame)
        if encoded_frame:
            return jsonify({'frame': encoded_frame}), 200
        else:
            return jsonify({'error': 'Failed to encode frame'}), 500

@app.route("/result", methods=["GET"])
def result():
    with lock:
        return jsonify({"plat_nomor": result_text}), 200

@app.route("/check_plate/<plat_nomor>", methods=["GET"])
def check_plate(plat_nomor):
    try:
        response = requests.get(f"https://alpu.web.id/api/check_plate/{plat_nomor}", timeout=5)
        if response.status_code == 200:
            return response.json(), 200
        else:
            return {"error": "Laravel server unavailable", "exists": False}, 200
    except Exception as e:
        print(f"[ERROR] check_plate: {e}")
        return {"error": str(e), "exists": False}, 200

# === Main Runner ===
if __name__ == "__main__":
    print("🚀 Memulai Flask YOLOv8 Server di VPS...")
    detection_thread = threading.Thread(target=detect_loop, daemon=True)
    detection_thread.start()
    app.run(host="0.0.0.0", port=5000, debug=True)
