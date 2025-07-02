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

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model YOLOv8 tidak ditemukan di {MODEL_PATH}")
else:
    print(f"✅ Model YOLOv8 ditemukan di {MODEL_PATH}. Siap digunakan.")

# STATE
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

# THREAD LOOP YOLOv8 DETEKSI
def detect_loop():
    global raw_frame, display_frame, result_text
    last_result = "-"
    first_detection_logged = False

    while True:
        frame_copy = None
        with lock:
            if raw_frame is not None:
                frame_copy = raw_frame.copy()

        if frame_copy is not None:
            try:
                det_frame, ocr_text = detect_plate_image(frame_copy, MODEL_PATH)
                with lock:
                    display_frame = det_frame
                    if ocr_text != "-" and ocr_text != last_result:
                        result_text = ocr_text
                        last_result = ocr_text
                        print(f"[INFO] Plat terdeteksi: {ocr_text}")

                # Info pertama kali YOLOv8 berjalan
                if not first_detection_logged:
                    print("✅ YOLOv8 model berhasil dijalankan dan deteksi aktif di VPS.")
                    first_detection_logged = True

            except Exception as e:
                print(f"[ERROR] Gagal deteksi: {e}")

        time.sleep(0.1)

# ENCODING FRAME KE BASE64
def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    encoded = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded}"

# ENDPOINT UPLOAD FRAME
@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    global raw_frame
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        image_data = data['image'].split(",")[1]
        img_array = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        with lock:
            raw_frame = frame

        print("[INFO] Frame diterima server.")
        return jsonify({"message": "Frame received successfully"}), 200

    except Exception as e:
        print(f"[ERROR] Upload frame: {e}")
        return jsonify({"error": str(e)}), 500

# ENDPOINT GET FRAME YANG SUDAH DIPROSES
@app.route('/get_processed_frame', methods=['GET'])
def get_processed_frame():
    with lock:
        if display_frame is None:
            return jsonify({'error': 'No frame to send'}), 400

        processed_frame_base64 = frame_to_base64(display_frame)
        return jsonify({'frame': processed_frame_base64})

# ENDPOINT RESULT PLAT NOMOR TERBARU
@app.route("/result", methods=["GET"])
def result():
    with lock:
        return jsonify({"plat_nomor": result_text}), 200

# ENDPOINT CEK PLAT NOMOR KE SERVER LARAVEL
@app.route("/check_plate/<plat_nomor>", methods=["GET"])
def check_plate(plat_nomor):
    try:
        response = requests.get(f"https://alpu.web.id/api/check_plate/{plat_nomor}", timeout=5)
        if response.status_code == 200:
            return response.json(), 200
        else:
            return {"error": "Laravel server unavailable", "exists": False}, 200
    except Exception as e:
        return {"error": str(e), "exists": False}, 200

# START SERVER
if __name__ == "__main__":
    print("🚀 Memulai Flask YOLOv8 Server di VPS...")
    threading.Thread(target=detect_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
