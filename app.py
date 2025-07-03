from flask import Flask, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import threading
import base64
from paddleocr import PaddleOCR
from inference import InferencePipeline
import requests

app = Flask(__name__)
CORS(app)

# === Global State ===
display_frame = None
result_text = "-"
lock = threading.Lock()

# === Init PaddleOCR ===
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

# === PaddleOCR Reader ===
def read_plate_text(plate_img):
    try:
        result = ocr_engine.ocr(plate_img, cls=True)
        for line in result:
            for box, (text, conf) in line:
                cleaned = ''.join(filter(str.isalnum, text.upper()))
                if 5 <= len(cleaned) <= 10:
                    print(f"[INFO] OCR Detected: {cleaned}")
                    return cleaned
    except Exception as e:
        print(f"[ERROR] PaddleOCR failed: {e}")
    return None

# === Frame to Base64 ===
def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    encoded = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded}"

# === Inference Pipeline Sink ===
def my_sink(result, video_frame):
    global display_frame, result_text
    try:
        frame = video_frame.numpy()[:, :, ::-1].copy()  # RGB to BGR
        predictions = result.get("predictions", [])
        texts = []

        for pred in predictions:
            if "x" in pred and "y" in pred and "width" in pred and "height" in pred:
                cx, cy, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                x_min = int(cx - w / 2)
                x_max = int(cx + w / 2)
                y_min = int(cy - h / 2)
                y_max = int(cy + h / 2)

                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(frame.shape[1] - 1, x_max)
                y_max = min(frame.shape[0] - 1, y_max)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                plate_crop = frame[y_min:y_max, x_min:x_max]
                if plate_crop.size == 0:
                    continue

                text = read_plate_text(plate_crop)
                if text:
                    texts.append(text)
                    cv2.putText(frame, text, (x_min, y_max + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        with lock:
            display_frame = frame.copy()
            result_text = ', '.join(texts) if texts else "-"
    except Exception as e:
        print(f"[ERROR] my_sink: {e}")

# === Initialize Inference Pipeline ===
def init_pipeline():
    print("[INFO] Initializing Inference Pipeline...")
    pipeline = InferencePipeline.init_with_workflow(
        api_key="1kkhDoupMwdi62nboV3L",
        workspace_name="tama-av3ne",
        workflow_id="detect-count-and-visualize-2",
        video_reference=0,  # Webcam, RTSP URL, or video file
        max_fps=30,
        on_prediction=my_sink
    )
    pipeline.start()
    print("[INFO] Inference Pipeline started.")

# === Routes ===
@app.route("/")
def home():
    return jsonify({
        "status": "✅ Flask SDK Pipeline with PaddleOCR running",
        "endpoints": ["/get_processed_frame", "/result", "/check_plate/<plat_nomor>"]
    })

@app.route("/get_processed_frame")
def get_frame():
    with lock:
        if display_frame is None:
            return jsonify({"error": "No frame yet"}), 400
        encoded = frame_to_base64(display_frame)
        return jsonify({"frame": encoded})

@app.route("/result")
def result():
    with lock:
        return jsonify({"text": result_text})

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

# === Gunicorn Hook ===
def post_fork(server, worker):
    threading.Thread(target=init_pipeline, daemon=True).start()
