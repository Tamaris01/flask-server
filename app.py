from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import base64
import numpy as np
import threading
import time
import os
import requests

from inference import InferencePipeline

app = Flask(__name__)
CORS(app)

# === GLOBAL STATE ===
display_frame = None
result_text = "-"
lock = threading.Lock()

# === Inference Pipeline Sink ===
def my_sink(result, video_frame):
    global display_frame, result_text
    try:
        with lock:
            if result.get("output_image"):
                # Convert SDK Image to OpenCV BGR
                display_frame = result["output_image"].numpy_image
            if "predictions" in result:
                result_text = str(result["predictions"])
        print("[INFO] Frame processed, prediction updated.")
    except Exception as e:
        print(f"[ERROR] my_sink: {e}")

# === Pipeline Initialization ===
def init_pipeline():
    global pipeline
    print("[INFO] Initializing Inference Pipeline...")
    pipeline = InferencePipeline.init_with_workflow(
        api_key="1kkhDoupMwdi62nboV3L",
        workspace_name="tama-av3ne",
        workflow_id="detect-count-and-visualize-2",
        video_reference=0,  # webcam, or "rtsp://..."
        max_fps=15,
        on_prediction=my_sink
    )
    pipeline.start()
    print("[INFO] Inference Pipeline started and running.")

# === Utility ===
def frame_to_base64(frame):
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        encoded = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        print(f"[ERROR] Failed to encode frame: {e}")
        return None

# === Routes ===
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "✅ Flask Inference Pipeline Server Running",
        "endpoints": ["/get_processed_frame [GET]", "/result [GET]"]
    })

@app.route("/get_processed_frame", methods=["GET"])
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
        return jsonify({"prediction": result_text}), 200
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
    print("[INFO] post_fork Gunicorn triggered, starting pipeline thread...")
    pipeline_thread = threading.Thread(target=init_pipeline, daemon=True)
    pipeline_thread.start()
    print("[INFO] Inference Pipeline should now be running.")

# === Local Dev Run ===
if __name__ == "__main__":
    init_pipeline()  # Local dev auto-run pipeline
    app.run(host="0.0.0.0", port=5000, debug=False)
