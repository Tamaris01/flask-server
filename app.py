from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import base64
import cv2
import numpy as np
import os
import threading
import time
from detect_plate import detect_plate_image

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model path check
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at: {MODEL_PATH}")

# Shared states
raw_frame = None
display_frame = None
result_text = "-"
frame_processed = True
lock = threading.Lock()

# Image input model
class FrameUpload(BaseModel):
    image: str

# Convert OpenCV frame to base64
def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    encoded = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded}"

# Detection loop thread
def detect_loop():
    global raw_frame, display_frame, result_text, frame_processed
    last_result = "-"
    print("[INFO] Detection loop started.")
    while True:
        with lock:
            frame_copy = raw_frame.copy() if raw_frame is not None else None
            current_processed = frame_processed

        if frame_copy is not None and not current_processed:
            try:
                det_frame, ocr_text = detect_plate_image(frame_copy, MODEL_PATH)
                with lock:
                    display_frame = det_frame
                    frame_processed = True
                    if ocr_text != "-" and ocr_text != last_result:
                        result_text = ocr_text
                        last_result = ocr_text
                        print(f"[INFO] Detected: {ocr_text}")
            except Exception as e:
                print(f"[ERROR] Detection failed: {e}")
        time.sleep(0.1)

@app.post("/upload_frame")
async def upload_frame(data: FrameUpload):
    global raw_frame, frame_processed
    try:
        if not data.image:
            return JSONResponse(content={"error": "No image provided"}, status_code=400)

        img_data = data.image.split(',')[1]
        img_array = np.frombuffer(base64.b64decode(img_data), np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse(content={"error": "Failed to decode image"}, status_code=400)

        # Optional resize to reduce lag
        frame = cv2.resize(frame, (640, 480))

        with lock:
            raw_frame = frame
            frame_processed = False

        print("[INFO] Frame received.")
        return {"message": "Frame received successfully"}

    except Exception as e:
        print(f"[ERROR] upload_frame: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/get_processed_frame")
async def get_processed_frame():
    with lock:
        if display_frame is None:
            return JSONResponse(content={"error": "No frame to send"}, status_code=400)
        return {"frame": frame_to_base64(display_frame)}

@app.get("/result")
async def get_result():
    with lock:
        return {"plat_nomor": result_text}

@app.get("/check_plate/{plat_nomor}")
async def check_plate(plat_nomor: str):
    try:
        response = requests.get(f"https://alpu.web.id/api/check_plate/{plat_nomor}")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Failed to connect to Laravel", "exists": False}
    except Exception as e:
        return {"error": str(e), "exists": False}

# Start detection loop
threading.Thread(target=detect_loop, daemon=True).start()
