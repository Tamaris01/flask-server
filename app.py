from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import base64
import cv2
import numpy as np
import os
import threading
from queue import Queue, Empty
from detect_plate import detect_plate_image

app = FastAPI(root_path="/server")

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

# Queue for frames (maxsize=1 to avoid backlog)
frame_queue = Queue(maxsize=1)

# Shared states
display_frame = None
result_text = "-"
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
    global display_frame, result_text
    last_result = "-"
    print("[INFO] Detection loop started.")

    while True:
        try:
            frame = frame_queue.get(timeout=1)  # block until frame available
            det_frame, ocr_text = detect_plate_image(frame, MODEL_PATH)

            with lock:
                display_frame = det_frame
                if ocr_text != "-" and ocr_text != last_result:
                    result_text = ocr_text
                    last_result = ocr_text
                    print(f"[INFO] Detected: {ocr_text}")

            frame_queue.task_done()

        except Empty:
            continue  # No frame in queue, loop again
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")

@app.post("/upload_frame")
async def upload_frame(data: FrameUpload):
    try:
        if not data.image:
            return JSONResponse(content={"error": "No image provided"}, status_code=400)

        img_data = data.image.split(',')[1]
        img_array = np.frombuffer(base64.b64decode(img_data), np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse(content={"error": "Failed to decode image"}, status_code=400)

        frame = cv2.resize(frame, (640, 480))

        # Put frame into queue if empty, else reject to avoid overload
        if frame_queue.full():
            return JSONResponse(content={"error": "Server is busy, try again"}, status_code=429)
        else:
            frame_queue.put(frame)
            print("[INFO] Frame received for processing.")
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
        response = requests.get(f"https://alpu.web.id/api/check_plate/{plat_nomor}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Failed to connect to Laravel", "exists": False}
    except Exception as e:
        return {"error": str(e), "exists": False}

# Start detection loop
threading.Thread(target=detect_loop, daemon=True).start()
