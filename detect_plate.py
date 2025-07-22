from ultralytics import YOLO
import cv2
import re
from paddleocr import PaddleOCR
import numpy as np

ocr = PaddleOCR(use_angle_cls=True, lang='en')
model_cache = None

def load_model(model_path):
    global model_cache
    if model_cache is None:
        model_cache = YOLO(model_path)
    return model_cache

def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return resized

def format_license_plate(text):
    match = re.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})?$', text)
    if match:
        return f"{match.group(1)} {match.group(2)} {match.group(3) if match.group(3) else ''}".strip()
    return text

def extract_text_paddle(preprocessed_img):
    result = ocr.ocr(preprocessed_img, cls=True)
    for line in result:
        for box, (text, conf) in line:
            cleaned = ''.join(filter(str.isalnum, text.upper()))
            if re.match(r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$', cleaned):
                return format_license_plate(cleaned)
    return None

def detect_plate_image(frame, model_path, last_track_ids):
    model = load_model(model_path)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    boxes = results[0].boxes

    ocr_texts = []
    new_track_ids = set()

    for box in boxes:
        track_id = int(box.id.item()) if box.id is not None else None
        if track_id is None or track_id in last_track_ids:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if conf < 0.3:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        plate_img = frame[y1:y2, x1:x2]
        preprocessed_img = preprocess_plate(plate_img)
        text = extract_text_paddle(preprocessed_img)
        if text:
            ocr_texts.append(text)
            cv2.putText(frame, text, (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            new_track_ids.add(track_id)

    return frame, (', '.join(ocr_texts) if ocr_texts else "-"), new_track_ids
