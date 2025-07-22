from ultralytics import YOLO
import cv2
import re
from paddleocr import PaddleOCR
import numpy as np
import logging

ocr = PaddleOCR(use_angle_cls=True, lang='en')
model_cache = None

def load_model(model_path):
    global model_cache
    if model_cache is None:
        logging.info(f"Loading YOLO model: {model_path}")
        model_cache = YOLO(model_path)
    return model_cache

def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def format_license_plate(text):
    match = re.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})?$', text)
    if match:
        return f"{match.group(1)} {match.group(2)} {match.group(3) or ''}".strip()
    return text

def extract_text_paddle(preprocessed_img):
    result = ocr.ocr(preprocessed_img, cls=True)
    for line in result:
        for box, (text, conf) in line:
            cleaned = ''.join(filter(str.isalnum, text.upper()))
            logging.info(f"OCR Raw: {text} (cleaned: {cleaned}, conf: {conf})")
            if conf > 0.5 and re.match(r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$', cleaned):
                return format_license_plate(cleaned)
    return None

def detect_plate_image(frame, model_path, last_track_ids):
    model = load_model(model_path)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    boxes = results[0].boxes

    ocr_texts, new_track_ids = [], set()
    for box in boxes:
        track_id = int(box.id.item()) if box.id is not None else None
        if track_id is None or track_id in last_track_ids:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if conf < 0.3:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        plate_img = frame[y1:y2, x1:x2]
        text = extract_text_paddle(preprocess_plate(plate_img))
        if text:
            ocr_texts.append(text)
            cv2.putText(frame, f"{text}", (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            new_track_ids.add(track_id)

    return frame, ', '.join(ocr_texts) if ocr_texts else "-", new_track_ids
