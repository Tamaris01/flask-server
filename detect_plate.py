from ultralytics import YOLO
import cv2
import re
from paddleocr import PaddleOCR

# Inisialisasi PaddleOCR sekali saja
ocr = PaddleOCR(use_angle_cls=True, lang='en')
_model = None

def load_model(model_path):
    global _model
    if _model is None:
        _model = YOLO(model_path)
    return _model

# 🔍 Untuk menampilkan hasil deteksi pada frame video
def detect_only(frame, model_path):
    model = load_model(model_path)
    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = results.names[int(box.cls[0])]
        conf = float(box.conf[0])
        if conf < 0.3:
            continue

        # Buat bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

# 📸 Untuk membaca plat nomor dari hasil deteksi
def run_ocr_snapshot(frame, model_path):
    model = load_model(model_path)
    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if conf < 0.3:
            continue

        # Crop area yang diduga plat nomor
        plate_crop = frame[y1:y2, x1:x2]

        if plate_crop.size == 0:
            continue  # skip jika crop kosong

        # Preprocessing gambar untuk OCR
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        result = ocr.ocr(resized, cls=True)

        for line in result[0]:
            text = line[1][0].upper()
            cleaned = ''.join(filter(str.isalnum, text))

            # Gunakan regex plat nomor Indonesia
            match = re.match(r'^[A-Z]{1,2}\d{1,4}[A-Z]{0,3}$', cleaned)
            if match:
                return format_plate(match.group(0))

    return "-"

# 🎯 Format ulang agar jadi "BP 1234 XY"
def format_plate(text):
    match = re.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})$', text)
    if match:
        return f"{match.group(1)} {match.group(2)} {match.group(3)}".strip()
    return text
