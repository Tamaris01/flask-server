from ultralytics import YOLO
import cv2
import re
from paddleocr import PaddleOCR
import time

# Inisialisasi OCR sekali di awal
ocr = PaddleOCR(use_angle_cls=True, lang='en')
model_cache = None

def load_model(model_path):
    global model_cache
    if model_cache is None:
        # Auto pakai GPU jika tersedia
        model_cache = YOLO(model_path)
    return model_cache

def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return resized

def format_license_plate(text):
    match = re.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})?$', text)
    if match:
        part1 = match.group(1)
        part2 = match.group(2)
        part3 = match.group(3) if match.group(3) else ""
        return f"{part1} {part2} {part3}".strip()
    return text

def extract_text_paddle(preprocessed_img):
    result = ocr.ocr(preprocessed_img, cls=True)
    for line in result:
        for box, (text, conf) in line:
            cleaned = ''.join(filter(str.isalnum, text.upper()))
            match = re.search(r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$', cleaned)
            if match:
                return format_license_plate(match.group(0))
    return None

def detect_plate_image(frame, model_path):
    model = load_model(model_path)

    # Resize untuk percepatan proses
    resized_w, resized_h = 480, 360
    small_frame = cv2.resize(frame, (resized_w, resized_h))

    results = model.predict(source=small_frame, conf=0.5, verbose=False)

    ocr_texts = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            w = x2 - x1
            h = y2 - y1
            aspect_ratio = w / h

            # Filter bounding box agar tidak noisy
            if w < 60 or h < 20 or aspect_ratio < 1.5 or aspect_ratio > 6:
                continue

            # Scale kembali ke frame asli
            x1 = int(x1 * frame.shape[1] / resized_w)
            y1 = int(y1 * frame.shape[0] / resized_h)
            x2 = int(x2 * frame.shape[1] / resized_w)
            y2 = int(y2 * frame.shape[0] / resized_h)

            # Validasi boundary crop
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            plate_img = frame[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue

            preprocessed_img = preprocess_plate(plate_img)
            text = extract_text_paddle(preprocessed_img)

            # Visualisasi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Plat ({conf:.2f})"
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if text:
                ocr_texts.append(text)
                cv2.putText(frame, f"{text}", (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    final_ocr = ', '.join(ocr_texts) if ocr_texts else "-"
    return frame, final_ocr

def run_realtime(model_path):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Untuk Windows
    # cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Untuk Linux

    print("Tekan 'q' untuk keluar.")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Tidak dapat membaca kamera.")
            break

        output_frame, ocr_result = detect_plate_image(frame, model_path)

        fps = 1 / max(time.time() - start_time, 0.001)
        cv2.putText(output_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Deteksi Plat Nomor", output_frame)
        print("Hasil OCR:", ocr_result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
