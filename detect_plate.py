from ultralytics import YOLO
import cv2
import re
from paddleocr import PaddleOCR
import time

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
    small_frame = cv2.resize(frame, (640, 480))

    results = model.predict(source=small_frame, conf=0.3, verbose=False)

    ocr_texts = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if (x2 - x1) < 30 or (y2 - y1) < 15:
                continue

            x1, y1, x2, y2 = map(lambda v: int(v * frame.shape[1] / 640), [x1, y1, x2, y2])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Plat ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            plate_img = frame[y1:y2, x1:x2]
            preprocessed_img = preprocess_plate(plate_img)
            text = extract_text_paddle(preprocessed_img)

            if text:
                ocr_texts.append(text)
                cv2.putText(frame, f"OCR: {text}", (x1, y2 + 25),
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
            break

        output_frame, ocr_result = detect_plate_image(frame, model_path)

        fps = 1 / (time.time() - start_time)
        cv2.putText(output_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Deteksi Plat Nomor", output_frame)
        print("Hasil OCR:", ocr_result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
