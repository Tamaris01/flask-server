from ultralytics import YOLO
import cv2
import re
from paddleocr import PaddleOCR
import numpy as np

# Inisialisasi PaddleOCR (hanya sekali untuk efisiensi)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Cache model YOLO agar tidak load berulang
model_cache = None

def load_model(model_path):
    global model_cache
    if model_cache is None:
        print(f"[INFO] Loading YOLO model from {model_path}")
        model_cache = YOLO(model_path)
    return model_cache

def preprocess_plate(plate_img):
    # Preprocessing untuk OCR
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return resized

def format_license_plate(text):
    # Format hasil OCR agar sesuai pola plat
    match = re.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})?$', text)
    if match:
        part1 = match.group(1)
        part2 = match.group(2)
        part3 = match.group(3) if match.group(3) else ""
        return f"{part1} {part2} {part3}".strip()
    return text

def extract_text_paddle(preprocessed_img):
    try:
        result = ocr.ocr(preprocessed_img, cls=True)
        for line in result:
            for box, (text, conf) in line:
                cleaned = ''.join(filter(str.isalnum, text.upper()))
                match = re.search(r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$', cleaned)
                if match:
                    return format_license_plate(match.group(0))
    except Exception as e:
        print(f"[ERROR] PaddleOCR failed: {e}")
    return None

def detect_plate_image(frame, model_path):
    """
    frame: np.ndarray (BGR)
    model_path: str (path to YOLO model)
    returns:
        - frame with overlay
        - recognized license plate text (string)
    """
    model = load_model(model_path)

    # Resize untuk deteksi agar cepat
    resized_frame = cv2.resize(frame, (640, 480))

    # Predict YOLO
    results = model.predict(source=resized_frame, conf=0.3, verbose=False)
    print("[DEBUG] YOLO prediction done.")

    ocr_texts = []

    # Handle ultralytics Results format
    r = results[0] if isinstance(results, list) else results

    if hasattr(r, 'boxes') and r.boxes is not None:
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())

            x1, y1, x2, y2 = map(int, xyxy)

            # Filter bounding box kecil agar noise tidak terbaca
            if (x2 - x1) < 30 or (y2 - y1) < 15:
                continue

            # Scaling kembali ke ukuran frame asli
            x_scale = frame.shape[1] / 640
            y_scale = frame.shape[0] / 480

            x1 = int(x1 * x_scale)
            x2 = int(x2 * x_scale)
            y1 = int(y1 * y_scale)
            y2 = int(y2 * y_scale)

            # Clamp agar tidak keluar frame
            x1 = max(0, min(frame.shape[1] - 1, x1))
            x2 = max(0, min(frame.shape[1] - 1, x2))
            y1 = max(0, min(frame.shape[0] - 1, y1))
            y2 = max(0, min(frame.shape[0] - 1, y2))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Plate ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            plate_img = frame[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue

            preprocessed_img = preprocess_plate(plate_img)
            text = extract_text_paddle(preprocessed_img)

            if text:
                ocr_texts.append(text)
                cv2.putText(frame, f"OCR: {text}", (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    final_ocr = ', '.join(ocr_texts) if ocr_texts else "-"
    print(f"[INFO] OCR Result: {final_ocr}")
    return frame, final_ocr

# Untuk testing lokal:
if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "best.pt"
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame, ocr_result = detect_plate_image(frame, model_path)
        cv2.imshow("YOLO Plate Detection", output_frame)
        print("Detected:", ocr_result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
