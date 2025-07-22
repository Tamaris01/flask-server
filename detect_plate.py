from paddleocr import PaddleOCR
from ultralytics import YOLO
import cv2

ocr = PaddleOCR(use_angle_cls=True, lang='en')
model = YOLO("best.pt")

def detect_plate_image(frame, model_path=None, last_track_ids=None):
    results = model(frame)
    det_frame = frame.copy()
    plates = []
    new_track_ids = set()

    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            roi = frame[y1:y2, x1:x2]
            ocr_results = ocr.ocr(roi, cls=True)
            for line in ocr_results:
                for _, (text, conf) in line:
                    if conf > 0.5:
                        plates.append(text)
                        cv2.rectangle(det_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(det_frame, text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return det_frame, plates[0] if plates else "-", new_track_ids
