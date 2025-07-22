# Gunakan Python slim stabil
FROM python:3.12-slim

# Install dependencies sistem untuk OpenCV, PaddleOCR, YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    build-essential \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Atur direktori kerja
WORKDIR /app

# Salin semua file proyek
COPY . .

# Upgrade pip dan install wheel
RUN pip install --upgrade pip wheel

# Install dependensi
RUN pip install --use-pep517 --no-cache-dir -r requirements.txt

# ✅ Pre-download semua model PaddleOCR agar tidak download ulang saat runtime
RUN python3 - <<EOF
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    det_model_dir="/root/.paddlex/official_models/PP-OCRv5_server_det",
    rec_model_dir="/root/.paddlex/official_models/PP-OCRv5_server_rec",
    use_angle_cls=True
)
print("✅ PaddleOCR preloaded successfully!")
EOF

# ✅ Paksa hanya pakai model lokal
ENV PPD_DOWNLOAD_FROM_SERVER=0

# ✅ Agar log keluar real-time
ENV PYTHONUNBUFFERED=1

# ✅ YOLO tidak menulis ke /root/.config
ENV YOLO_CONFIG_DIR=/tmp

# ✅ Jalankan Flask via Gunicorn (produksi)
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
