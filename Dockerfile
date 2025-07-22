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

# ✅ Pre-download PaddleOCR models agar tidak di-download ulang saat runtime
RUN python3 - <<EOF
from paddleocr import PaddleOCR
print("⬇️  Preloading PaddleOCR model...")
ocr = PaddleOCR(use_angle_cls=True, lang='en')
print("✅ PaddleOCR model preloaded successfully!")
EOF

# ✅ Agar log keluar real-time
ENV PYTHONUNBUFFERED=1

# ✅ Jalankan Flask via Gunicorn (produksi)
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
