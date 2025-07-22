FROM python:3.12-slim

# Install dependencies OS
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

WORKDIR /app

# Salin semua file project
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip wheel
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Install PaddleOCR tanpa dependensi tambahan (hindari PyMuPDF error)
RUN pip install --no-deps paddleocr==2.6.1 paddlepaddle==2.6.1

# Preload PaddleOCR biar model otomatis terunduh saat build (opsional)
RUN python3 - <<EOF
from paddleocr import PaddleOCR
print("⬇️ Preloading PaddleOCR model...")
PaddleOCR(use_angle_cls=True, lang='en')
print("✅ PaddleOCR model preloaded!")
EOF

ENV PYTHONUNBUFFERED=1

# Jalankan Gunicorn server
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
