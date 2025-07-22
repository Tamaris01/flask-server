FROM python:3.12-slim

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

# ✅ Salin dan install requirements dulu
COPY requirements.txt .
RUN pip install --upgrade pip wheel
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Baru salin semua file proyek
COPY . .

# ✅ Preload PaddleOCR
RUN python3 - <<EOF
from paddleocr import PaddleOCR
print("⬇️ Preloading PaddleOCR model...")
PaddleOCR(use_angle_cls=True, lang='en')
print("✅ PaddleOCR model preloaded successfully!")
EOF

ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
