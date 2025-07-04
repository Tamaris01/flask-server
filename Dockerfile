# Gunakan image Python slim yang ringan
FROM python:3.12-slim

# Install dependencies sistem untuk OpenCV, PaddleOCR, dan YOLO SDK
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

# Salin semua file proyek ke dalam container
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install PaddlePaddle eksplisit agar PaddleOCR berjalan stabil
RUN pip install paddlepaddle

# Install semua dependencies YOLO SDK + PaddleOCR
RUN pip install --no-cache-dir -r requirements.txt \
    paddleocr \
    'inference[transformers]' \
    'inference[grounding-dino]' \
    'inference[yolo-world]' \
    'inference[sam]' \
    'inference[gaze]'

# Agar log keluar real-time
ENV PYTHONUNBUFFERED=1

# Jalankan server menggunakan Gunicorn dengan post_fork agar pipeline YOLO berjalan otomatis
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app", "-c", "gunicorn_conf.py"]
