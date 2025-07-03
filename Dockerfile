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

# Install dependencies dari requirements.txt tanpa cache
RUN pip install --no-cache-dir -r requirements.txt

# Install PaddleOCR + PaddlePaddle jika belum dicantumkan di requirements.txt
RUN pip install --no-cache-dir paddleocr paddlepaddle

# Suppress inference SDK warnings untuk multimodal models agar log bersih
ENV PALIGEMMA_ENABLED=False
ENV FLORENCE2_ENABLED=False
ENV QWEN_2_5_ENABLED=False

# Jalankan server menggunakan Gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
