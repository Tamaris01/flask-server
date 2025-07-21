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

# Install dependensi dengan PEP517 untuk menghindari warning GPUtil
RUN pip install --use-pep517 --no-cache-dir -r requirements.txt

# Agar log keluar real-time
ENV PYTHONUNBUFFERED=1

# Jalankan Flask via Gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
