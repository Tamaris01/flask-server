# Gunakan image Python slim
FROM python:3.12-slim

# Install dependencies sistem untuk OpenCV, Torch, dan ultralytics
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    build-essential \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Atur direktori kerja
WORKDIR /app

# Salin semua file ke dalam container
COPY . .

# Upgrade pip & install Python dependencies
RUN pip install --upgrade pip

# Install dependencies dari requirements.txt tanpa cache
RUN pip install --no-cache-dir -r requirements.txt

# Jalankan server menggunakan Gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
