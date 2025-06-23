# Gunakan image Python yang lebih kecil dan ringan sebagai base image
FROM python:3.12-slim

# Install dependencies sistem yang diperlukan oleh OpenCV dan pustaka lainnya
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Set direktori kerja di dalam container
WORKDIR /app

# Salin file aplikasi ke dalam direktori /app di dalam container
COPY . /app

# Install dependensi Python dari file requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Tentukan perintah yang akan dijalankan untuk menjalankan aplikasi
CMD ["python3", "app.py"]
