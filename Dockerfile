# Gunakan image python resmi
FROM python:3.12-slim

# Install dependensi untuk OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Atur direktori kerja
WORKDIR /app

# Copy semua file ke dalam container
COPY . .

# Install dependensi Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Jalankan Flask pakai Gunicorn
CMD ["gunicorn", "-w", "3", "-b", "0.0.0.0:5000", "app:app"]
