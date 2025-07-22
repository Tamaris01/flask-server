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

# ✅ Pre-download semua model agar tidak di-download ulang saat container start
RUN python3 - <<EOF
import paddlex as pdx
models = [
    'PP-LCNet_x1_0_doc_ori',
    'UVDoc',
    'PP-LCNet_x1_0_textline_ori',
    'PP-OCRv5_server_det',
    'PP-OCRv5_server_rec'
]
for m in models:
    try:
        print(f"⬇️  Preloading model: {m}")
        pdx.load_model(m)
        print(f"✅ Model {m} preloaded successfully!")
    except Exception as e:
        print(f"❌ Failed to preload {m}: {e}")
EOF

# ✅ Atur agar PaddleX hanya menggunakan model lokal (tanpa cek ulang ke server)
ENV PADDLEX_OFFLINE_MODE=1

# ✅ Agar log keluar real-time
ENV PYTHONUNBUFFERED=1

# ✅ (Opsional) YOLO tidak menulis ke /root/.config
ENV YOLO_CONFIG_DIR=/tmp

# ✅ Jalankan Flask via Gunicorn (produksi)
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
