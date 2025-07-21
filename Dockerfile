# ================= BASE =================
FROM python:3.11-slim

# ================= ENV SETUP =================
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ================= INSTALL DEPENDENCIES =================
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# ================= INSTALL PYTHON DEPENDENCIES =================
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ================= COPY APP =================
COPY . .

# ================= EXPOSE PORT =================
EXPOSE 5000

# ================= ENTRYPOINT =================
CMD ["python", "app.py"]
