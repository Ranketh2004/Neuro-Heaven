FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV, MNE (audio/signal), nibabel, GL, and PyTorch geometric
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    libgomp1 \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# Override the hardcoded host/port in main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
