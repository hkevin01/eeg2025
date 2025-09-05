FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 python3-pip python3-venv git wget curl ca-certificates \
  build-essential libssl-dev libffi-dev \
  libsndfile1 libgl1-mesa-glx libglib2.0-0 \
  libsm6 libxext6 libxrender-dev libgomp1 \
  libfftw3-dev pkg-config \
  && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace/eeg2025

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Optional: link to parent repo code if mounted
ENV PYTHONPATH=/workspace/eeg2025:/workspace

COPY . .

# Create necessary directories
RUN mkdir -p data/bids_symlinks data/splits outputs runs logs

CMD ["bash", "-lc", "sleep infinity"]
