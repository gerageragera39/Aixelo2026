FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    LC_ALL=C

# Copy requirements file
COPY requirements.txt /opt/requirements.txt

# 1. Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git \
        ca-certificates \
        build-essential \
        && rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 2. Pin Numpy < 2.0 (CRITICAL for scientific libraries)
RUN python3 -m pip install "numpy<2.0"

# 3. Install PyTorch 2.4.0 (Required for newer transformers)
# Compatible with CUDA 12.1
RUN python3 -m pip install --no-cache-dir \
    torch==2.4.0+cu121 \
    torchvision==0.19.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# 4. Install DGL 2.1.0 (Compatible with PyTorch 2.4 and CUDA 12.1)
RUN python3 -m pip install --no-cache-dir \
    dgl==2.1.0+cu121 \
    -f https://data.dgl.ai/wheels/cu121/repo.html

# 5. Install chemistry stack
RUN python3 -m pip install --no-cache-dir \
    rdkit-pypi \
    dgllife

# 6. Install remaining dependencies
# IMPORTANT: We explicitly update transformers to a stable version if not specified in requirements
RUN python3 -m pip install --no-cache-dir -r /opt/requirements.txt