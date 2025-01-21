# Start FROM PyTorch image https://hub.docker.com/r/pytorch/pytorch or nvcr.io/nvidia/pytorch:23.03-py3
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MKL_THREADING_LAYER=GNU \
    OMP_NUM_THREADS=1 

# Downloads to user config dir
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

# Install linux packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc git zip unzip wget curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 libsm6 && \
    rm -rf /var/lib/apt/lists/*

# Security updates
RUN apt upgrade --no-install-recommends -y openssl tar

# Install google cloud sdk
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update -y && \
    apt-get install -y google-cloud-cli && \
    gcloud --version

# Create working directory
WORKDIR /app

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src ./src/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Remove unnecessary git configurations
#RUN sed -i '/^\[http "https:\/\/github\.com\/"\]/,+1d' .git/config

# Install pip packages
RUN pip install uv
RUN uv pip install --system -e ".[export]" tensorrt-cu12 "albumentations>=1.4.6" comet pycocotools

ADD https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt .

ENV GOOGLE_APPLICATION_CREDENTIALS="/root/.config/gcloud/application_default_credentials.json"
RUN mkdir -p /root/.config/gcloud
RUN gcloud --version && gsutil --version

CMD ["python", "-u", "src/pv_defection_classification/train.py"]