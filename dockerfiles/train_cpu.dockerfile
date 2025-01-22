# Use official Python base image for reproducibility (3.11.10 for export and 3.12.6 for inference)
FROM python:3.11.10-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Downloads to user config dir
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

# Install linux packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip git zip unzip wget curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /ultralytics

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src ./src/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy contents and configure git
#RUN sed -i '/^\[http "https:\/\/github\.com\/"\]/,+1d' .git/config

# Install pip packages
RUN pip install uv
RUN uv pip install --system -e ".[export]" --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-first-match


ADD https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt .

# RUN python src/pv_defection_classification/data.py
# RUN python src/pv_defection_classification/data.py --raw-data-path data/raw/pv_defection/dataset_1

CMD ["python", "-u", "src/pv_defection_classification/train.py", "--data-path", "/gcs/test-pv-2/data/processed/pv_defection/pv_defection.yaml"]
