
# Base image
FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN apt update && \
    apt install --no-install-recommends -y \
    build-essential gcc g++ git curl gnupg \
    libgl1 libglib2.0-0 cmake ninja-build && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*
    
WORKDIR /app

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src ./src/
COPY data/raw ./data/raw

RUN pip install -r requirements.txt --no-cache-dir

RUN pip install torch==2.5.1 torchvision==0.20.1
RUN pip install git+https://github.com/facebookresearch/detectron2.git@b1c43ffbc995426a9a6b5c667730091a384e0fa4
RUN mkdir -p /app/data/processed
RUN python src/pv_defection_classification/data.py

ENTRYPOINT ["python", "-u", "src/pv_defection_classification/train.py"]