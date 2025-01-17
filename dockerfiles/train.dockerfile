# # WITH MOUNTING
# FROM python:3.10-slim

# RUN apt update && \
#     apt install --no-install-recommends -y \
#     build-essential gcc g++ git curl gnupg \
#     libgl1 libglib2.0-0 cmake ninja-build && \
#     echo "deb http://packages.cloud.google.com/apt gcsfuse-bullseye main" | tee /etc/apt/sources.list.d/gcsfuse.list && \
#     curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
#     apt update && \
#     apt install --no-install-recommends -y gcsfuse && \
#     apt clean && \
#     rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# COPY requirements.txt requirements.txt
# COPY pyproject.toml pyproject.toml
# COPY src ./src/

# RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --no-cache-dir

# RUN pip install torch==2.5.1 torchvision==0.20.1
# RUN pip install git+https://github.com/facebookresearch/detectron2.git@b1c43ffbc995426a9a6b5c667730091a384e0fa4

# RUN mkdir -p /app/pv-defection-data

# # Entry point for training
# ENTRYPOINT ["sh", "-c", "gcsfuse pv-defection-data /app/pv-defection-data || echo 'GCS mount failed' && \
#           python -u src/pv_defection_classification/train.py"]
# #ENTRYPOINT ["/bin/bash"]



# WITHOUT MOUNTING
FROM python:3.10-slim

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

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --no-cache-dir

RUN pip install torch==2.5.1 torchvision==0.20.1
RUN pip install git+https://github.com/facebookresearch/detectron2.git@b1c43ffbc995426a9a6b5c667730091a384e0fa4

RUN mkdir -p /app/data/processed

RUN python src/pv_defection_classification/data.py

ENTRYPOINT ["python", "-u", "src/pv_defection_classification/train.py"]
