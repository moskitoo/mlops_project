# Build base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04 AS builder

ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install --no-install-recommends -y \
    python3.10 python3.10-venv python3.10-distutils python3.10-dev python3-pip \
    build-essential gcc g++ git curl gnupg \
   # python3 python3-distutils python3-venv python3-pip \
    libgl1 libglib2.0-0 cmake ninja-build && \
    #ln -s /usr/bin/python3.10 /usr/bin/python && \
    apt-get remove -y python3.8 && \
    rm -rf /usr/bin/python3.8 /usr/lib/python3.8 /usr/local/lib/python3.8 && \    
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Python 3.10 setup (cuda 12.1 default is 3.8 which does not work with pytorch)
RUN rm -rf /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    python --version && python3 --version && which python3.10 && \
    python3.10 -m ensurepip --upgrade && \
    python3.10 -m pip install --upgrade pip && \
    which pip && pip --version

WORKDIR /app

# Install PyTorch
RUN python3.10 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())" 

# Install Detectron2
ENV PATH="/opt/venv/bin:$PATH"	
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
#RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.4/index.html

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src ./src/
COPY data/raw ./data/raw 

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    libjpeg-dev zlib1g-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Reinstall Pytorch (Basically forcing the correct versions - probably a better way to do this)
RUN /opt/venv/bin/pip install --no-cache-dir --force-reinstall torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121

RUN /opt/venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); import torchvision; print(torchvision.__version__);"

#RUN python src/pv_defection_classification/data.py
RUN /opt/venv/bin/python src/pv_defection_classification/data.py

# Build runtime image
FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    python3.10 python3.10-venv python3.10-distutils python3.10-pip \
    libgl1 libglib2.0-0 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built environment from build stage
COPY --from=builder /app /app
COPY --from=builder /opt/venv /opt/venv

RUN mkdir -p /app/data/processed

RUN python --version && python3 --version && /opt/venv/bin/python --version

#ENTRYPOINT ["python3", "-u", "src/pv_defection_classification/train.py"]
ENTRYPOINT ["/opt/venv/bin/python", "-u", "src/pv_defection_classification/train.py"]
