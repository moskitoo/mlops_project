# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

#ENV GOOGLE_APPLICATION_CREDENTIALS: ${{ secrets.GCLOUD_SERVICE_KEY }}

WORKDIR /bento

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc libgl1 libglib2.0-0 && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY src/pv_defection_classification/bentoml_service.py bentoml_service.py
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["bentoml", "serve", "src.pv_defection_classification.bentoml_service:PVClassificationService"]
