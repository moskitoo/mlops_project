# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

#ENV GOOGLE_APPLICATION_CREDENTIALS: ${{ secrets.GCLOUD_SERVICE_KEY }}
EXPOSE 8080
WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc libgl1 libglib2.0-0 && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY src/pv_defection_classification/pv_monitoring_api.py pv_monitoring_api.py
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install ultralytics fastapi opencv-python numpy pandas scikit-learn evidently onnxruntime google-api-python-client google-cloud-storage google-cloud-core --no-cache-dir --verbose
#RUN pip install fastapi google-cloud-storage scikit-learn evidently pandas --no-cache-dir
RUN pip install . --no-deps --no-cache-dir --verbose

CMD ["uvicorn", "pv_monitoring_api:app", "--port", "8080", "--host", "0.0.0.0", "--workers", "1"]


