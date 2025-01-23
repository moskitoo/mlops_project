FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git libgl1 libglib2.0-0 && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY requirements_ui.txt /app/requirements_ui.txt
COPY src/pv_defection_classification/ui.py /app/ui.py

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_ui.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]