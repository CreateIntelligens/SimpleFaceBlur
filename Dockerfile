FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt flask flask-cors

EXPOSE 8905

CMD ["sh", "-c", "if [ ! -f Yolo10m/model.onnx ]; then mkdir -p Yolo10m && curl -L -o Yolo10m/model.onnx https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8n-face/model.onnx; fi && python -u api_server.py"]
