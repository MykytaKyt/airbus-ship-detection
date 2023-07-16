FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY weights/model_final.h5 ./weights/model_final.h5

CMD ["python", "app.py", "--model-path", "weights/model_final.h5", "--image-size", "256"]

#docker build -t fastapi-app .
#docker run -p 8000:8000 fastapi-app