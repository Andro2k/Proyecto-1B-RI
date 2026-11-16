# Imagen para Hugging Face Spaces (Docker, puerto 7860) o Render (usa $PORT).
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY frontend ./frontend
RUN python -m backend.scripts.prefetch_model
ENV PYTHONUNBUFFERED=1
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
