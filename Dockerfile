# syntax=docker/dockerfile:1.2
FROM python:3.11-slim

WORKDIR /challenge

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY challenge/ .

EXPOSE 8000

ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000
ENV UVICORN_TIMEOUT_KEEP_ALIVE=120

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120"]