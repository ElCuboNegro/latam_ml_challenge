# syntax=docker/dockerfile:1.2
FROM amancevice/pandas:1.3.5-python3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /challenge

COPY requirements.txt .

# Actualizar pip
RUN pip install --upgrade pip
RUN pip install pandas

# Instalar dependencias restantes
RUN pip install --no-cache-dir -r requirements.txt

ENV APP_HOME /root
WORKDIR $APP_HOME
COPY /challenge $APP_HOME/challenge
COPY /tests $APP_HOME/tests
COPY /data $APP_HOME/data

EXPOSE 8000

ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000
ENV UVICORN_TIMEOUT_KEEP_ALIVE=120

CMD ["uvicorn", "challenge:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120"]