FROM python:3.11
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# 1) torch/torchvision을 CPU 휠로 '먼저' 설치  ← CUDA 거대 휠 방지!
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchvision==0.18.1

# 2) 나머지 의존성 (requirements에 torch가 있어도 이미 설치되어서 스킵됨)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mnist_model.py ./mnist_model.py
ENV SHARE_DIR=/app/share_storage
CMD ["python", "-u", "mnist_model.py"]
