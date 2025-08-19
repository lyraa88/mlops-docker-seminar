FROM python:3.11
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=Asia/Seoul
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 tzdata \
 && rm -rf /var/lib/apt/lists/*

# CPU 전용 휠로 먼저 설치
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchvision==0.18.1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY trigger.py ./trigger.py
ENV SHARE_DIR=/app/share_storage
CMD ["python","-u","trigger.py"]
