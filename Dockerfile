FROM python:3.10

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && apt-get install -y ffmpeg

RUN apt-get update && apt-get install -y \
    sox \
    libsox-dev \
    libsox-fmt-all \
    libsndfile1

RUN python -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

RUN python -m pip install "numpy<2.0"

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . ./

EXPOSE 8088

ENTRYPOINT ["python3", "-u", "main.py"]
