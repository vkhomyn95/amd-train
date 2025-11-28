FROM python:3.10

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && apt-get install -y ffmpeg

RUN python -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

RUN python -m pip install "numpy<2.0"

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . ./

EXPOSE 8088

ENTRYPOINT ["python3", "-u", "main.py"]
