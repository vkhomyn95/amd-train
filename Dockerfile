FROM python:3.10

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . ./

EXPOSE 8088

ENTRYPOINT ["python3", "-u", "main.py"]
