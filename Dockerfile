FROM python:3.8-slim-buster

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \  
    xvfb \
    python-opengl \
    ffmpeg \
    git \
 && rm -rf /var/lib/apt/lists/*


RUN pip3 install -U pip setuptools wheel
RUN pip3 install -r requirements.txt