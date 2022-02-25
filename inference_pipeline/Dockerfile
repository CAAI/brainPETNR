# syntax=docker/dockerfile:1

FROM python:3.9-buster

RUN pip3 install --upgrade pip setuptools wheel

RUN mkdir /app
WORKDIR /app

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "microservice.py"]