# FROM python:3.7-slim
FROM continuumio/miniconda3:4.8.2
LABEL maintainer="pydemia@gmail.com"

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    git gcc

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workdir:${PYTHONPATH}

ENV HTTP_PORT=8080
ENV GRPC_PORT=8081
ENV MODEL_DIR="/models"
ENV MODEL_NAME="model"
ENV MODEL_VERSION="1"

EXPOSE 8080 8081

RUN mkdir -p /workdir
WORKDIR /workdir

COPY models /models
# COPY example_models /models
COPY sklearnserver sklearnserver
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY README.md README.md
COPY run_sklearnserver /usr/local/bin/run_sklearnserver

RUN chmod +x /usr/local/bin/run_sklearnserver

RUN pip install --upgrade pip && \
    pip install -e . && \
    pip install -r ./requirements.txt --no-cache-dir

RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ENTRYPOINT ["run_sklearnserver"]
