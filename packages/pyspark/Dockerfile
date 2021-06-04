# FROM python:3.7-slim
# FROM bitnami/spark:3.1.1  # python3.6
# FROM bitnami/spark:2.4.6  # python3.6
# continuumio/anaconda3:2020.02  # python3.7
# FROM continuumio/miniconda3:4.9.2
# FROM continuumio/miniconda3:4.8.2  # python3.7
# FROM jupyter/pyspark-notebook:spark-2  # python3.7
# FROM jupyter/pyspark-notebook:spark-3.1.1  # python3.9
FROM pydemia/pysparkserver-base:spark-3.1.1-py3.9
LABEL maintainer="pydemia@gmail.com"


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
COPY pysparkserver pysparkserver
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY README.md README.md
COPY run_pysparkserver /usr/local/bin/run_pysparkserver

RUN chmod +x /usr/local/bin/run_pysparkserver

# pip install pyspark[sql]==2.4.7
RUN pip install --upgrade pip && \
    pip install -e . && \
    pip install -r ./requirements.txt --no-cache-dir

RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ENTRYPOINT ["run_pysparkserver"]
