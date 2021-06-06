# FROM python:3.7-slim
# LABEL maintainer="Youngju Kim yj.kim1@sk.com, Juyoung Jung jyjung16@sk.com"

# # RUN apt-get update && \
# #     apt-get install libgomp1 && \
# #     rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# COPY modelserver modelserver
# COPY run_modelserver /usr/local/bin/run_modelserver

# RUN pip install --upgrade pip && \
#     pip install -e ./modelserver && \
#     pip install -r ./modelserver/requirements.txt --use-feature=2020-resolver

# ENTRYPOINT ["./run_modelserver"]

FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3
LABEL maintainer="pydemia@gmail.com"

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    software-properties-common \
    git curl vim htop

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
COPY tensorflowserver tensorflowserver
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY README.md README.md
COPY run_tfserver /usr/local/bin/run_tfserver

RUN chmod +x /usr/local/bin/run_tfserver

RUN pip install --upgrade pip && \
    pip install -e . && \
    pip install -r ./requirements.txt --no-cache-dir

RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ENTRYPOINT ["run_tfserver"]
