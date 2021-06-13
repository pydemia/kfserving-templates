FROM python:3.7-buster


LABEL maintainer="Youngju Kim pydemia@gmail.com"

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    software-properties-common \
    git curl vim htop

RUN mkdir -p /workdir
WORKDIR /workdir

COPY runtime_client runtime_client
COPY setup.py setup.py
# COPY requirements.txt requirements.txt
COPY README.md README.md

RUN pip install --upgrade pip && \
    pip install -e . --no-cache-dir

RUN pip install numba --no-cache-dir

RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
