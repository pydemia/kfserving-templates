# FROM nvcr.io/nvidia/tritonserver:20.08-py3
FROM nvcr.io/nvidia/tritonserver:21.04-py3
LABEL maintainer="Youngju Kim pydemia@gmail.com"

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    software-properties-common \
    git curl vim htop

ENV CONDA_DIR="/opt/conda"
# Miniconda3-py38_4.8.3-Linux-x86_64.sh
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh -o miniconda.sh \
    && (echo "\n";echo yes; echo $CONDA_DIR; echo yes) | bash -f miniconda.sh \
    && rm miniconda.sh
# Miniconda3-py39_4.9.2-Linux-x86_64.sh
# RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh -o miniconda.sh \
#     && (echo "\n";echo yes; echo $CONDA_DIR; echo yes) | bash -f miniconda.sh \
#     && rm miniconda.sh
ENV PATH="${CONDA_DIR}/bin:${PATH}"

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workdir:${PYTHONPATH}

ENV HTTP_PORT=8080
ENV GRPC_PORT=8081
ENV TRITON_HTTP_PORT=18080
ENV TRITON_GRPC_PORT=18081
ENV MODEL_DIR="/models"
ENV MODEL_NAME="model"
ENV MODEL_VERSION="1"

EXPOSE 8080 8081

RUN mkdir -p /workdir
WORKDIR /workdir

# COPY graphdef/simple /models/
COPY savedmodel /models/archived
COPY tritonserver tritonserver
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY README.md README.md

COPY multipart-requests multipart-requests

# COPY run_tritonserver /usr/local/bin/run_tritonserver
# RUN chmod +x /usr/local/bin/run_tritonserver

RUN pip install --upgrade pip && \
    pip install -e . && \
    pip install -r ./requirements.txt --no-cache-dir
RUN pip install tritonclient[all] --extra-index-url=https://pypi.ngc.nvidia.com --no-cache-dir
# RUN pip install tritonclient[all] --extra-index-url=https://pypi.ngc.nvidia.com

RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY run_tritonserver /usr/local/bin/run_tritonserver
RUN chmod +x /usr/local/bin/run_tritonserver
# RUN echo "/opt/tritonserver/nvidia_entrypoint.sh" >> /etc/profile

ENTRYPOINT ["run_tritonserver"]
