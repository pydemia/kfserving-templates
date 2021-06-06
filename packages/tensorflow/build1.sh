#!/bin/bash

# See: https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes
# 21.04: cuda11.3.0, tf2, py38 | 20.03: cuda10.2.89, tf2, py36

# NV_VERSION="21.04"  
# CUDA_VERSION="11.3"
# TF_VERSION="2"
# PY_VERSION="38"
NV_VERSION="20.03"  
CUDA_VERSION="10.2"
TF_VERSION="2"
PY_VERSION="36"

IMAGE_TAG="${NV_VERSION}-cuda${CUDA_VERSION}-tf${TF_VERSION}-py${PY_VERSION}"

printf ${IMAGE_TAG}

docker build . -f tensorflowserver-base-${NV_VERSION}.Dockerfile \
    -t pydemia/tensorflowserver-base:${IMAGE_TAG} \
    > build-tensorflowserver-base.log 2>&1 \
&& docker push pydemia/tensorflowserver-base:${IMAGE_TAG} \
&& docker save pydemia/tensorflowserver-base:${IMAGE_TAG} \
    -o pydemia--tensorflowserver-base:${IMAGE_TAG}.tar.gz \
&& gsutil -m cp ./pydemia--tensorflowserver-base:${IMAGE_TAG}.tar.gz \
    gs://aiip-runtime-installer/install-offline/pydemia--tensorflowserver-base:${IMAGE_TAG}.tar.gz
