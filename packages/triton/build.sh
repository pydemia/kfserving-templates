#!/bin/bash

TRITON_VERSION=21.04  # ${TRITON_VERSION}

docker build . -f tritonserver-base.Dockerfile -t pydemia/tritonserver-base:triton-${TRITON_VERSION}-py38 > build-tritonserver-base.log 2>&1 && docker push pydemia/tritonserver-base:triton-${TRITON_VERSION}-py38 \
&& docker save pydemia/tritonserver-base:triton-${TRITON_VERSION}-py38 -o pydemia--tritonserver-base:triton-${TRITON_VERSION}-py38.tar.gz \
&& gsutil -m cp ./pydemia--tritonserver-base:triton-${TRITON_VERSION}-py38.tar.gz gs://aiip-runtime-installer/install-offline/pydemia--tritonserver-base:triton-${TRITON_VERSION}-py38.tar.gz
