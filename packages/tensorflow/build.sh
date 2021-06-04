#!/bin/bash


TF_VERSION="2"

docker build . -f tensorflowserver-base.Dockerfile -t pydemia/tensorflowserver-base:tf${TF_VERSION}-py38 > build-tensorflowserver-base.log 2>&1 && docker push pydemia/tensorflowserver-base:tf${TF_VERSION}-py38 \
&& docker save pydemia/tensorflowserver-base:tf${TF_VERSION}-py38 -o pydemia--tensorflowserver-base:tf${TF_VERSION}-py38.tar.gz \
&& gsutil cp ./pydemia--tensorflowserver-base:tf${TF_VERSION}-py38.tar.gz gs://aiip-runtime-installer/install-offline/pydemia--tensorflowserver-base:tensorflowtf${TF_VERSION}-py38.tar.gz
