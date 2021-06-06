#!/bin/bash


docker build . -t pydemia/runtime-client \
    > build-runtime-client.log 2>&1 \
&& docker push pydemia/runtime-client \
&& docker save pydemia/runtime-client \
    -o ./pydemia--runtime-client.tar.gz \
&& gsutil -m cp ./pydemia--runtime-client.tar.gz \
    gs://aiip-runtime-installer/install-offline/pydemia--runtime-client.tar.gz
