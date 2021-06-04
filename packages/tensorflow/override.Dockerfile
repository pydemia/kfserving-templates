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

FROM pydemia/tensorflowserver-base:tf-py38
LABEL maintainer="pydemia@gmail.com"

WORKDIR /workdir

COPY tensorflowserver tensorflowserver
