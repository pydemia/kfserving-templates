FROM python:3.7-slim
LABEL maintainer="Youngju Kim yj.kim1@sk.com, Juyoung Jung jyjung16@sk.com"

RUN apt-get update && \
    apt-get install libgomp1 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY xgbserver xgbserver 
# COPY third_party third_party

# pip 20.x breaks xgboost wheels https://github.com/dmlc/xgboost/issues/5221
RUN pip install pip==19.3.1 && pip install -e ./kfserving
RUN pip install -e ./xgbserver
ENTRYPOINT ["python", "-m", "xgbserver"]
