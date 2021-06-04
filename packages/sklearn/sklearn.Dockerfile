FROM python:3.7-slim
LABEL maintainer="Youngju Kim yj.kim1@sk.com, Juyoung Jung jyjung16@sk.com"

# RUN apt-get update && \
#     apt-get install libgomp1 && \
#     rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY sklearnserver sklearnserver

RUN pip install --upgrade pip && pip install -e ./sklearnserver

ENTRYPOINT ["python", "-m", "sklearnserver"]
