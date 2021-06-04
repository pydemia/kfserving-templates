# FROM nvcr.io/nvidia/tritonserver:20.08-py3
FROM pydemia/tritonserver-base:triton-20.08-py38
LABEL maintainer="Youngju Kim pydemia@gmail.com"

WORKDIR /workdir

# COPY savedmodel/simple /models/
COPY savedmodel /models/archived

EXPOSE 8080 8081
