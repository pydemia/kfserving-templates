FROM pydemia/nvidia-
MAINTAINER 

mv /models/archived

WORKDIR /workdir

COPY tritonserver tritonserver
COPY run_tritonserver /usr/local/run_tritonserver
COPY
