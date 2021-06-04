# FROM python:3.7-slim
FROM continuumio/miniconda3:4.9.2
LABEL maintainer="pydemia@gmail.com"

# RUN apt-get update && \
#     apt-get install libgomp1 && \
#     rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN apt-get update && \
    apt-get install --no-install-recommends -y git

ENV MODEL_NAME="model"
ENV HTTP_PORT=8080
ENV GRPC_PORT=8081
ENV MODEL_DIR="/mnt/models"

EXPOSE 8080 8081

RUN mkdir -p /workdir
WORKDIR /workdir

COPY pysparkserver pysparkserver
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY README.md README.md
COPY run_pysparkserver /usr/local/bin/run_pysparkserver

RUN chmod +x /usr/local/bin/run_pysparkserver

RUN pip install --upgrade pip && \
    pip install -e . && \
    pip install -r ./requirements.txt --no-cache-dir

RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ENTRYPOINT ["run_pysparkserver"]

FROM jupyter/scipy-notebook

# USER root

# # Spark dependencies
# # Default values can be overridden at build time
# # (ARGS are in lower case to distinguish them from ENV)
# ARG spark_version="3.1.1"
# ARG hadoop_version="3.2"
# ARG spark_checksum="E90B31E58F6D95A42900BA4D288261D71F6C19FA39C1CB71862B792D1B5564941A320227F6AB0E09D946F16B8C1969ED2DEA2A369EC8F9D2D7099189234DE1BE"
# ARG openjdk_version="11"

# ENV APACHE_SPARK_VERSION="${spark_version}" \
#     HADOOP_VERSION="${hadoop_version}"

# RUN apt-get -y update && \
#     apt-get install --no-install-recommends -y \
#     "openjdk-${openjdk_version}-jre-headless" \
#     ca-certificates-java && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*

# # Spark installation
# WORKDIR /tmp
# RUN wget -q "https://archive.apache.org/dist/spark/spark-${APACHE_SPARK_VERSION}/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" && \
#     echo "${spark_checksum} *spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" | sha512sum -c - && \
#     tar xzf "spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" -C /usr/local --owner root --group root --no-same-owner && \
#     rm "spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz"

# WORKDIR /usr/local

# # Configure Spark
# ENV SPARK_HOME=/usr/local/spark
# ENV SPARK_OPTS="--driver-java-options=-Xms1024M --driver-java-options=-Xmx4096M --driver-java-options=-Dlog4j.logLevel=info" \
#     PATH=$PATH:$SPARK_HOME/bin

# RUN ln -s "spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" spark && \
#     # Add a link in the before_notebook hook in order to source automatically PYTHONPATH
#     mkdir -p /usr/local/bin/before-notebook.d && \
#     ln -s "${SPARK_HOME}/sbin/spark-config.sh" /usr/local/bin/before-notebook.d/spark-config.sh

# # Fix Spark installation for Java 11 and Apache Arrow library
# # see: https://github.com/apache/spark/pull/27356, https://spark.apache.org/docs/latest/#downloading
# RUN cp -p "$SPARK_HOME/conf/spark-defaults.conf.template" "$SPARK_HOME/conf/spark-defaults.conf" && \
#     echo 'spark.driver.extraJavaOptions -Dio.netty.tryReflectionSetAccessible=true' >> $SPARK_HOME/conf/spark-defaults.conf && \
#     echo 'spark.executor.extraJavaOptions -Dio.netty.tryReflectionSetAccessible=true' >> $SPARK_HOME/conf/spark-defaults.conf

# USER $NB_UID

# # Install pyarrow
# RUN conda install --quiet --yes --satisfied-skip-solve \
#     'pyarrow=4.0.*' && \
#     conda clean --all -f -y && \
#     fix-permissions "${CONDA_DIR}" && \
#     fix-permissions "/home/${NB_USER}"

# WORKDIR $HOME