FROM continuumio/miniconda3:4.9.2
LABEL maintainer="pydemia@gmail.com"

USER 0

# Spark dependencies
# Default values can be overridden at build time
# (ARGS are in lower case to distinguish them from ENV)
ARG spark_version="3.1.1"
ARG hadoop_version="3.2"
ARG spark_checksum="E90B31E58F6D95A42900BA4D288261D71F6C19FA39C1CB71862B792D1B5564941A320227F6AB0E09D946F16B8C1969ED2DEA2A369EC8F9D2D7099189234DE1BE"
ARG openjdk_version="11"

# To avoid error: Package openjdk-11-jre-headless:amd64 which provides java8-runtime-headless is not configured yet.
RUN mkdir -p /usr/share/man/man1
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    software-properties-common \
    git curl vim htop locales \
    "openjdk-${openjdk_version}-jre-headless" \
    ca-certificates-java

# ENV CONDA_DIR="/opt/conda"
# RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh -o miniconda.sh \
#     && (echo "\n";echo yes; echo $CONDA_DIR; echo yes) | bash -f miniconda.sh \
#     && rm miniconda.sh
# ENV PATH="${CONDA_DIR}/bin:${PATH}"


# Spark installation
ENV APACHE_SPARK_VERSION="${spark_version}" \
    HADOOP_VERSION="${hadoop_version}"

WORKDIR /tmp
RUN wget -q "https://archive.apache.org/dist/spark/spark-${APACHE_SPARK_VERSION}/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" && \
    echo "${spark_checksum} *spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" | sha512sum -c - && \
    tar xzf "spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" -C /usr/local --owner root --group root --no-same-owner && \
    rm "spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz"

WORKDIR /usr/local

# Configure Spark
ENV SPARK_HOME=/usr/local/spark
ENV SPARK_OPTS="--driver-java-options=-Xms1024M --driver-java-options=-Xmx4096M --driver-java-options=-Dlog4j.logLevel=info" \
    PATH=$PATH:$SPARK_HOME/bin

RUN ln -s "spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" spark && \
    # Add a link in the before_notebook hook in order to source automatically PYTHONPATH
    mkdir -p /usr/local/bin/before-notebook.d && \
    ln -s "${SPARK_HOME}/sbin/spark-config.sh" /usr/local/bin/before-notebook.d/spark-config.sh

# Fix Spark installation for Java 11 and Apache Arrow library
# see: https://github.com/apache/spark/pull/27356, https://spark.apache.org/docs/latest/#downloading
RUN cp -p "$SPARK_HOME/conf/spark-defaults.conf.template" "$SPARK_HOME/conf/spark-defaults.conf" && \
    echo 'spark.driver.extraJavaOptions -Dio.netty.tryReflectionSetAccessible=true' >> $SPARK_HOME/conf/spark-defaults.conf && \
    echo 'spark.executor.extraJavaOptions -Dio.netty.tryReflectionSetAccessible=true' >> $SPARK_HOME/conf/spark-defaults.conf

# Install pyarrow
RUN pip install "pyarrow<4.1,>=4.0"

CMD ["bash" "-c"]
