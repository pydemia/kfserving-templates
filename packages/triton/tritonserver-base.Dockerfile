# FROM nvcr.io/nvidia/tritonserver:20.08-py3
FROM nvcr.io/nvidia/tritonserver:21.04-py3
LABEL maintainer="Youngju Kim pydemia@gmail.com"

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    software-properties-common \
    git curl vim htop

ENV CONDA_DIR="/opt/conda"
# Miniconda3-py38_4.8.3-Linux-x86_64.sh
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh -o miniconda.sh \
    && (echo "\n";echo yes; echo $CONDA_DIR; echo yes) | bash -f miniconda.sh \
    && rm miniconda.sh
# Miniconda3-py39_4.9.2-Linux-x86_64.sh
# RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh -o miniconda.sh \
#     && (echo "\n";echo yes; echo $CONDA_DIR; echo yes) | bash -f miniconda.sh \
#     && rm miniconda.sh
ENV PATH="${CONDA_DIR}/bin:${PATH}"

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workdir:${PYTHONPATH}

ENV HTTP_PORT=8080
ENV GRPC_PORT=8081
ENV TRITON_HTTP_PORT=8000
ENV TRITON_GRPC_PORT=8001
ENV TRITON_METRIC_PORT=8002
ENV MODEL_DIR="/models"
ENV MODEL_NAME="model"
ENV MODEL_VERSION="1"

EXPOSE 8080 8081 8000 8002

RUN mkdir -p /workdir
WORKDIR /workdir

# COPY graphdef/simple /models/
COPY savedmodel /models/archived
COPY tritonserver tritonserver
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY README.md README.md
COPY config.md config.md

# COPY run_tritonserver /usr/local/bin/run_tritonserver
# RUN chmod +x /usr/local/bin/run_tritonserver

RUN pip install --upgrade pip && \
    pip install -e . && \
    pip install -r ./requirements.txt --no-cache-dir
RUN pip install tritonclient[all] --extra-index-url=https://pypi.ngc.nvidia.com --no-cache-dir
# RUN pip install tritonclient[all] --extra-index-url=https://pypi.ngc.nvidia.com

RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY run_tritonserver /usr/local/bin/run_tritonserver
RUN chmod +x /usr/local/bin/run_tritonserver
# RUN echo "/opt/tritonserver/nvidia_entrypoint.sh" >> /etc/profile

### Custom Settings ######################
RUN apt-get update -q && \
    echo "69" | echo "6" | apt install -y -qq dnsutils && \
    apt-get install -y curl software-properties-common \
    git vim bash-completion unzip tree \
    apt-transport-https ca-certificates iputils-ping \
    gnupg jq python3.8 python3-pip mysql-client

# Add 3.8 to the available alternatives and set python3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --set python /usr/bin/python3.8 && \
    pip3 install --upgrade pip && pip install jq

RUN apt-get install -y bash-completion
RUN bash -c "$(curl -fsSL https://raw.githubusercontent.com/pydemia/pydemia-theme/master/install_themes.sh)"


# kubectl
ENV KUBECTL_VERSION="v1.20.0"
RUN curl -fsSL -O https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl && \
    chmod +x kubectl && \
    mv kubectl /usr/local/bin/kubectl

RUN echo 'source <(kubectl completion bash)' >> /etc/bash.bashrc

# k9s
RUN k9s_version="v0.24.2" && \
    k_os_type="linux" && \
    curl -L https://github.com/derailed/k9s/releases/download/"${k9s_version}"/k9s_"$(echo "${k_os_type}" |sed 's/./\u&/')"_x86_64.tar.gz -o k9s.tar.gz && \
    mkdir -p ./k9s && \
    tar -zxf k9s.tar.gz -C ./k9s && \
    mv ./k9s/k9s /usr/local/bin/ && \
    rm -rf ./k9s ./k9s.tar.gz && \
    echo "\nInstalled in: $(which k9s)"


# VIM IDE
RUN git clone https://github.com/rapphil/vim-python-ide.git vim-python-ide && \
    cd vim-python-ide && echo ""| echo ""| ./install.sh && \
    cd .. && rm -rf vim-python-ide && \
    sed -i 's/^colorscheme cobalt2/"colorscheme Monokai\ncolorscheme cobalt2/' ~/.vimrc

ENTRYPOINT ["run_tritonserver"]
