ARG UBUNTU_RELEASE=22.04
FROM ubuntu:${UBUNTU_RELEASE}

LABEL maintainer "https://github.com/ehfd"

ARG UBUNTU_RELEASE

RUN apt-get clean && apt-get update && apt-get upgrade -y && apt-get install --no-install-recommends -y \
        apt-transport-https \
        apt-utils \
        ca-certificates \
        openssh-client \
        curl \
        iptables \
        git \
        gnupg \
        software-properties-common \
        supervisor \
        wget && \
    rm -rf /var/lib/apt/list/*

# NVIDIA Container Toolkit and Docker
RUN mkdir -pm755 /etc/apt/keyrings && curl -fsSL "https://download.docker.com/linux/ubuntu/gpg" | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && chmod a+r /etc/apt/keyrings/docker.gpg && \
    mkdir -pm755 /etc/apt/sources.list.d && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(grep UBUNTU_CODENAME= /etc/os-release | cut -d= -f2 | tr -d '\"') stable" > /etc/apt/sources.list.d/docker.list && \
    mkdir -pm755 /usr/share/keyrings && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
    curl -fsSL "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list" | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null && \
    apt-get update && apt-get install --no-install-recommends -y \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin \
        nvidia-container-toolkit && \
    rm -rf /var/lib/apt/list/* && \
    nvidia-ctk runtime configure --runtime=docker

COPY nvidia-dind/modprobe nvidia-dind/start-docker.sh nvidia-dind/entrypoint.sh /usr/local/bin/
COPY nvidia-dind/supervisor/ /etc/supervisor/conf.d/
COPY nvidia-dind/logger.sh /opt/bash-utils/logger.sh

RUN chmod +x /usr/local/bin/start-docker.sh /usr/local/bin/entrypoint.sh /usr/local/bin/modprobe

VOLUME /var/lib/docker

# install python
RUN apt upgrade python3
RUN apt install pip -y

# update pip & poetry
RUN pip install --upgrade pip
RUN pip install "poetry==1.1.13"

WORKDIR /app
COPY . .

# install package with poetry
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

ENTRYPOINT ["entrypoint.sh"]
CMD ["python3", "server.py"]

EXPOSE 12700 12760