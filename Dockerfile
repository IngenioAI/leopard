FROM nvidia/cuda:11.4.1-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
# ENV DOCKER_DRIVER=overlay2

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# install python and PIP
RUN apt update && apt -y upgrade \
  && apt-get -y install \
       python3 python3-dev \
       libffi-dev libssl-dev build-essential git curl bash \
       cargo gcc musl-dev \
  && curl 'https://bootstrap.pypa.io/get-pip.py' -o get-pip.py \
  && python3 get-pip.py \
  && rm get-pip.py

# install docker
RUN apt-get -y install apt-transport-https \
      ca-certificates \
      gnupg2 \
      software-properties-common && \
    curl -fsSL https://download.docker.com/linux/$(. /etc/os-release; echo "$ID")/gpg > /tmp/dkey; apt-key add /tmp/dkey && \
    add-apt-repository \
      "deb [arch=amd64] https://download.docker.com/linux/$(. /etc/os-release; echo "$ID") \
      $(lsb_release -cs) \
      stable" && \
   apt-get update && \
   apt-get -y install docker-ce

# install nvidia-container toolkit
RUN apt-get -y install nvidia-container-toolkit
   
RUN groupadd -f docker

# update pip
RUN pip install --upgrade pip

WORKDIR /app
COPY . .

#install poetry with pip
RUN pip install "poetry==1.1.13"
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi


CMD ["python3", "server.py"]

EXPOSE 12700

# 현재 도커를 쓰면 도커안의 도커에서 볼륨 파일에 접근할 수 없는 문제가 있다