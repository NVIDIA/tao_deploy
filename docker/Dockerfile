ARG FROM_SCRIPTS_IMAGE=gitlab-master.nvidia.com:5005/dl/devops/build-scripts:latest
FROM ${FROM_SCRIPTS_IMAGE} AS build-scripts

FROM nvcr.io/nvidia/tensorrt:25.09-py3

RUN apt-get update && \
    apt-get install -y pkg-config && \
    apt-get install -y git && \
    apt-get install -y zlib1g && \
    apt-get install -y zlib1g-dev && \
    apt --only-upgrade -y install openssl && \
    apt --only-upgrade -y install libpmix2

RUN DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get install ffmpeg libsm6 libxext6  -y
# upgrade pip
RUN pip install --upgrade pip

WORKDIR /workspace/
COPY docker/requirements-dev.txt /workspace/
RUN pip install -r requirements-dev.txt && \
    rm -rf requirements-dev.txt

# TAO-CORE microservice dependencies
COPY docker/requirements-apt.txt /workspace/
RUN apt-get update && apt-get install --no-install-suggests --no-install-recommends --yes $(cat /workspace/requirements-apt.txt)
RUN rm -rf /workspace/requirements-apt.txt


# Create user that will run commands
ARG user_id=1000
ARG user_name=developer
ARG groups=developer:1000
ARG home=/home/developer
RUN echo "ALL   ALL = (ALL) NOPASSWD: ALL" > /etc/sudoers \
    && mkdir -p "$(dirname $home)"
WORKDIR /workspace/