FROM nvcr.io/nvidia/tensorrt:23.02-py3

RUN apt-get update && \
    apt-get install -y pkg-config && \
    apt-get install -y git && \
    apt-get install -y zlib1g && \
    apt-get install -y zlib1g-dev

RUN DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get install ffmpeg libsm6 libxext6  -y
# upgrade pip
RUN pip install --upgrade pip

# Copy requirement and install
WORKDIR /workspace/
COPY docker/requirements-dev.txt /workspace/
RUN pip install nvidia-pyindex
RUN pip install -r requirements-dev.txt && \
    rm -rf requirements-dev.txt

ARG TRT_VERSION_MAJOR=8
ARG TRT_VERSION_MINOR=5
ARG TRT_VERSION_PATCH=3
ARG TRT_VERSION_BUILD=1
ARG TRT_VERSION_MAJOR_MINOR=$TRT_VERSION_MAJOR.$TRT_VERSION_MINOR
ARG TRT_VERSION_MAJOR_MINOR_PARTCH=$TRT_VERSION_MAJOR.$TRT_VERSION_MINOR.$TRT_VERSION_PATCH
ARG TRT_VERSION_FULL=$TRT_VERSION_MAJOR_MINOR_PARTCH.$TRT_VERSION_BUILD

ARG CUDA_VERSION_MAJOR=12
ARG CUDA_VERSION_MINOR=0
ARG CUDA_VERSION_PATCH=0
ARG CUDA_VERSION_BUILD=031
ARG CUDA_VERSION_MAJOR_MINOR=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
ARG CUDA_VERSION_FULL=$CUDA_VERSION_MAJOR_MINOR.$CUDA_VERSION_PATCH.$CUDA_VERSION_BUILD
ARG CUDNN_VERSION=8.7

ENV TRT_VERSION=$TRT_VERSION_MAJOR.$TRT_VERSION_MINOR.$TRT_VERSION_PATCH.$TRT_VERSION_BUILD+cuda$CUDA_VERSION_FULL

RUN cd /opt
# ENV TRT_TAG "release/$TRT_VERSION_MAJOR_MINOR"
ENV TRT_TAG "release/8.6"
RUN mkdir trt_oss_src && \
   cd trt_oss_src && \
   echo "$PWD Building TRT OSS..." && \
   git clone -b $TRT_TAG https://github.com/nvidia/TensorRT TensorRT && \
   cd TensorRT && \
   git submodule update --init --recursive && \
   mkdir -p build && cd build  && \
   #cmake .. -DGPU_ARCHS="52 53 60 61 70 75 80 86 90" -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu -DTRT_BIN_DIR=`pwd`/out -DCUDA_VERSION=$CUDA_VERSION_MAJOR_MINOR -DCUDNN_VERSION=$CUDNN_VERSION && \
   cmake .. -DGPU_ARCHS="52 53 60 61 70 75 80 86 90" -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu -DTRT_BIN_DIR=`pwd`/out -DCUDA_VERSION=11.8 -DCUDNN_VERSION=$CUDNN_VERSION -DCUBLAS_LIB=/usr/local/cuda-11.8/lib64/stubs/libcublas.so -DCUBLASLT_LIB=/usr/local/cuda-11.8/lib64/stubs/libcublasLt.so && \
   make -j32  && \
   cp libnvinfer_plugin.so.8.6.* /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.$TRT_VERSION_MAJOR_MINOR_PARTCH && \
   cp libnvinfer_plugin_static.a /usr/lib/x86_64-linux-gnu/libnvinfer_plugin_static.a && \
#    cp libnvonnxparser.so.8.6.* /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.$TRT_VERSION_MAJOR_MINOR_PARTCH && \
   cp libnvcaffeparser.so.8.6.* /usr/lib/x86_64-linux-gnu/libnvcaffeparser.so.$TRT_VERSION_MAJOR_MINOR_PARTCH && \
   cp trtexec /usr/local/bin/ && \
   rm -rf trt_oss_src

# Only required if we're building from scratch. Not required if base image is from DLFW
# RUN pip install tensorrt==$TRT_VERSION_FULL && \
#     pip install uff --index-url https://pypi.ngc.nvidia.com && \
#     pip install graphsurgeon --index-url https://pypi.ngc.nvidia.com && \
#     pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com

# Create user that will run commands
ARG user_id=1000
ARG user_name=developer
ARG groups=developer:1000
ARG home=/home/developer
RUN echo "ALL   ALL = (ALL) NOPASSWD: ALL" > /etc/sudoers \
    && mkdir -p "$(dirname $home)"
WORKDIR /workspace/
