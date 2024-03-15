FROM nvcr.io/nvidia/tensorrt:23.12-py3

RUN apt-get update && \
    apt-get install -y pkg-config && \
    apt-get install -y git && \
    apt-get install -y zlib1g && \
    apt-get install -y zlib1g-dev

RUN apt --only-upgrade install openssl && \
    apt --only-upgrade install libpmix2

RUN DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get install ffmpeg libsm6 libxext6  -y
# upgrade pip
RUN pip install --upgrade pip

# Copy requirement and install
WORKDIR /workspace/
COPY docker/requirements-dev.txt /workspace/
RUN pip install nvidia-pyindex Cython==0.29.36
RUN pip install -r requirements-dev.txt && \
    rm -rf requirements-dev.txt
RUN pip install scikit-learn==0.24.2 --no-build-isolation

ARG TRT_VERSION_MAJOR=8
ARG TRT_VERSION_MINOR=6
ARG TRT_VERSION_PATCH=1
ARG TRT_VERSION_BUILD=6
ARG TRT_VERSION_MAJOR_MINOR=$TRT_VERSION_MAJOR.$TRT_VERSION_MINOR
ARG TRT_VERSION_MAJOR_MINOR_PATCH=$TRT_VERSION_MAJOR.$TRT_VERSION_MINOR.$TRT_VERSION_PATCH
ARG TRT_VERSION_FULL=$TRT_VERSION_MAJOR_MINOR_PATCH.$TRT_VERSION_BUILD

ARG CUDA_VERSION_MAJOR=12
ARG CUDA_VERSION_MINOR=0
ARG CUDA_VERSION_PATCH=1
ARG CUDA_VERSION_BUILD=011
ARG CUDA_VERSION_MAJOR_MINOR=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
ARG CUDA_VERSION_FULL=$CUDA_VERSION_MAJOR_MINOR.$CUDA_VERSION_PATCH.$CUDA_VERSION_BUILD
ARG CUDNN_VERSION=8.9

ENV TRT_VERSION=$TRT_VERSION_FULL+cuda$CUDA_VERSION_FULL
# RUN --mount=from=build-scripts,source=build-scripts,target=/nvidia/build-scripts \
#     unset TRTOSS_VERSION && \
#     /nvidia/build-scripts/installTRT.sh

RUN cd /opt
ENV TRT_TAG "release/$TRT_VERSION_MAJOR_MINOR"
RUN mkdir trt_oss_src && \
   cd trt_oss_src && \
   echo "$PWD Building TRT OSS..." && \
   git clone -b $TRT_TAG https://github.com/nvidia/TensorRT TensorRT && \
   cd TensorRT && \
   git submodule update --init --recursive && \
   mkdir -p build && cd build  && \
   #cmake .. -DGPU_ARCHS="52 53 60 61 70 75 80 86 90" -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu -DTRT_BIN_DIR=`pwd`/out -DCUDA_VERSION=$CUDA_VERSION_MAJOR_MINOR -DCUDNN_VERSION=$CUDNN_VERSION && \
   cmake .. \
    -DGPU_ARCHS="53;60;61;70;75;80;86;90" \
    -DCMAKE_CUDA_ARCHITECTURES="53;60;61;70;75;80;86;90" \
    -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu \
    -DTRT_BIN_DIR=`pwd`/out \
    -DCUDA_VERSION=$CUDA_VERSION_MAJOR_MINOR \
    -DCUDNN_VERSION=$CUDNN_VERSION && \
   make -j32  && \
   cp libnvinfer_plugin.so.8.6.* /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.$TRT_VERSION_MAJOR_MINOR_PATCH && \
   cp libnvinfer_plugin_static.a /usr/lib/x86_64-linux-gnu/libnvinfer_plugin_static.a && \
   cp libnvonnxparser.so.8.6.* /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.$TRT_VERSION_MAJOR_MINOR_PATCH && \
   cp libnvcaffeparser.so.8.6.* /usr/lib/x86_64-linux-gnu/libnvcaffeparser.so.$TRT_VERSION_MAJOR_MINOR_PATCH && \
   cp trtexec /usr/local/bin/ && \
   rm -rf trt_oss_src

# Only required if we're building from scratch. Not required if base image is from DLFW
# RUN pip install tensorrt==$TRT_VERSION_FULL && \
#     pip install uff --index-url https://pypi.ngc.nvidia.com && \
#     pip install graphsurgeon --index-url https://pypi.ngc.nvidia.com && \
#     pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com

WORKDIR /workspace/
# Create user that will run commands
ARG user_id=1000
ARG user_name=developer
ARG groups=developer:1000
ARG home=/home/developer
RUN echo "ALL   ALL = (ALL) NOPASSWD: ALL" > /etc/sudoers \
    && mkdir -p "$(dirname $home)"
WORKDIR /workspace/
