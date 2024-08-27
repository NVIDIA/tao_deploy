#!/usr/bin/env bash

set -eo pipefail

# Setting up the required env variables.
ENV_SET=$NV_TAO_DEPLOY_TOP/scripts/envsetup.sh

REGISTRY="nvcr.io"
TENSORRT_VERSION="8.5.3.1"
CUDA_VERSION=12.0
TAO_VERSION="5.2.0"
REPOSITORY="nvstaging/tao/tao-toolkit-deploy"
BUILD_ID="01"
tag="v${TAO_VERSION}-trt${TENSORRT_VERSION}-${BUILD_ID}-dev-cuda${CUDA_VERSION}"

# Setting up the environment.
source $ENV_SET

# Build parameters.
BUILD_DOCKER="0"
BUILD_WHEEL="0"
PUSH_DOCKER="0"
FORCE="0"


# Parse command line.
while [[ $# -gt 0 ]]
    do
    key="$1"

    case $key in
        -b|--build)
        BUILD_DOCKER="1"
        RUN_DOCKER="0"
        shift # past argument
        ;;
        -w|--wheel)
        BUILD_WHEEL="1"
        RUN_DOCKER="0"
        shift # past argument
        ;;
        -p|--push)
        PUSH_DOCKER="1"
        shift # past argument
        ;;
        -f|--force)
        FORCE=1
        shift
        ;;
        -r|--run)
        RUN_DOCKER="1"
        BUILD_DOCKER="0"
        shift # past argument
        ;;
        --default)
        BUILD_DOCKER="0"
        RUN_DOCKER="1"
        FORCE="0"
        PUSH_DOCKER="0"
        shift # past argument
        ;;
        *)    # unknown option
        POSITIONAL+=("$1") # save it in an array for later
        shift # past argument
        ;;
    esac
done


if [ $BUILD_DOCKER = "1" ]; then
    echo "Building base docker ..."
    if [ $FORCE = "1" ]; then
        echo "Forcing docker build without cache ..."
        NO_CACHE="--no-cache"
    else
        NO_CACHE=""
    fi
    if [ $BUILD_WHEEL = "1" ]; then
        echo "Building source code wheel ..."
    #    tao_deploy -- make build
       tao_deploy -- python3 setup.py bdist_wheel
    else
        echo "Skipping wheel builds ..."
    fi
    
    docker build --pull -f $NV_TAO_DEPLOY_TOP/release/docker/Dockerfile.release -t $REGISTRY/$REPOSITORY:$tag $NO_CACHE --network=host $NV_TAO_DEPLOY_TOP/.

    if [ $PUSH_DOCKER = "1" ]; then
        echo "Pusing docker ..."
        docker push $REGISTRY/$REPOSITORY:$tag
    else
        echo "Skip pushing docker ..."
    fi

    if [ $BUILD_WHEEL = "1" ]; then
        echo "Cleaning wheels ..."
        tao_deploy -- make clean
    else
        echo "Skipping wheel cleaning ..."
    fi
elif [ $RUN_DOCKER ="1" ]; then
    echo "Running docker interactively..."
    docker run --gpus all -v /home/$USER/tlt-experiments:/workspace/tlt-experiments \
                          --net=host --shm-size=30g --ulimit memlock=-1 --ulimit stack=67108864 \
                          --rm -it $REGISTRY/$REPOSITORY:$tag /bin/bash
else
    echo "Usage: ./deploy.sh [--build] [--wheel] [--run] [--default]"
fi
