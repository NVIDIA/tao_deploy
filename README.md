# TAO Toolkit - Deploy Backend

<!-- vscode-markdown-toc -->
* [Overview](#Overview)
* [Getting Started](#GettingStarted)
	* [Requirements](#Requirements)
		* [Hardware Requirements](#HardwareRequirements)
		* [Software Requirements](#SoftwareRequirements)
	* [Instantiating the development container](#Instantiatingthedevelopmentcontainer)
	* [Updating the base docker](#Updatingthebasedocker)
		* [Build base docker](#Buildbasedocker)
		* [Test the newly built base docker](#Testthenewlybuiltbasedocker)
		* [Update the new docker](#Updatethenewdocker)
* [Building a release container](#Buildingareleasecontainer)
* [Running TAO Deploy on Jetson devices](#JetsonDevices)
* [Contribution Guidelines](#ContributionGuidelines)
* [License](#License)

<!-- vscode-markdown-toc-config
	numbering=false
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

## <a name='Overview'></a>Overview

TAO Toolkit is a Python package hosted on the NVIDIA Python Package Index. It interacts with lower-level TAO dockers available from the NVIDIA GPU Accelerated Container Registry (NGC). The TAO containers come pre-installed with all dependencies required for training. The output of the TAO workflow is a trained model that can be deployed for inference on NVIDIA devices using DeepStream, TensorRT and Triton.

This repository contains the required implementation for all the deep learning components and networks using the TensorRT backend. These routines are packaged as part of the TAO Toolkit TensorRT container in the Toolkit package. The source code here is compatible with TensorRT version <= 8.5.3

## <a name='GettingStarted'></a>Getting Started

As soon as the repository is cloned, run the `envsetup.sh` file to check
if the build environment has the necessary dependencies, and the required
environment variables are set.

```sh
source scripts/envsetup.sh
```

We recommend adding this command to your local `~/.bashrc` file, so that every new terminal instance receives this.

### <a name='Requirements'></a>Requirements

#### <a name='HardwareRequirements'></a>Hardware Requirements

##### Minimum system configuration

* 8 GB system RAM
* 4 GB of GPU RAM
* 8 core CPU
* 1 NVIDIA GPU
* 100 GB of SSD space

##### Recommended system configuration

* 32 GB system RAM
* 32 GB of GPU RAM
* 8 core CPU
* 1 NVIDIA GPU
* 100 GB of SSD space

#### <a name='SoftwareRequirements'></a>Software Requirements

| **Software**                     | **Version** |
| :--- | :--- |
| Ubuntu LTS                       | >=18.04     |
| python                           | >=3.8.x     |
| docker-ce                        | >19.03.5    |
| docker-API                       | 1.40        |
| `nvidia-container-toolkit`       | >1.3.0-1    |
| nvidia-container-runtime         | 3.4.0-1     |
| nvidia-docker2                   | 2.5.0-1     |
| nvidia-driver                    | >525.85     |
| python-pip                       | >21.06      |

### <a name='Instantiatingthedevelopmentcontainer'></a>Instantiating the development container

In order to maintain a uniform development environment across all users, TAO Toolkit provides a base environment docker that has been built and uploaded to NGC for the developers. For instantiating the docker, simply run the `tao_deploy` CLI. The usage for the command line launcher is mentioned below.

```sh
usage: tao_deploy [-h] [--gpus GPUS] [--volume VOLUME] [--env ENV] [--mounts_file MOUNTS_FILE] [--shm_size SHM_SIZE] [--run_as_user] [--ulimit ULIMIT] [--port PORT]

Tool to run the TAO Toolkit Deploy container.

optional arguments:
  -h, --help            show this help message and exit
  --gpus GPUS           Comma separated GPU indices to be exposed to the docker.
  --volume VOLUME       Volumes to bind.
  --env ENV             Environment variables to bind.
  --mounts_file MOUNTS_FILE
                        Path to the mounts file.
  --shm_size SHM_SIZE   Shared memory size for docker
  --run_as_user         Flag to run as user
  --ulimit ULIMIT       Docker ulimits for the host machine.
  --port PORT           Port mapping (e.g. 8889:8889).

```

A sample command to instantiate an interactive session in the base development docker is mentioned below.

```sh
tao_deploy --gpus all --volume /path/to/data/on/host:/path/to/data/on/container --volume /path/to/results/on/host:/path/to/results/in/container
```

### <a name='Updatingthebasedocker'></a>Updating the base docker

There will be situations where developers would be required to update the third party dependencies to newer versions, or upgrade CUDA, etc. In such a case, please follow the steps below:

#### <a name='Buildbasedocker'></a>Build base docker

The base dev docker is defined in `$NV_TAO_DEPLOY_TOP/docker/Dockerfile`. The python packages required for the TAO dev is defined in `$NV_TAO_DEPLOY_TOP/docker/requirements-pip.txt` and the third party apt packages are defined in `$NV_TAO_DEPLOY_TOP/docker/requirements-apt.txt`. Once you have made the required change, please update the base docker using the build script in the same directory.

```sh
cd $NV_TAO_DEPLOY_TOP/docker
./build.sh --build
```

#### <a name='Testthenewlybuiltbasedocker'></a>Test the newly built base docker

Developers may tests their new docker by using the `tao_deploy` command.

```sh
tao_deploy -- script args
```

#### <a name='Updatethenewdocker'></a>Update the new docker

Once you are sufficiently confident about the newly built base docker, please do the following

1. Push the newly built base docker to the registry

    ```sh
    bash $NV_TAO_DEPLOY_TOP/docker/build.sh --build --push
    ```

2. The above step produces a digest file associated with the docker. This is a unique identifier for the docker. So please note this, and update all references of the old digest in the repository with the new digest. You may find the old digest in the `$NV_TAO_DEPLOY_TOP/docker/manifest.json`.

Push you final updated changes to the repository so that other developers can leverage and sync with the new dev environment.

Please note that if for some reason you would like to force build the docker without using a cache from the previous docker, you may do so by using the `--force` option.

```sh
bash $NV_TAO_DEPLOY_TOP/docker/build.sh --build --push --force
```

## <a name='Buildingareleasecontainer'></a>Building a release container

The TAO container is built on top of the TAO Deploy base dev container, by building a python wheel for the `nvidia_tao_deploy` module in this repository and installing the wheel in the Dockerfile defined in `release/docker/Dockerfile`. The whole build process is captured in a single shell script which may be run as follows:

```sh
source scripts/envsetup.sh
cd $NV_TAO_DEPLOY_TOP/release/docker
./deploy.sh --build --wheel
```

In order to build a new docker, please edit the `deploy.sh` file in `$NV_TAO_DEPLOY_TOP/release/docker` to update the patch version and re-run the steps above.

## <a name='JetsonDevices'></a>Running TAO Deploy on Jetson devices

The released TAO Deploy container is based on `x86` platform so it will not work on `aarch64` platforms. In order to run TAO Deploy on Jetson devices, please instantiate [TensorRT-L4T docker hosted on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorrt) using below command. Note that this TensorRT version may not match with the version from the released TAO container.

```sh
sudo docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix nvcr.io/nvidia/l4t-tensorrt:xx
```

After the container is instantiated, run below steps to install the TAO Deploy wheel and corresponding dependencies on your system.

```sh
apt install libopenmpi-dev
pip install nvidia_tao_deploy==5.0.0.423.dev0
pip install https://files.pythonhosted.org/packages/f7/7a/ac2e37588fe552b49d8807215b7de224eef60a495391fdacc5fa13732d11/nvidia_eff_tao_encryption-0.1.7-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install https://files.pythonhosted.org/packages/0d/05/6caf40aefc7ac44708b2dcd5403870181acc1ecdd93fa822370d10cc49f3/nvidia_eff-0.6.2-py38-none-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

## <a name='ContributionGuidelines'></a>Contribution Guidelines
TAO Toolkit Deploy backend is not accepting contributions as part of the TAO 5.0 release, but will be open in the future.

## <a name='License'></a>License
This project is licensed under the [Apache-2.0](./LICENSE) License.
