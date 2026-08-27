# TAO Toolkit Deploy Backend

TAO Deploy contains the TensorRT deployment backend for TAO Toolkit models. It
packages model-specific engine generation, inference, evaluation, calibration,
and container tooling as the `nvidia-tao-deploy` Python package.

The source-oriented documentation for contributors, maintainers, and coding
agents starts in [docs/index.md](docs/index.md).

## Quick Start

Set up the source checkout:

```sh
source scripts/envsetup.sh
```

Start the development container:

```sh
tao_deploy --gpus all --volume /host/data:/workspace/data
```

Run a command inside the development container:

```sh
tao_deploy -- python3 -m pytest tests/core/test_dual_logging.py
```

Build the base development image for the host platform:

```sh
bash docker/build.sh --build --x86
```

Build a release container and wheel:

```sh
bash release/docker/deploy.sh --build --wheel
```

## Documentation Map

| Start here | Use it for |
| :--- | :--- |
| [Source docs hub](docs/index.md) | Choosing the right guide for repository work. |
| [Agent onboarding](docs/agent_onboarding.md) | First-pass audit commands, repo mental model, and safety checks. |
| [Architecture](docs/architecture.md) | Command dispatch, configuration flow, TensorRT runtime flow, and extension points. |
| [Development workflows](docs/development_workflows.md) | Common recipes for source, Docker, release, and generated-doc changes. |
| [Testing and debugging](docs/testing_and_debugging.md) | Static checks, functional tests, targeted pytest commands, and common failures. |
| [Container power users](docs/container_power_users.md) | `tao_deploy`, direct Docker equivalents, manifests, mounts, GPUs, and Jetson notes. |
| [Deploy backend integration](docs/deploy_backend_integration.md) | Source-backed checklist for adding or updating a model deploy backend. |
| [Supported commands](docs/supported_commands.md) | Generated console-command inventory from `setup.py`. |

## Source Map

| Path | Responsibility |
| :--- | :--- |
| `setup.py` | Package metadata and installed model command entrypoints. |
| `runner/tao_deploy.py` | Local wrapper that launches the development container. |
| `docker/` | Base development image Dockerfiles, requirements, build script, and image manifest. |
| `release/docker/` | Release container build, obfuscation, and wheel install flow. |
| `nvidia_tao_deploy/` | TensorRT engine builders, inferencers, task scripts, configs, metrics, and utilities. |
| `tests/` | Model-specific and core pytest coverage. |
| `ci/` | Static and functional test launchers used locally and in GitLab CI. |

## License

This project is licensed under the [Apache-2.0](LICENSE) License.

<!-- blossom CI smoke test: verify 7.1.0 OSS migration tests pass — DO NOT MERGE -->
