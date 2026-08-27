# TAO Deploy Source Documentation

This directory documents the TAO Deploy repository from source. It is written for
contributors, maintainers, coding agents, and power users who need to trace a
command from its CLI entry point to TensorRT runtime behavior.

## Start Here

| Goal | Guide |
| :--- | :--- |
| Learn the repository structure and module layout | [Codebase tour](codebase_tour.md) |
| Build a mental model before editing | [Agent onboarding](agent_onboarding.md) |
| Trace command, config, and runtime flow | [Architecture](architecture.md) |
| Make common source or container changes | [Development workflows](development_workflows.md) |
| Choose and run validation | [Testing and debugging](testing_and_debugging.md) |
| Run or debug the prebuilt development container | [Container power users](container_power_users.md) |
| Add or update a model deploy backend | [Deploy backend integration](deploy_backend_integration.md) |
| Look up installed model commands | [Supported commands](supported_commands.md) |

## Repository Map

| Area | Source of truth |
| :--- | :--- |
| Python package and command inventory | `setup.py` |
| Development container launcher | `runner/tao_deploy.py` and `scripts/envsetup.sh` |
| Base image registry and digests | `docker/manifest.json` |
| Base image build logic | `docker/build.sh` |
| Release image build logic | `release/docker/deploy.sh` and `release/docker/Dockerfile.release` |
| Package version metadata | `release/python/version.py` |
| Model deployment implementations | `nvidia_tao_deploy/cv/` and `nvidia_tao_deploy/multimodal/` |
| Structured config dataclasses | `nvidia_tao_deploy/config/` |
| Shared TensorRT runtime code | `nvidia_tao_deploy/engine/` and `nvidia_tao_deploy/inferencer/` |
| Shared datasets, metrics, and utilities | `nvidia_tao_deploy/dataloader/`, `nvidia_tao_deploy/metrics/`, and `nvidia_tao_deploy/utils/` |
| Static checks and CI | `.pre-commit-config.yaml` and `.github/workflows/` |

## Diagrams

Architecture and workflow diagrams are checked in as SVG files under
`docs/assets/` and embedded with normal Markdown image links. The SVG files are
the canonical editable sources.

| Diagram | Source |
| :--- | :--- |
| Command runtime flow | [assets/source_runtime_flow.svg](assets/source_runtime_flow.svg) |
| Configuration flow | [assets/config_flow.svg](assets/config_flow.svg) |
| Package module map | [assets/module_map.svg](assets/module_map.svg) |
| Container launch flow | [assets/container_flow.svg](assets/container_flow.svg) |

## Documentation Maintenance

`docs/supported_commands.md` is generated from `setup.py` and each command's
`scripts/` package. Update it with:

```sh
python tools/update_docs_supported_commands.py
```

Check for drift with:

```sh
python tools/update_docs_supported_commands.py --check
```

The pre-commit hook regenerates the file on commit, and the GitHub Actions
`static-tests` workflow runs the same hooks, so command documentation stays
aligned with source. Install the hooks
once per clone:

```sh
pip install pre-commit
pre-commit install
```
