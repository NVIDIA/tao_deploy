# Testing and Debugging

Use this guide to choose targeted checks and diagnose common TAO Deploy
development failures.

## Static Checks

CI runs static checks on pull requests through GitHub Actions
(`.github/workflows/static-tests.yml`), which executes the repository's pre-commit
hooks against the changed files:

* SPDX license headers (`.github/hooks/check_license_header.py`)
* `pylint` (`.pylintrc`), `pydocstyle`, and `flake8`, scoped to
  `nvidia_tao_deploy/`
* The generated-documentation hook (`tools/update_docs_supported_commands.py`),
  which regenerates `docs/supported_commands.md` and fails on drift

Reproduce locally:

```sh
pip install pre-commit
pre-commit install
pre-commit run --from-ref origin/main --to-ref HEAD
```

Separate workflows enforce DCO sign-off on every commit (`dco.yml`), secret
scanning (`secret-scan.yml`), and PR title format (`pr-title.yml`).
Functional (GPU) tests run through the externally triggered
`blossom-ci.yml` workflow, not on every push.

## Test Environment Setup

The base development image has pytest and the runtime dependencies
(`docker/requirements.txt`, `requirements-dev.txt`), but **not** the TAO
packages themselves. Set up once per container:

```sh
# On the host
git submodule update --init          # tao-core/ is empty otherwise
source scripts/envsetup.sh
tao_deploy --gpus all -- bash

# Inside the container (repository is mounted at /workspace/tao-deploy)
pip install /workspace/tao-deploy/tao-core/.   # provides nvidia_tao_core
cd /workspace/tao-deploy && python setup.py develop   # console commands + package
pytest --color=yes -v tests/core/test_dual_logging.py  # smoke check
```

`nvidia_tao_core` is not in the requirements files, and its import in
`status_logging.py` is unguarded, so nearly every test and command fails until
the submodule is installed. Engine and inferencer tests additionally need a
GPU, TensorRT, and model artifacts; several suites expect datasets mounted
from private paths.

## Test Map

| Area | Examples |
| :--- | :--- |
| Core utilities | `tests/core/test_decrypt.py`, `tests/core/test_dual_logging.py`, `tests/core/test_common_config.py` |
| Schema drift vs tao-pytorch | `tests/core/test_backbone_schema_drift.py` |
| TensorRT engine flows | `tests/<model>/test_engine.py`, `tests/<model>/test_engine_builder.py` |
| Dataloaders | `tests/<model>/test_dataloader.py` |
| Inference helpers | `tests/clip/test_inferencer.py` and other model inferencer tests |
| Metrics and evaluation | `tests/clip/test_evaluation.py`, model `evaluate` tests |
| Full modern template | `tests/clip/`, `tests/video_clip/` (config, dataloader, engine builder, entrypoint, evaluation, inferencer) |

The repository has no `conftest.py` or pytest configuration file, so the
per-family markers used in test files are unregistered. Prefer path-based
selection:

```sh
pytest --color=yes -v tests/core/
pytest --color=yes -v tests/<model>/
```

Many engine tests require GPUs, TensorRT, model artifacts, and datasets mounted
from private paths. Prefer targeted tests for small source changes and state
clearly when you did not run the heavier checks.

## Documentation Checks

For documentation-only changes:

```sh
python tools/update_docs_supported_commands.py --check
python -m py_compile tools/update_docs_supported_commands.py
git diff --check -- README.md docs/*.md docs/assets/*.svg tools/*.py .pre-commit-config.yaml
```

## Common Failures

| Symptom | Likely source | Check |
| :--- | :--- | :--- |
| `git status` fails on LFS clean filter | Read-only `.git/lfs/tmp` | Use `git -c filter.lfs.process= -c filter.lfs.required=false status --short --branch`. |
| Import errors for `nvidia_tao_core` | Uninitialized `tao-core/` submodule | `git submodule update --init`; in containers, `envsetup.sh` adds `/workspace/tao-deploy/tao-core` to `PYTHONPATH`. This import is unguarded in `status_logging.py`, so almost every command fails without it. |
| Base image pull fails | NGC login or network access | `docker login nvcr.io` and inspect `docker/manifest.json`. |
| TensorRT or CUDA errors | Host and container mismatch or missing GPU | Check `--gpus`, driver, CUDA, TensorRT, and `nvidia-container-toolkit`. |
| Generated documentation check fails | `setup.py` or a `scripts/` package changed | Run `python tools/update_docs_supported_commands.py`. |
| `default_specs` or a subtask cannot find its template | Specification template name diverges from the subtask name | Read the script's `hydra_runner(config_name=...)`; refer to the divergence table in [Architecture](architecture.md). |
| Breakpoints in a `scripts/*.py` never trigger | The entrypoint launches scripts as a fresh subprocess | Run the script module directly with `--config-path`/`--config-name`. |
| UFF model paths raise `NotImplementedError` | TensorRT >= 9 removed UFF support | Legacy families are ONNX/ETLT-ONNX only on current base images. |
| Engine build OOMs or workspace looks wrong | `workspace_size` unit mismatch (megabytes in some specifications, gigabytes in `EngineBuilder`) | Check for `// 1024` conversions in the family's `gen_trt_engine.py`. |

## Debugging Patterns

Trace command launch:

```sh
rg -n "<command>=" setup.py
sed -n '1,260p' nvidia_tao_deploy/cv/common/entrypoint/entrypoint_hydra.py
```

Trace config:

```sh
find nvidia_tao_deploy/config/<model> -maxdepth 1 -type f | sort
find nvidia_tao_deploy/cv/<model>/specs -maxdepth 2 -type f | sort
rg -n "config_name=" nvidia_tao_deploy/cv/<model>/scripts/
```

Trace engine building:

```sh
rg -n "EngineBuilder|create_network|create_engine" nvidia_tao_deploy/cv/<model> nvidia_tao_deploy/engine
```

Trace inference:

```sh
rg -n "TRTInferencer|allocate_buffers|do_inference|infer\(" nvidia_tao_deploy/cv/<model> nvidia_tao_deploy/inferencer
```

Run a script in-process (debuggable, bypasses the subprocess launch):

```sh
python nvidia_tao_deploy/cv/<model>/scripts/<subtask>.py \
  --config-path /abs/path/to/spec/dir --config-name <spec_name>
```
