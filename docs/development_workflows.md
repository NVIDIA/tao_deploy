# Development Workflows

This guide gives concrete recipes for common source-code changes.

## Source Setup

Prepare a fresh checkout with:

```sh
git submodule update --init
source scripts/envsetup.sh
```

This sets `NV_TAO_DEPLOY_TOP`, defines the `tao_deploy` shell function, and
installs the repository's pre-commit hooks.

Build and install a wheel locally:

```sh
make build
make install
```

Build the L4T (Jetson) wheel:

```sh
make build_l4t
```

Clean generated package artifacts:

```sh
make clean
```

## Run Inside the Base Container

Use `--` to split launcher arguments from the command that runs inside the
container:

```sh
tao_deploy --gpus all -- python3 -m pytest tests/core/test_dual_logging.py
```

Useful launcher options are documented in
[Container power users](container_power_users.md).

## Trace a Command to Code

Use this when a task mentions a command such as `dino gen_trt_engine`.

```sh
rg -n "dino=" setup.py
sed -n '1,80p' nvidia_tao_deploy/cv/dino/entrypoint/dino.py
sed -n '1,200p' nvidia_tao_deploy/cv/dino/scripts/gen_trt_engine.py
rg -n "config_name=" nvidia_tao_deploy/cv/dino/scripts/
```

Then inspect the configuration, specifications, and tests:

```sh
find nvidia_tao_deploy/config/dino -maxdepth 1 -type f | sort
find nvidia_tao_deploy/cv/dino/specs -maxdepth 1 -type f | sort
find tests -path '*dino*' -type f | sort
```

Not every family has a test directory (there is no `tests/dino/`, for
example). An empty result does not mean you searched wrong.

Remember that some families borrow runtime classes from a sibling — if there is
no `engine_builder.py` or `inferencer.py` in the package, search for the import
in the scripts.

## Add a Configuration Field

For a Hydra-style family:

1. Add the dataclass field under `nvidia_tao_deploy/config/<model>/`, using the
   factories from `nvidia_tao_deploy/config/utils/types.py` (`STR_FIELD`,
   `INT_FIELD`, ...), including a `description`.
2. Mirror the field in the relevant specification template under
   `nvidia_tao_deploy/cv/<model>/specs/` if the default must be visible to users.
3. Read the field from `cfg` in the consuming script or builder.
4. If the field mirrors a training-side configuration, confirm the tao-pytorch schema
   uses the same name and semantics.
5. Add or update a configuration test (refer to
   `tests/core/test_common_config.py` and `tests/clip/test_config.py` for
   patterns).

```sh
rg -n "class .*Config|DATACLASS_FIELD|STR_FIELD|INT_FIELD" nvidia_tao_deploy/config/<model>
rg -n "<field_name>" nvidia_tao_deploy/cv/<model> tests
```

## Update a Model Deploy Flow

For a Hydra-style model:

1. Update the dataclasses in `nvidia_tao_deploy/config/<model>/`.
2. Update the matching templates in `nvidia_tao_deploy/cv/<model>/specs/` or
   `nvidia_tao_deploy/multimodal/<model>/specs/`.
3. Update `scripts/gen_trt_engine.py`, `scripts/inference.py`, or
   `scripts/evaluate.py`.
4. Update any model-specific builder, inferencer, dataloader, metric, or
   post-processing code.
5. Add or update focused tests under `tests/<model>/`.

For a proto-style model, follow the existing `build_command_line_parser()` and
proto loader pattern in that model package. Regenerate `*_pb2.py` files with
`internal/generate_pb2.sh <model>` after editing `.proto` files.

## Add a New Command

1. Add the model package under `nvidia_tao_deploy/cv/` or
   `nvidia_tao_deploy/multimodal/`.
2. Add an `entrypoint/` wrapper and `scripts/` package.
3. Add a `console_scripts` entry in `setup.py`.
4. Add config dataclasses under `nvidia_tao_deploy/config/` so the command
   supports schema validation and default specification generation.
5. Regenerate supported-command docs.
6. Add tests under `tests/<model>/`. The `tests/clip/` and `tests/video_clip/`
   suites are the most complete templates (configuration, dataloader, engine
   builder, entry point, evaluation, and inferencer).

Refer to [Deploy backend integration](deploy_backend_integration.md) for the
source-backed checklist.

## Update Generated Command Documentation

When `setup.py` console scripts or model `scripts/` packages change:

```sh
python tools/update_docs_supported_commands.py
python tools/update_docs_supported_commands.py --check
```

The generated file is `docs/supported_commands.md`. A pre-commit hook
regenerates it on commit and fails the commit if it changed, so drift never
lands.

## Update the Base Image

The base image source is under `docker/`.

```sh
bash docker/build.sh --build --x86
bash docker/build.sh --build --arm
bash docker/build.sh --build --l4t
```

Push and record the new digest only after validation:

```sh
bash docker/build.sh --build --x86 --push
```

Then update the new digest in both places that pin it: the matching platform
entry in `docker/manifest.json` (read by the `tao_deploy` launcher) and the
default `X86_DIGEST`/`ARM64_DIGEST` build args at the top of
`release/docker/Dockerfile.release`. Find any stragglers with
`grep -rl <old-digest> .`. The Jetson stack (`docker/Dockerfile.l4t`) is
tracked separately.

## Build a Release Image

The release image installs a wheel built from this repository plus the
tao-core wheel.

```sh
source scripts/envsetup.sh
cd release/docker
./deploy.sh --build --wheel
```

Release image tags are assembled in `release/docker/deploy.sh`; package version
metadata comes from `release/python/version.py`.
