# Agent Onboarding

This guide is the fast path for coding agents and new maintainers. It favors
source-backed orientation over assumptions. For a full directory walk, read the
[Codebase tour](codebase_tour.md).

## Mental Model

TAO Deploy is a deployment runtime repository, not a training repository. Most
commands accept an exported model, build or load a TensorRT engine, run
inference or evaluation, and write outputs under a results directory.

The normal flow is:

```text
console command (setup.py)
  -> nvidia_tao_deploy/cv/common/entrypoint/entrypoint_hydra.py  (or entrypoint_proto.py)
  -> nvidia_tao_deploy/<cv|multimodal>/<model>/scripts/<subtask>.py   (fresh subprocess)
  -> Hydra spec validated against nvidia_tao_deploy/config/<model>/
  -> EngineBuilder / TRTInferencer / dataloader / metrics
```

The important layers are:

| Layer | What to inspect |
| :--- | :--- |
| Launcher | `scripts/envsetup.sh`, `runner/tao_deploy.py`, `docker/manifest.json` |
| Installed commands | `setup.py` console scripts |
| Model dispatch | `<model>/entrypoint/*.py` and `<model>/scripts/*.py` |
| Configuration | `nvidia_tao_deploy/config/<model>/` and `<model>/specs/` |
| TensorRT runtime | `nvidia_tao_deploy/engine/`, `nvidia_tao_deploy/inferencer/`, model-specific builders |
| Validation | `tests/`, `.pre-commit-config.yaml`, `.github/workflows/` |

## First Audit

Run these before editing:

```sh
pwd
git remote -v
git branch -vv
git -c filter.lfs.process= -c filter.lfs.required=false status --short --branch
find . -maxdepth 2 -type d
sed -n '1,220p' README.md
rg -n "console_scripts|entry_points" setup.py
ls .github/workflows/
sed -n '1,80p' .pre-commit-config.yaml
```

Use the LFS-disabled `git status` form when the local checkout cannot write to
`.git/lfs/tmp`. Initialize the submodule before running anything:

```sh
git submodule update --init
```

## Runtime Trace

When a task touches command behavior, gather the real flow with:

```sh
rg -n "ArgumentParser|docker/manifest|manifest.json|docker run|--gpus|--tag" runner scripts docker release
rg -n "hydra_runner|default_specs|get_subtasks|entrypoint|console_scripts" nvidia_tao_deploy setup.py
rg -n "config_name=" nvidia_tao_deploy/cv/<model>/scripts/
```

Then inspect the specific model package under `nvidia_tao_deploy/cv/` or
`nvidia_tao_deploy/multimodal/`.

## Common Agent Questions

| Question | Where to look |
| :--- | :--- |
| What command invokes this package? | `setup.py` `console_scripts` |
| What subtasks exist? | `<model>/scripts/*.py` and [Supported commands](supported_commands.md) |
| Is this a Hydra or proto family? | Hydra families have `nvidia_tao_deploy/config/<model>/`; proto families have `<model>/proto/` |
| What configuration fields are valid? | `nvidia_tao_deploy/config/<model>/default_config.py` |
| Which specification template does a subtask load? | The script's `hydra_runner(config_name=...)`, not the script filename |
| Where is the engine built? | `<model>/engine_builder.py` if present, else a sibling family's (for example, `dino` uses the `deformable_detr` builder), else `engine/builder.py` |
| Where does inference run? | `<model>/inferencer.py` or `inferencer/trt_inferencer.py` |
| How is an `.etlt` file decoded? | `nvidia_tao_deploy/utils/decoding.py::decode_model` |
| How do I add a new backend? | [Deploy backend integration](deploy_backend_integration.md) |
| How do I run in containers? | [Container power users](container_power_users.md) |

## Dirty Worktree Safety

Treat untracked files and unrelated edits as user-owned. Check status before and
after edits, and do not remove local files unless the task explicitly requires
it.

Generated documentation has one source of truth:

```sh
python tools/update_docs_supported_commands.py
python tools/update_docs_supported_commands.py --check
```

If the check fails, regenerate the file rather than hand-editing the generated
block.

## Targeted Checks

For documentation-only changes:

```sh
python tools/update_docs_supported_commands.py --check
python -m py_compile tools/update_docs_supported_commands.py
git diff --check -- README.md docs/*.md docs/assets/*.svg tools/*.py .pre-commit-config.yaml
```

For code changes, run the same static checks CI runs (`pre-commit run` on the
changed files; refer to [Testing and debugging](testing_and_debugging.md)),
then the nearest pytest suite. Call out GPU, Docker, TensorRT, private
checkpoint, and dataset-heavy tests explicitly when they fall outside the
change's blast radius.

All commits need a DCO sign-off (`git commit -s`); CI enforces it.
