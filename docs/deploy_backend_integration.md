# Deploy Backend Integration

Use this checklist when adding or substantially changing a TAO Deploy model
backend. It is calibrated against real in-repo patterns rather than a single
idealized layout.

## Source Exemplars

| Exemplar | Why it matters |
| :--- | :--- |
| `nvidia_tao_deploy/cv/dino` | Modern CV Hydra entrypoint, `gen_trt_engine`/`inference`/`evaluate` scripts, YAML specs, config dataclasses, shared D-DETR builder and inferencer reuse. |
| `nvidia_tao_deploy/cv/classification_tf1` | Legacy proto and argparse flow through `entrypoint_proto.launch_job`, script-local parsers, proto loaders, and a model-specific engine builder. |
| `nvidia_tao_deploy/multimodal/clip` | Multimodal Hydra package outside `cv/`, CLIP config namespace, combined or separate encoder handling, tokenizer/config artifact copying, and retrieval metrics. |

Also check aliases in `setup.py`: `dssd` dispatches to the SSD entrypoint, and
`yolo_v4_tiny` dispatches to the YOLOv4 entrypoint.

## Choose The Command Shape

Prefer the existing Hydra-style pattern for new work:

```text
nvidia_tao_deploy/DOMAIN/MODEL_NAME/
  entrypoint/MODEL_NAME.py
  scripts/gen_trt_engine.py
  scripts/inference.py
  scripts/evaluate.py
  specs/*.yaml
```

Use `nvidia_tao_deploy/cv/` for CV models and
`nvidia_tao_deploy/multimodal/` for multimodal models. Match the existing
package domain before adding a new top-level family.

Keep proto-style wiring only when maintaining a legacy proto backend.

## Add Configuration

Hydra-style commands need a config package under `nvidia_tao_deploy/config/`.
For example:

```text
nvidia_tao_deploy/config/dino/default_config.py
nvidia_tao_deploy/config/multimodal/clip/default_config.py
```

`default_specs` support depends on both a config package and a deploy
implementation with an `entrypoint/` directory. The default-spec discovery code
lives in `nvidia_tao_deploy/cv/common/spec_utils/default_specs.py`.

Name spec templates from the script's `hydra_runner(config_name=...)`, not from
the subtask name. DINO's `inference` script uses `config_name="infer"`, so its
template is `specs/infer.yaml`.

## Add Entrypoint And Scripts

For Hydra-style commands:

1. Import the model `scripts` package.
2. Build subtasks with `get_subtasks(scripts)`.
3. Parse the subtask and optional `-e/--experiment_spec_file` with
   `command_line_parser()`.
4. Call `launch(vars(args), unknown_args, subtasks, network="MODEL_NAME")`.

For proto-style commands, follow `classification_tf1`: the entrypoint calls
`launch_job()`, and each script exposes `build_command_line_parser()`.

Add the installed command in `setup.py` under `console_scripts`, then regenerate
the command documentation:

```sh
python tools/update_docs_supported_commands.py
```

## Build TensorRT Runtime Code

A typical `gen_trt_engine.py` script:

1. Uses `@hydra_runner(..., schema=ExperimentConfig)`.
2. Uses `@monitor_status(name="MODEL_NAME", mode="gen_trt_engine")`.
3. Calls `decode_model()` for encrypted or plain model inputs.
4. Calls `initialize_gen_trt_engine_experiment()` when the model follows the
   shared config structure.
5. Creates an `EngineBuilder` subclass when generic parsing is not enough.
6. Calls `create_network()` and `create_engine()`.

A typical inference or evaluation script creates a model-specific
`TRTInferencer`, builds batches from model config, applies post-processing, and
writes results under `results_dir`.

## Handle Known Variants

| Surface | Source-backed rule |
| :--- | :--- |
| Domain | `cv` and `multimodal` both exist. Do not assume every model lives under `cv`. |
| Specs | YAML names come from `hydra_runner(config_name=...)`; proto models use `.txt` specs and parsers. |
| Builders | Some models use generic `EngineBuilder`, some subclass it, and some reuse another model's builder. |
| Inferencers | Most subclass `TRTInferencer`; CLIP also supports separate encoder inferencers. |
| Calibration | INT8 flows may use image directories, tensorfiles, calibration caches, or QDQ strongly typed ONNX handling. |
| Default specs | Requires config discovery plus an entrypoint package. |
| Aliases | Installed command names can point at another model implementation. |

## Tests And Validation

Add focused tests under `tests/MODEL_NAME/`. Match existing local patterns:

```sh
pre-commit run --from-ref origin/main --to-ref HEAD
pytest --color=yes -v tests/MODEL_NAME
```

The modern test template is `tests/clip/` / `tests/video_clip/` (config,
dataloader, engine builder, entrypoint, evaluation, and inferencer tests).

For GPU, TensorRT, model-artifact, or private-dataset tests that cannot be run
locally, state exactly which checks were skipped and why.

## Final Checklist

Before review:

| Check | Done |
| :--- | :--- |
| Console script added or updated in `setup.py`. | |
| Entrypoint imports the correct scripts package. | |
| Script names, subtask names, and YAML `config_name` values are consistent. | |
| Config dataclasses and spec templates cover engine generation, inference, and evaluation as applicable. | |
| Builder, inferencer, dataloader, post-processing, and metrics are source-backed and tested. | |
| `default_specs` works or the command is documented as legacy/proto-only. | |
| `docs/supported_commands.md` regenerated. | |
| Static and targeted functional tests selected and run. | |
