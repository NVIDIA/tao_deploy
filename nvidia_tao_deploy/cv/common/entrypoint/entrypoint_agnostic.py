# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TAO Deploy model-agnostic entrypoint that parses spec YAML for model name"""

import argparse
import importlib
import yaml

from nvidia_tao_deploy.cv.common.entrypoint.entrypoint_hydra import get_subtasks, launch


def get_subtask_list(model):
    """Return the list of subtasks by inspecting the scripts package."""
    try:
        scripts = importlib.import_module(f"nvidia_tao_deploy.cv.{model}.scripts")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"{model} is not a supported TAO model") from e
    return get_subtasks(scripts)


def main():
    """Construct a model-entrypoint launch() call from a specfile that populates the model_name field.

    For now, this only works for PyTorch networks.
    """
    parser = argparse.ArgumentParser(
        "TAO Deploy",
        add_help=True,
        description="Train Adapt Optimize Deploy model-agnostic entrypoint"
    )
    parser.add_argument(
        "subtask",
        default="gen_trt_engine",
        help="Subtask for a given task/model.",
    )
    parser.add_argument(
        "-e",
        "--experiment_spec_file",
        help="Path to the experiment spec file.",
        required=True,
        default=None,
    )

    # Original flow got subtasks from model name, used that to limit choices in argparser
    # Now, since we only get model name *after* argparser, we have to limit choices after

    # Parse the arguments.
    args, unknown_args = parser.parse_known_args()

    # Getting the model name from the specfile
    spec_file = vars(args)["experiment_spec_file"]
    with open(spec_file, 'r', encoding='utf-8') as f:
        spec = yaml.safe_load(f)
        if spec and 'model_name' in spec:
            model = spec['model_name']
        else:
            raise KeyError(f"'model' field not found in {spec_file}. If you wish to use the model-agnostic entrypoint, please populate this value. Otherwise, use the direct entrypoint for the model you desire.")

    # Obtain the list of substasks
    subtasks = get_subtask_list(model)
    subtask = vars(args)["subtask"]
    if subtask not in subtasks:
        raise KeyError(f"{subtask} is not a valid subtask for {model}. Please choose from {list(subtasks.keys())}")

    # Parse the arguments and launch the subtask.
    launch(vars(args), unknown_args, subtasks, network=model)


if __name__ == '__main__':
    main()
