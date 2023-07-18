# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""TAO Deploy (TF2) command line wrapper to invoke CLI scripts."""

import importlib
import os
import pkgutil
import shlex
import subprocess
import sys
from time import time
import pycuda.driver as cuda

from nvidia_tao_deploy.cv.common.telemetry.nvml_utils import get_device_details
from nvidia_tao_deploy.cv.common.telemetry.telemetry import send_telemetry_data


def get_subtasks(package):
    """Get supported subtasks for a given task.

    This function lists out the python tasks in a folder.

    Returns:
        subtasks (dict): Dictionary of files.
    """
    module_path = package.__path__
    modules = {}

    # Collect modules dynamically.
    for _, task, is_package in pkgutil.walk_packages(module_path):
        if is_package:
            continue
        module_name = package.__name__ + '.' + task
        module_details = {
            "module_name": module_name,
            "runner_path": os.path.abspath(importlib.import_module(module_name).__file__),
        }
        modules[task] = module_details
    return modules


def check_valid_gpus(gpu_id):
    """Check if IDs is valid.

    This function scans the machine using the nvidia-smi routine to find the
    number of GPU's and matches the id's accordingly.

    Once validated, it finally also sets the CUDA_VISIBLE_DEVICES env variable.

    Args:
        gpu_id (int): GPU index used by the user.

    Returns:
        No explicit returns
    """
    cuda.init()
    num_gpus_available = cuda.Device.count()

    assert gpu_id >= 0, (
        "GPU id cannot be negative."
    )
    assert gpu_id < num_gpus_available, (
        "Checking for valid GPU ids and num_gpus."
    )
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{gpu_id}"


def set_gpu_info_single_node(gpu_id):
    """Set gpu environment variable for single node."""
    check_valid_gpus(gpu_id)

    env_variable = ""
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if visible_devices is not None:
        env_variable = f" CUDA_VISIBLE_DEVICES={visible_devices}"

    return env_variable


def command_line_parser(parser, subtasks):
    """Build command line parser."""
    parser.add_argument(
        'subtask',
        default='gen_trt_engine',
        choices=subtasks.keys(),
        help="Subtask for a given task/model.",
    )
    parser.add_argument(
        "-k",
        "--key",
        help="User specific encoding key to load an .etlt model."
    )
    # Add standard TLT arguments.
    parser.add_argument(
        "-r",
        "--results_dir",
        help="Path to a folder where the experiment outputs should be written. (DEFAULT: ./)",
    )
    parser.add_argument(
        "-e",
        "--experiment_spec_file",
        help="Path to the experiment spec file.",
        required=True,
        default=None
    )
    parser.add_argument(
        '--gpu_index',
        type=int,
        default=0,
        help="The index of the GPU to be used.",
    )
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        default=None,
        help='Confidence threshold for inference.'
    )
    # Parse the arguments.
    return parser


def launch(parser,
           subtasks,
           override_results_dir="result_dir",
           override_threshold="evaluate.min_score_thresh",
           override_key="encryption_key",
           network="tao-deploy"):
    """Parse the command line and kick off the entrypoint.

    Args:

        parser (argparse.ArgumentParser): Parser object to define the command line args.
        subtasks (list): List of subtasks.
    """
    # Subtasks for a given model.
    parser = command_line_parser(parser, subtasks)

    cli_args = sys.argv[1:]
    args, unknown_args = parser.parse_known_args(cli_args)
    args = vars(args)

    scripts_args = ""
    assert args["experiment_spec_file"], (
        f"Experiment spec file needs to be provided for this task: {args['subtask']}"
    )
    if not os.path.exists(args["experiment_spec_file"]):
        raise FileNotFoundError(f"Experiment spec file doesn't exist at {args['experiment_spec_file']}")
    path, name = os.path.split(args["experiment_spec_file"])
    if path != "":
        scripts_args += f" --config-path {path}"
    scripts_args += f" --config-name {name}"

    if args['subtask'] in ["evaluate", "inference"]:
        if args['results_dir']:
            scripts_args += f" {override_results_dir}={args['results_dir']}"

    if args['subtask'] in ['inference']:
        if args['threshold'] and override_threshold:
            scripts_args += f" {override_threshold}={args['threshold']}"

    # Add encryption key.
    if args['subtask'] in ["gen_trt_engine"]:
        if args['key']:
            scripts_args += f" {override_key}={args['key']}"

    gpu_ids = args["gpu_index"]

    script = subtasks[args['subtask']]["runner_path"]

    unknown_args_string = " ".join(unknown_args)
    task_command = f"python {script} {scripts_args} {unknown_args_string}"
    print(task_command)
    env_variables = set_gpu_info_single_node(gpu_ids)

    run_command = f"bash -c \'{env_variables} {task_command}\'"
    process_passed = True
    start = time()
    try:
        subprocess.run(
            shlex.split(run_command),
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True
        )
    except (KeyboardInterrupt, SystemExit):
        print("Command was interrupted.")
    except subprocess.CalledProcessError as e:
        process_passed = False
        if e.output is not None:
            print(f"TAO Deploy task: {args['subtask']} failed with error:\n{e.output}")

    end = time()
    time_lapsed = end - start

    try:
        gpu_data = []
        for device in get_device_details():
            gpu_data.append(device.get_config())
        print("Sending telemetry data.")
        send_telemetry_data(
            network,
            args['subtask'],
            gpu_data,
            num_gpus=1,
            time_lapsed=time_lapsed,
            pass_status=process_passed
        )
    except Exception as e:
        print("Telemetry data couldn't be sent, but the command ran successfully.")
        print(f"[WARNING]: {e}")
        pass

    if not process_passed:
        print("Execution status: FAIL")
        sys.exit(1)  # returning non zero return code from the process.

    print("Execution status: PASS")
