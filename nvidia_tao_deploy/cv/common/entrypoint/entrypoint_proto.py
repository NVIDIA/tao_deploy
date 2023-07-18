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

"""TAO Deploy (TF1) command line wrapper to invoke CLI scripts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import importlib
import logging
import os
import pkgutil
import shlex
import subprocess
import sys
from time import time
import pycuda.driver as cuda

from nvidia_tao_deploy.cv.common.telemetry.nvml_utils import get_device_details
from nvidia_tao_deploy.cv.common.telemetry.telemetry import send_telemetry_data

RELEASE = True

logger = logging.getLogger(__name__)


def get_modules(package):
    """Function to get module supported tasks.

    This function lists out the modules in the iva.X.scripts package
    where the module subtasks are listed, and walks through it to generate a dictionary
    of tasks, parser_function and path to the executable.

    Args:
        No explicit args.

    Returns:
        modules (dict): Dictionary of modules.
    """
    modules = {}
    module_path = package.__path__
    for _, task, _ in pkgutil.walk_packages(module_path):
        module_name = package.__name__ + '.' + task
        if hasattr(importlib.import_module(module_name), "build_command_line_parser"):
            build_parser = getattr(importlib.import_module(module_name),
                                   "build_command_line_parser")
        else:
            build_parser = None
        module_details = {
            "module_name": module_name,
            "build_parser": build_parser,
            "runner_path": os.path.abspath(
                importlib.import_module(module_name).__file__
            )
        }
        modules[task] = module_details
    return modules


def build_command_line_parser(package_name, modules=None):
    """Simple function to build command line parsers.

    This function scans the dictionary of modules determined by the
    get_modules routine and builds a chained parser.

    Args:
        modules (dict): Dictionary of modules as returned by the get_modules function.

    Returns:
        parser (argparse.ArgumentParser): An ArgumentParser class with all the
            subparser instantiated for chained parsing.
    """
    parser = argparse.ArgumentParser(
        package_name,
        add_help=True,
        description="Transfer Learning Toolkit"
    )
    parser.add_argument(
        '--gpu_index',
        type=int,
        default=0,
        help="The index of the GPU to be used.",
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help="Path to the output log file.",
        required=False,
    )

    # module subparser for the respective tasks.
    module_subparsers = parser.add_subparsers(title="tasks")
    for task, details in modules.items():
        subparser = module_subparsers.add_parser(
            task,
            parents=[parser],
            add_help=False)
        subparser = details['build_parser'](subparser)
    return parser


def format_command_line_args(args):
    """Format command line args from command line.

    Args:
        args (dict): Dictionary of parsed command line arguments.

    Returns:
        formatted_string (str): Formatted command line string.
    """
    assert isinstance(args, dict), (
        "The command line args should be formatted to a dictionary."
    )
    formatted_string = ""
    for arg, value in args.items():
        if arg in ["gpu_index", "log_file"]:
            continue
        # Fix arguments that defaults to None, so that they will
        # not be converted to string "None". Simply drop args
        # that have value None.
        # For example, export output_file arg and engine_file arg
        # same for "" for cal_image_dir in export.
        if value in [None, ""]:
            continue
        if isinstance(value, bool):
            if value:
                formatted_string += f"--{arg} "
        elif isinstance(value, list):
            formatted_string += f"--{arg} {' '.join(value)} "
        else:
            formatted_string += f"--{arg} {value} "
    return formatted_string


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


def launch_job(package, package_name, cl_args=None):
    """Wrap CLI builders.

    This function should be included inside package entrypoint/*.py

    import sys
    import nvidia_tao_deploy.cv.X.scripts
    from nvidia_tao_deploy.cv.common.entrypoint import launch_job

    if __name__ == "__main__":
        launch_job(nvidia_tao_deploy.cv.X.scripts, "X", sys.argv[1:])
    """
    # Configure the logger.
    verbosity = "INFO"
    if not RELEASE:
        verbosity = "DEBUG"
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                        level=verbosity)

    # build modules
    modules = get_modules(package)
    parser = build_command_line_parser(package_name, modules)

    # parse command line arguments to module entrypoint script.
    args = vars(parser.parse_args(cl_args))
    gpu_ids = args["gpu_index"]

    log_file = None
    if args['log_file'] is not None:
        log_file = os.path.realpath(args['log_file'])
        log_root = os.path.dirname(log_file)
        if not os.path.exists(log_root):
            os.makedirs(log_root)

    # Get the task to be called from the raw command line arguments.
    task = None
    for arg in sys.argv[1:]:
        if arg in list(modules.keys()):
            task = arg
        break

    # Format final command.
    env_variables = set_gpu_info_single_node(gpu_ids)
    formatted_args = format_command_line_args(args)
    task_command = f"python3 {modules[task]['runner_path']}"

    run_command = f"bash -c '{env_variables} {task_command} {formatted_args}'"

    logger.debug("Run command: %s", run_command)

    process_passed = True
    start = time()
    try:
        if isinstance(log_file, str):
            with open(log_file, "a", encoding="utf-8") as lf:
                subprocess.run(shlex.split(run_command),
                               shell=False,
                               stdout=lf,
                               stderr=lf,
                               check=True)
        else:
            subprocess.run(shlex.split(run_command),
                           shell=False,
                           stdout=sys.stdout,
                           stderr=sys.stdout,
                           check=True)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Command was interrupted.")
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            print(f"TAO Deploy task: {task} failed with error:\n{e.output}")
        process_passed = False

    end = time()
    time_lapsed = end - start

    try:
        gpu_data = []
        for device in get_device_details():
            gpu_data.append(device.get_config())
        logger.info("Sending telemetry data.")
        send_telemetry_data(
            package_name,
            task,
            gpu_data,
            num_gpus=1,
            time_lapsed=time_lapsed,
            pass_status=process_passed
        )
    except Exception as e:
        logger.warning("Telemetry data couldn't be sent, but the command ran successfully.")
        logger.warning("[Error]: {}".format(e))  # noqa pylint: disable=C0209
        pass

    if not process_passed:
        logger.warning("Execution status: FAIL")
        sys.exit(1)  # returning non zero return code from the process.

    logger.info("Execution status: PASS")
