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

import re
import ast
import logging
import importlib
import os
import pkgutil
import shlex
import subprocess
import sys
from time import time
from contextlib import contextmanager

import yaml
import pycuda.driver as cuda

from nvidia_tao_deploy.cv.common.telemetry.nvml_utils import get_device_details
from nvidia_tao_deploy.cv.common.logging import status_logging
from nvidia_tao_core.telemetry.telemetry import send_telemetry_data

RELEASE = True
# Configure the logger.
verbosity = "INFO"
if not RELEASE:
    verbosity = "DEBUG"
logging.basicConfig(
    format="%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s: %(message)s",
    level=verbosity,
)
logger = logging.getLogger(__name__)


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
        module_name = package.__name__ + "." + task
        module_details = {
            "module_name": module_name,
            "runner_path": os.path.abspath(
                importlib.import_module(module_name).__file__
            ),
        }
        modules[task] = module_details
    return modules


def check_valid_gpus(gpu_ids):
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

    assert len(gpu_ids) <= num_gpus_available, "Checking for valid GPU ids and num_gpus."
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)[1:-1]


def set_gpu_info_single_node(gpu_ids):
    """Set gpu environment variable for single node."""
    check_valid_gpus(gpu_ids)

    env_variable = ""
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if visible_devices is not None:
        env_variable = f" CUDA_VISIBLE_DEVICES={visible_devices}"

    return env_variable


def command_line_parser(parser, subtasks):
    """Build command line parser."""
    parser.add_argument(
        "subtask",
        choices=subtasks.keys(),
        help="Subtask for a given task/model.",
    )
    parser.add_argument(
        "-e",
        "--experiment_spec_file",
        help="Path to the experiment spec file.",
        required=True,
        default=None,
    )
    # Parse the arguments.
    args, unknown_args = parser.parse_known_args()

    return args, unknown_args


@contextmanager
def dual_output(log_file=None):
    """Context manager to handle dual output redirection for subprocess.

    Args:
    - log_file (str, optional): Path to the log file. If provided, output will be
      redirected to both sys.stdout and the specified log file. If not provided,
      output will only go to sys.stdout.

    Yields:
    - stdout_target (file object): Target for stdout output (sys.stdout or log file).
    - log_target (file object or None): Target for log file output, or None if log_file
      is not provided.
    """
    if log_file:
        with open(log_file, "a") as f:  # pylint: disable=unspecified-encoding
            yield sys.stdout, f
    else:
        yield sys.stdout, None


def launch(args, unknown_args, subtasks, network="tao-deploy"):
    """Parse the command line and kick off the entrypoint.

    Args:

        parser (argparse.ArgumentParser): Parser object to define the command line args.
        subtasks (list): List of subtasks.
    """
    script_args = ""
    # Check for whether the experiment spec file exists.
    if not os.path.exists(args["experiment_spec_file"]):
        raise FileNotFoundError(
            f'Experiment spec file wasn not found at {args["experiment_spec_file"]}'
        )
    path, name = os.path.split(args["experiment_spec_file"])
    if path != "":
        script_args += f" --config-path {os.path.realpath(path)}"
    script_args += f" --config-name {name}"

    # This enables a results_dir arg to be passed from the microservice side,
    # but there is no --results_dir cmdline arg. Instead, the spec field must be used
    if "results_dir" in args:
        script_args += " results_dir=" + args["results_dir"]

    unknown_args_as_str = " ".join(unknown_args)

    # Precedence these settings: cmdline > specfile > default
    multigpu_support = ["evaluate", "inference"]
    overrides = ["num_gpus", "gpu_ids"]
    num_gpus = 1
    gpu_ids = [0]
    if args["subtask"] in multigpu_support:
        # Parsing cmdline override
        if any(arg in unknown_args_as_str for arg in overrides):
            if "num_gpus" in unknown_args_as_str:
                num_gpus = int(
                    unknown_args_as_str.split('num_gpus=')[1].split()[0]
                )
            if "gpu_ids" in unknown_args_as_str:
                gpu_ids = ast.literal_eval(
                    unknown_args_as_str.split('gpu_ids=')[1].split()[0]
                )
        # If no cmdline override, look at specfile
        else:
            with open(args["experiment_spec_file"], 'r', encoding='utf-8') as spec:
                exp_config = yaml.safe_load(spec)
                if args["subtask"] in exp_config:
                    if 'num_gpus' in exp_config[args["subtask"]]:
                        num_gpus = exp_config[args["subtask"]]['num_gpus']
                    if 'gpu_ids' in exp_config[args["subtask"]]:
                        gpu_ids = exp_config[args["subtask"]]['gpu_ids']
        # @seanf: For now, we don't support multi-gpu for any task
        num_gpus = 1
        if len(gpu_ids) > 1:
            gpu_ids = [max(gpu_ids)]
    else:
        if "gen_trt_engine.gpu_id" in unknown_args_as_str:
            gpu_ids = [int(
                unknown_args_as_str.split('gpu_id=')[1].split()[0]
            )]
        else:
            with open(args["experiment_spec_file"], 'r', encoding='utf-8') as spec:
                exp_config = yaml.safe_load(spec)
                if args["subtask"] in exp_config:
                    if 'gpu_id' in exp_config[args["subtask"]]:
                        gpu_ids = [exp_config[args["subtask"]]['gpu_id']]

    if num_gpus != len(gpu_ids):
        logging.info("The number of GPUs ({num_gpus}) must be the same as the number of GPU indices ({gpu_ids}) provided.".format(num_gpus=num_gpus, gpu_ids=gpu_ids))
        num_gpus = max(num_gpus, len(gpu_ids))
        gpu_ids = list(range(num_gpus)) if len(gpu_ids) != num_gpus else gpu_ids
        logging.info("Using GPUs {gpu_ids} (total {num_gpus})".format(num_gpus=num_gpus, gpu_ids=gpu_ids))

    script = subtasks[args["subtask"]]["runner_path"]

    log_file = ""
    if os.getenv("JOB_ID"):
        logs_dir = os.getenv('TAO_MICROSERVICES_TTY_LOG', '/results')
        log_file = f"{logs_dir}/{os.getenv('JOB_ID')}/microservices_log.txt"

    task_command = f"python {script} {script_args} {unknown_args_as_str}"
    logger.debug(task_command)
    env_variables = set_gpu_info_single_node(gpu_ids)

    run_command = f"bash -c '{env_variables} {task_command}'"
    process_passed = False
    start = time()
    progress_bar_pattern = re.compile(r"Epoch \d+: \s*\d+%|\[.*\]")

    try:
        # Run the script.
        with dual_output(log_file) as (stdout_target, log_target):
            with subprocess.Popen(
                shlex.split(run_command),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,  # Line-buffered
                universal_newlines=True  # Text mode
            ) as proc:
                last_progress_bar_line = None

                for line in proc.stdout:
                    # Check if the line contains \r or matches the progress bar pattern
                    if '\r' in line or progress_bar_pattern.search(line):
                        last_progress_bar_line = line.strip()
                        # Print the progress bar line to the terminal
                        stdout_target.write('\r' + last_progress_bar_line)
                        stdout_target.flush()
                    else:
                        # Write the final progress bar line to the log file before a new log line
                        if last_progress_bar_line:
                            if log_target:
                                log_target.write(last_progress_bar_line + '\n')
                                log_target.flush()
                            last_progress_bar_line = None
                        stdout_target.write(line)
                        stdout_target.flush()
                        if log_target:
                            log_target.write(line)
                            log_target.flush()

                proc.wait()  # Wait for the process to complete
                # Write the final progress bar line after process completion
                if last_progress_bar_line and log_target:
                    log_target.write(last_progress_bar_line + '\n')
                    log_target.flush()
                if proc.returncode == 0:
                    process_passed = True

    except (KeyboardInterrupt, SystemExit) as e:
        logging.info("Command was interrupted due to {}".format(str(e)))
        process_passed = True
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            logging.info(e.output)
        process_passed = False

    end = time()
    time_lapsed = int(end - start)

    try:
        gpu_data = []
        for device in get_device_details():
            gpu_data.append(device.get_config())
        logger.info("Sending telemetry data.")
        send_telemetry_data(
            network,
            args["subtask"],
            gpu_data,
            num_gpus=num_gpus,
            time_lapsed=time_lapsed,
            pass_status=process_passed,
        )
    except Exception as e:
        logger.warning(
            "Telemetry data couldn't be sent, but the command ran successfully."
        )
        logger.warning("{}".format(e))
        pass

    if not process_passed:
        status_logging.get_status_logger().write(
            message=f"{args['subtask']} action failed for {network}",
            status_level=status_logging.Status.FAILURE,
        )
        logger.info("Execution status: FAIL")
        sys.exit(1)

    logger.info("Execution status: PASS")
    sys.exit(0)
