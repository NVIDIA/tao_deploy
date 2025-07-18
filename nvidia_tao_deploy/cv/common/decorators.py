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

"""Common decorators used in TAO Toolkit."""

from functools import wraps
import inspect
import os
from omegaconf import OmegaConf

from nvidia_tao_deploy.cv.common.logging import status_logging
from nvidia_tao_deploy.cv.common.utils import TASKS, update_results_dir


def monitor_status(name='module name', mode='gen_trt_engine', hydra=True):
    """Status monitoring decorator."""
    def inner(runner):
        @wraps(runner)
        def _func(cfg, **kwargs):

            if hydra:
                cfg = update_results_dir(cfg, task=mode)
                os.makedirs(cfg.results_dir, exist_ok=True)

                OmegaConf.save(cfg, os.path.join(cfg.results_dir, "experiment.yaml"))

            status_file = os.path.join(cfg.results_dir, "status.json")
            status_logging.set_status_logger(
                status_logging.StatusLogger(
                    filename=status_file,
                    is_master=True,
                    verbosity=1,
                    append=True
                )
            )
            s_logger = status_logging.get_status_logger()
            try:
                s_logger.write(
                    status_level=status_logging.Status.STARTED,
                    message=f"Starting {name} {TASKS[mode]}."
                )
                runner(cfg, **kwargs)
                s_logger.write(
                    status_level=status_logging.Status.RUNNING,
                    message=f"{TASKS[mode].capitalize()} finished successfully."
                )
                if os.getenv("CLOUD_BASED") == "True":
                    s_logger.write(
                        status_level=status_logging.Status.RUNNING,
                        message="Job artifacts in results dir are being uploaded to the cloud"
                    )
            except (KeyboardInterrupt, SystemError):
                s_logger.write(
                    message=f"{TASKS[mode].capitalize()} was interrupted",
                    verbosity_level=status_logging.Verbosity.INFO,
                    status_level=status_logging.Status.FAILURE
                )
            except Exception as e:
                s_logger.write(
                    message=str(e),
                    status_level=status_logging.Status.FAILURE
                )
                raise e

        return _func
    return inner


def override(method):
    """Override decorator.

    Decorator implementing method overriding in python
    Must also use the @subclass class decorator
    """
    method.override = True
    return method


def subclass(class_object):
    """Subclass decorator.

    Verify all @override methods
    Use a class decorator to find the method's class
    """
    for name, method in class_object.__dict__.items():
        if hasattr(method, "override"):
            found = False
            for base_class in inspect.getmro(class_object)[1:]:
                if name in base_class.__dict__:
                    if not method.__doc__:
                        # copy docstring
                        method.__doc__ = base_class.__dict__[name].__doc__
                    found = True
                    break
            assert found, f'"{class_object.__name__}.{name}" not found in any base class'
    return class_object
