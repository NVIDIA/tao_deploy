# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common decorators used in TAO Toolkit."""

from functools import wraps
import inspect
import os
import warnings

from colorama import init
from termcolor import colored
# Import Hydra exception classes for config error handling
try:
    from hydra.errors import ConfigCompositionException, MissingConfigException, HydraException
    from omegaconf.errors import ConfigKeyError, MissingMandatoryValue, UnsupportedInterpolationType
    from omegaconf import OmegaConf
    # Alias for backward compatibility
    HydraError = HydraException
except ImportError:
    # Fallback for older versions or if imports fail
    ConfigCompositionException = Exception
    MissingConfigException = Exception
    HydraError = Exception
    HydraException = Exception
    ConfigKeyError = Exception
    MissingMandatoryValue = Exception
    UnsupportedInterpolationType = Exception
    OmegaConf = Exception

from nvidia_tao_deploy.cv.common.logging import status_logging
from nvidia_tao_deploy.cv.common.utils import TASKS, update_results_dir


# Import validation error classes
try:
    from marshmallow.exceptions import ValidationError as MarshmallowValidationError
except ImportError:
    MarshmallowValidationError = Exception


def experimental(reason):
    """
    This is a decorator which can be used to mark functions
    as experimental. It will result in a warning being emitted
    when the function is used.
    """
    init()
    if isinstance(reason, str):
        def decorator(func1):

            fmt1 = colored("Call to experimental function {name} ({reason}).", 'white', 'on_yellow')
            if inspect.isclass(func1):
                fmt1 = colored("Call to experimental class {name} ({reason}).", 'white', 'on_yellow')

            @wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', UserWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=UserWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', UserWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    if inspect.isclass(reason) or inspect.isfunction(reason):
        func2 = reason

        fmt2 = colored("Call to experimental function {name}.", 'white', 'on_yellow')
        if inspect.isclass(func2):
            fmt2 = colored("Call to experimental class {name}.", 'white', 'on_yellow')

        @wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', UserWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=UserWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', UserWarning)
            return func2(*args, **kwargs)

        return new_func2

    raise TypeError(repr(type(reason)))


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
                if os.getenv("CLOUD_BASED") == "True":
                    s_logger.write(
                        status_level=status_logging.Status.RUNNING,
                        message="Job artifacts in results dir are being uploaded to the cloud"
                    )
                s_logger.write(
                    status_level=status_logging.Status.SUCCESS,
                    message=f"{TASKS[mode].capitalize()} finished successfully."
                )
            except (KeyboardInterrupt, SystemError):
                s_logger.write(
                    message=f"User/System interruption: {mode.capitalize()} was interrupted",
                    verbosity_level=status_logging.Verbosity.INFO,
                    status_level=status_logging.Status.FAILURE
                )
            except (
                ConfigCompositionException,
                MissingConfigException,
                ConfigKeyError,
                MissingMandatoryValue,
                UnsupportedInterpolationType,
            ) as e:
                s_logger.write(
                    message=f"Configuration error: {str(e)}",
                    status_level=status_logging.Status.FAILURE
                )
                raise e
            except NotImplementedError as e:
                s_logger.write(
                    message=f"Feature not implemented: {str(e)}",
                    status_level=status_logging.Status.FAILURE
                )
                raise e
            except (ValueError, TypeError) as e:
                s_logger.write(
                    message=f"Parameter validation error: {str(e)}",
                    status_level=status_logging.Status.FAILURE
                )
                raise e
            except (FileNotFoundError, PermissionError, OSError, IOError) as e:
                s_logger.write(
                    message=f"File system error: {str(e)}",
                    status_level=status_logging.Status.FAILURE
                )
                raise e
            except MarshmallowValidationError as e:
                s_logger.write(
                    message=f"Schema validation error: {str(e)}",
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
