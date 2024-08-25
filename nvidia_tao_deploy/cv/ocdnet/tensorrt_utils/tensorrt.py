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

"""OCDNet INT8 calibration APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pycuda.autoinit  # noqa pylint: disable=W0611
import tensorrt as trt


"""Logger for data export APIs."""
logger = logging.getLogger(__name__)

# Array of TensorRT loggers. We need to keep global references to
# the TensorRT loggers that we create to prevent them from being
# garbage collected as those are referenced from C++ code without
# Python knowing about it.
tensorrt_loggers = []


def _create_tensorrt_logger(verbose=False):
    """Create a TensorRT logger.

    Args:
        verbose (bool): whether to make the logger verbose.
    """
    if str(os.getenv('SUPPRES_VERBOSE_LOGGING', '0')) == '1':
        # Do not print any warnings in TLT docker
        trt_verbosity = trt.Logger.Severity.ERROR
    elif verbose:
        trt_verbosity = trt.Logger.Severity.INFO
    else:
        trt_verbosity = trt.Logger.Severity.WARNING
    tensorrt_logger = trt.Logger(trt_verbosity)
    tensorrt_loggers.append(tensorrt_logger)
    return tensorrt_logger
