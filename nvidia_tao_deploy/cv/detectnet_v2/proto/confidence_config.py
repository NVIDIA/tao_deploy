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

"""Confidence config class that holds parameters for postprocessing confidence."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_deploy.cv.detectnet_v2.proto.postprocessing_config_pb2 import ConfidenceConfig as ConfidenceProto


def build_confidence_config(confidence_config):
    """Build ConfidenceConfig from a proto.

    Args:
        confidence_config: confidence_config proto message.

    Returns:
        ConfidenceConfig object.
    """
    return ConfidenceConfig(confidence_config.confidence_model_filename,
                            confidence_config.confidence_threshold)


def build_confidence_proto(confidence_config):
    """Build proto from ConfidenceConfig.

    Args:
        confidence_config: ConfidenceConfig object.

    Returns:
        confidence_config: confidence_config proto.
    """
    proto = ConfidenceProto()
    proto.confidence_model_filename = confidence_config.confidence_model_filename
    proto.confidence_threshold = confidence_config.confidence_threshold
    return proto


class ConfidenceConfig(object):
    """Hold the parameters for postprocessing confidence."""

    def __init__(self, confidence_model_filename, confidence_threshold):
        """Constructor.

        Args:
            confidence_model_filename (str): Absolute path to the confidence model hdf5.
            confidence_threshold (float): Confidence threshold value. Must be >= 0.
        Raises:
            ValueError: If the input arg is not within the accepted range.
        """
        if confidence_threshold < 0.0:
            raise ValueError("ConfidenceConfig.confidence_threshold must be >= 0")

        self.confidence_model_filename = confidence_model_filename
        self.confidence_threshold = confidence_threshold
