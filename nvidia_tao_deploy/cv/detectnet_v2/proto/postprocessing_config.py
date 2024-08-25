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

"""PostProcessingConfig class that holds postprocessing parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from nvidia_tao_deploy.cv.detectnet_v2.proto.clustering_config import build_clustering_config
from nvidia_tao_deploy.cv.detectnet_v2.proto.clustering_config import build_clustering_proto
from nvidia_tao_deploy.cv.detectnet_v2.proto.confidence_config import build_confidence_config
from nvidia_tao_deploy.cv.detectnet_v2.proto.confidence_config import build_confidence_proto
from nvidia_tao_deploy.cv.detectnet_v2.proto.postprocessing_config_pb2 import PostProcessingConfig as\
    PostProcessingProto


def build_postprocessing_config(postprocessing_proto):
    """Build PostProcessingConfig from a proto.

    Args:
        postprocessing_proto: proto.postprocessing_config proto message.

    Returns:
        configs: A dict of PostProcessingConfig instances indexed by target class name.
    """
    configs = {}
    for class_name, config in six.iteritems(postprocessing_proto.target_class_config):
        clustering_config = build_clustering_config(config.clustering_config)
        confidence_config = build_confidence_config(config.confidence_config)
        configs[class_name] = PostProcessingConfig(clustering_config, confidence_config)
    return configs


class PostProcessingConfig(object):
    """Hold the post-processing parameters for one class."""

    def __init__(self, clustering_config, confidence_config):
        """Constructor.

        Args:
            clustering_config (ClusteringConfig object): Built clustering configuration object.
            confidence_config (ConfidenceConfig object): Built confidence configuration object.
        """
        self.clustering_config = clustering_config
        self.confidence_config = confidence_config


def build_postprocessing_proto(postprocessing_config):
    """Build proto from a PostProcessingConfig dictionary.

    Args:
        postprocessing_config: A dict of PostProcessingConfig instances indexed by target class
            name.

    Returns:
        postprocessing_proto: proto.postprocessing_config proto message.
    """
    proto = PostProcessingProto()

    for target_class_name, target_class in six.iteritems(postprocessing_config):
        proto.target_class_config[target_class_name].clustering_config.CopyFrom(
            build_clustering_proto(target_class.clustering_config))
        proto.target_class_config[target_class_name].confidence_config.CopyFrom(
            build_confidence_proto(target_class.confidence_config))

    return proto
