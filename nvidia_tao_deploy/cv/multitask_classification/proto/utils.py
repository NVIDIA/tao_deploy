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

"""TAO Deploy Config Base Utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

from google.protobuf.text_format import Merge as merge_text_proto
from nvidia_tao_deploy.cv.multitask_classification.proto.experiment_pb2 import Experiment

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


def load_proto(config):
    """Load the experiment proto."""
    proto = Experiment()

    def _load_from_file(filename, pb2):
        if not os.path.exists(filename):
            raise IOError(f"Specfile not found at: {filename}")
        with open(filename, "r", encoding="utf-8") as f:
            merge_text_proto(f.read(), pb2)
    _load_from_file(config, proto)

    return proto
