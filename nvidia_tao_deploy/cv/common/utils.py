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

"""Helper functions."""

import logging
import os
logger = logging.getLogger(__name__)


def update_results_dir(cfg, task):
    """Update global results_dir based on task.results_dir.

    This function should be called at the beginning of a pipeline script.

    Args:
        cfg (Hydra config): Config object loaded by Hydra
        task (str): TAO pipeline name
    Return:
        Updated cfg
    """
    if cfg[task]['results_dir']:
        cfg.results_dir = cfg[task]['results_dir']
    else:
        cfg.results_dir = os.path.join(cfg.results_dir, task)
        cfg[task]['results_dir'] = cfg.results_dir
    logger.info(f"{task.capitalize()} results will be saved at: %s", cfg.results_dir)
    return cfg
