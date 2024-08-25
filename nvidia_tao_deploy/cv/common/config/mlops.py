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

"""Default config file"""

from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class WandBConfig:
    """Configuration element wandb client."""

    project: str = "TAO Toolkit"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=lambda: [])
    reinit: bool = False
    sync_tensorboard: bool = True
    save_code: bool = False
    name: str = None


@dataclass
class ClearMLConfig:
    """Configration element for clearml client."""

    project: str = "TAO Toolkit"
    task: str = "train"
    deferred_init: bool = False
    reuse_last_task_id: bool = False
    continue_last_task: bool = False
    tags: List[str] = field(default_factory=lambda: [])
