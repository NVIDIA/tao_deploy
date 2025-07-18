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

"""TAO Deploy command line wrapper to invoke CLI scripts."""

import sys
from nvidia_tao_deploy.cv.common.entrypoint.entrypoint_proto import launch_job
import nvidia_tao_deploy.cv.faster_rcnn.scripts


def main():
    """Function to launch the job."""
    launch_job(nvidia_tao_deploy.cv.faster_rcnn.scripts, "faster_rcnn", sys.argv[1:])


if __name__ == "__main__":
    main()
