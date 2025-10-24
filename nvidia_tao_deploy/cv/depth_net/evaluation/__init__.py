# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""DepthNet evaluation module.

This module provides comprehensive evaluation capabilities for DepthNet models, supporting
both monocular and stereo depth estimation with industry-standard metrics and utilities.

The module includes:
- MonoDepthEvaluator class for monocular depth evaluation
- StereoDepthEvaluator class for stereo depth evaluation
- Depth alignment utilities using least squares optimization
- Support for multiple evaluation metrics (abs_rel, d1, bp1, bp2, bp3, epe)
- Batch processing capabilities for efficient evaluation
- Configurable depth range filtering and validation

Supported Metrics:
- Absolute Relative Error (abs_rel): Mean of |pred - gt| / gt
- Delta Accuracy (d1): Percentage of pixels with ratio < 1.25
- Bad Pixel Rate (bp1, bp2, bp3): Percentage of pixels with error > 1, 2, 3 pixels
- End-Point-Error (epe): Mean absolute difference for stereo estimation

Key Features:
- Automatic depth alignment for fair evaluation
- Support for both monocular and stereo depth estimation
- Configurable depth range filtering
- Efficient batch processing
- Integration with TensorRT inference pipeline
- Consistent interface for both mono and stereo evaluation

Usage:
    ```python
    from nvidia_tao_deploy.cv.depth_net.evaluation import MonoDepthEvaluator, StereoDepthEvaluator

    # For monocular depth estimation
    mono_evaluator = MonoDepthEvaluator(align_gt=True, min_depth=0.1, max_depth=80.0)
    mono_evaluator.update(prediction_results)
    mono_metrics = mono_evaluator.compute()

    # For stereo depth estimation
    stereo_evaluator = StereoDepthEvaluator(max_disparity=416)
    stereo_evaluator.update(prediction_results)
    stereo_metrics = stereo_evaluator.compute()
    ```

Dependencies:
    - numpy for numerical computations
    - typing for type hints
"""

from .mono_evaluator import MonoDepthEvaluator
from .stereo_evaluator import StereoDepthEvaluator

__all__ = ['MonoDepthEvaluator', 'StereoDepthEvaluator']
