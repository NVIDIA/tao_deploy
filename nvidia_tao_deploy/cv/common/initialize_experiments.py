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

"""Common Deploy Flow"""

import os
import tempfile

from nvidia_tao_deploy.cv.common.constants import VALID_IMAGE_EXTENSIONS


def initialize_gen_trt_engine_experiment(cfg):
    """
    Provide the common parameters for TensorRT engine building. All models must
    instantiate an EngineBuilder and call create_engine(); many parameters to these
    are shared across models since they share gen_trt_engine dataclass configs.
    In addition, this handles output filepaths and calibration parameter checks.

    Args:
        cfg: A ExperimentConfig dataclass

    Returns:
        dict: engine_builder_kwargs
        dict: create_engine_kwargs
    """
    engine_file = cfg.gen_trt_engine.trt_engine
    if engine_file is None:
        engine_handle, temp_engine_path = tempfile.mkstemp()
        os.close(engine_handle)
        output_engine_path = temp_engine_path
    else:
        output_engine_path = engine_file

    engine_builder_kwargs = {
        "verbose": cfg.gen_trt_engine.verbose,
        "min_batch_size": cfg.gen_trt_engine.tensorrt.min_batch_size,
        "opt_batch_size": cfg.gen_trt_engine.tensorrt.opt_batch_size,
        "max_batch_size": cfg.gen_trt_engine.tensorrt.max_batch_size,
        "batch_size": cfg.gen_trt_engine.batch_size,
        "timing_cache_path": cfg.gen_trt_engine.timing_cache
    }

    data_type = cfg.gen_trt_engine.tensorrt.data_type.lower()

    # INT8 related configs
    calib_input = None
    calib_cache = None
    calib_batch_size = 0
    calib_batches = 0

    # The networks which have int8 enabled have a calibration config
    # The non-int8 networks are: classification_pyt, gdino, mask_gdino, mask2former, optical_inspection, segformer, vcn
    if data_type == "int8":
        calib_input = cfg.gen_trt_engine.tensorrt.calibration.get('cal_image_dir', None)
        calib_cache = cfg.gen_trt_engine.tensorrt.calibration.get('cal_cache_file', None)
        calib_batch_size = cfg.gen_trt_engine.tensorrt.calibration.cal_batch_size
        calib_batches = cfg.gen_trt_engine.tensorrt.calibration.cal_batches

        if calib_batches <= 0:
            raise ValueError(
                f"Calibration number of batches {calib_batches} is non-positive."
            )
        if calib_batch_size <= 0:
            raise ValueError(
                f"Calibration batch size {calib_batch_size} is non-positive."
            )
        num_cal_images = 0
        for cal_image_dir in calib_input:
            if not os.path.isdir(cal_image_dir):
                raise FileNotFoundError(
                    f"Calibration image directory {cal_image_dir} not found."
                )
            num_imgs = sum(len([f for f in files if f.endswith(VALID_IMAGE_EXTENSIONS)]) for _, _, files in os.walk(cal_image_dir))
            if num_imgs == 0:
                raise FileNotFoundError(
                    f"Calibration image directory {cal_image_dir} is empty."
                )
            num_cal_images += num_imgs
        if num_cal_images < calib_batches * calib_batch_size:
            raise ValueError(
                f"Number of calibration images ({num_cal_images}) should be larger than batches * batch_size ({calib_batches * calib_batch_size})."
            )

    calib_num_images = calib_batch_size * calib_batches

    layers_precision = cfg.gen_trt_engine.tensorrt.layers_precision
    layers_precision_dict = {}
    if layers_precision:
        for layer in layers_precision:
            idx = layer.rfind(':')
            if idx == -1:
                raise IndexError(
                    f"Invalid format to specify layer precision: {layer}. Please set {{layer}}:{{precision}}"
                )
            layer_name = layer[:idx]
            precision = layer[idx + 1:]
            layers_precision_dict[layer_name] = precision

    create_engine_kwargs = {
        "engine_path": output_engine_path,
        "precision": data_type,
        "calib_input": calib_input,
        "calib_cache": calib_cache,
        "calib_num_images": calib_num_images,
        "calib_batch_size": calib_batch_size,
        "layers_precision": layers_precision_dict
    }

    return engine_builder_kwargs, create_engine_kwargs
