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

"""Standalone TensorRT inference."""

import logging

import os
import json
import pandas as pd
import numpy as np

from omegaconf import OmegaConf
import tensorrt as trt
from tqdm.auto import tqdm

from nvidia_tao_core.config.classification_tf2.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.classification_tf1.inferencer import ClassificationInferencer
from nvidia_tao_deploy.cv.classification_tf1.dataloader import ClassificationLoader

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="experiment_spec", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for TRT engine inference."""
    run_inference(cfg=cfg)


@monitor_status(name='classification_tf2', mode='inference')
def run_inference(cfg: ExperimentConfig) -> None:
    """Classification TRT inference."""
    classmap = cfg.inference.classmap

    if classmap:
        # if classmap is provided, we explicitly set the mapping from the json file
        if not os.path.exists(classmap):
            raise FileNotFoundError(f"{classmap} does not exist!")

        with open(classmap, "r", encoding="utf-8") as f:
            mapping_dict = json.load(f)
    else:
        # If not, the order of the classes are alphanumeric as defined by Keras
        # Ref: https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/preprocessing/image.py#L507
        mapping_dict = {}
        for idx, subdir in enumerate(sorted(os.listdir(cfg.evaluate.dataset_path))):
            if os.path.isdir(os.path.join(cfg.evaluate.dataset_path, subdir)):
                mapping_dict[subdir] = idx

    mode = cfg.dataset.preprocess_mode
    interpolation_method = cfg.model.resize_interpolation_method
    crop = "center" if cfg.dataset.augmentation.enable_center_crop else None
    data_format = cfg.data_format  # channels_first
    image_mean = OmegaConf.to_container(cfg.dataset.image_mean)
    batch_size = cfg.evaluate.batch_size

    trt_infer = ClassificationInferencer(cfg.inference.trt_engine, data_format='channel_first', batch_size=batch_size)

    dl = ClassificationLoader(
        trt_infer.input_tensors[0].shape,
        [cfg.inference.image_dir],
        mapping_dict,
        is_inference=True,
        data_format=data_format,
        interpolation_method=interpolation_method,
        mode=mode,
        crop=crop,
        batch_size=cfg.evaluate.batch_size,
        image_mean=image_mean,
        image_depth=cfg.model.input_image_depth,
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype))

    os.makedirs(cfg.results_dir, exist_ok=True)
    result_csv_path = os.path.join(cfg.results_dir, 'result.csv')
    with open(result_csv_path, 'w', encoding="utf-8") as csv_f:
        for i, (imgs, _) in tqdm(enumerate(dl), total=len(dl), desc="Producing predictions"):
            image_paths = dl.image_paths[np.arange(batch_size) + batch_size * i]

            y_pred = trt_infer.infer(imgs)
            # Class output from softmax layer
            class_indices = np.argmax(y_pred, axis=1)
            # Map label index to label name
            class_labels = map(lambda i: list(mapping_dict.keys())
                               [list(mapping_dict.values()).index(i)],
                               class_indices)
            conf = np.max(y_pred, axis=1)
            # Write predictions to file
            df = pd.DataFrame(zip(image_paths, class_labels, conf))
            df.to_csv(csv_f, header=False, index=False)
    logging.info("Finished inference.")


if __name__ == '__main__':

    main()
