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
import numpy as np
import pandas as pd
import tensorrt as trt
from tqdm.auto import tqdm

from nvidia_tao_core.config.ml_recog.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.ml_recog.inferencer import MLRecogInferencer
from nvidia_tao_deploy.cv.ml_recog.dataloader import MLRecogInferenceLoader, MLRecogClassificationLoader
from nvidia_tao_deploy.utils.path_utils import expand_path

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="inference", schema=ExperimentConfig
)
@monitor_status(name='ml_recog', mode='inference')
def main(cfg: ExperimentConfig) -> None:
    """MLRecog TRT inference."""
    classmap = cfg.dataset.class_map

    if classmap:
        # if classmap is provided, we explicitly set the mapping from the text file
        if not os.path.exists(classmap):
            raise FileNotFoundError(f"{classmap} does not exist!")

        with open(classmap, "r", encoding="utf-8") as f:
            mapping_dict = {line.rstrip(): idx for idx, line in enumerate(f.readlines())}
    else:
        # If not, the order of the classes are alphanumeric as defined by Keras
        # Ref: https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/preprocessing/image.py#L507
        mapping_dict = {}
        for idx, subdir in enumerate(sorted(os.listdir(cfg.dataset.val_dataset["reference"]))):
            if os.path.isdir(os.path.join(cfg.dataset.val_dataset["reference"], subdir)):
                mapping_dict[subdir] = idx

    top_k = cfg.inference.topk
    img_mean = cfg.dataset.pixel_mean
    img_std = cfg.dataset.pixel_std
    batch_size = cfg.inference.batch_size
    input_shape = (batch_size, cfg.model.input_channels, cfg.model.input_height, cfg.model.input_width)

    trt_infer = MLRecogInferencer(cfg.inference.trt_engine,
                                  input_shape=input_shape,
                                  batch_size=batch_size)

    gallery_dl = MLRecogClassificationLoader(
        trt_infer.input_tensors[0].shape,
        cfg.dataset.val_dataset["reference"],
        mapping_dict,
        batch_size=batch_size,
        image_mean=img_mean,
        image_std=img_std,
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype))

    query_dl = MLRecogInferenceLoader(
        trt_infer.input_tensors[0].shape,
        cfg.inference.input_path,
        cfg.inference.inference_input_type,
        batch_size=batch_size,
        image_mean=img_mean,
        image_std=img_std,
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype))

    logging.info("Loading gallery dataset...")
    trt_infer.train_knn(gallery_dl, k=top_k)

    result_csv_path = os.path.join(cfg.results_dir, 'trt_result.csv')
    with open(expand_path(result_csv_path), 'w', encoding="utf-8") as csv_f:
        for i, imgs in tqdm(enumerate(query_dl), total=len(query_dl), desc="Producing predictions"):
            dists, indices = trt_infer.infer(imgs)
            image_paths = query_dl.image_paths[np.arange(batch_size) + batch_size * i]
            class_indices = []
            class_labels = []
            for ind in indices:
                class_inds = [gallery_dl.labels[i] for i in ind]
                class_indices.append(class_inds)

                # Map label index to label name
                class_lab = map(lambda i: list(mapping_dict.keys())
                                [list(mapping_dict.values()).index(i)],
                                class_inds)
                class_labels.append(list(class_lab))

            dist_list = []
            for dist in dists:
                dist_list.append([round(float(d)**2, 4) for d in dist])  # match tao distance outputs: L2 distance squared

            # Write predictions to file
            df = pd.DataFrame(zip(image_paths, class_labels, dist_list))
            df.to_csv(csv_f, header=False, index=False)
    logging.info("Finished inference.")


if __name__ == '__main__':

    main()
