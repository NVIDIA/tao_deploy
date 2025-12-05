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

"""Standalone TensorRT inference."""

import os
import cv2
import logging
import numpy as np
from PIL import Image
import tensorrt as trt
from tqdm.auto import tqdm

from nvidia_tao_core.config.oneformer.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.oneformer.tokenizer.tokenizer import Tokenize
from nvidia_tao_deploy.cv.oneformer.inferencer import OneformerInferencer

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.mask2former.d2.visualizer import ColorMode, Visualizer
from nvidia_tao_deploy.cv.mask2former.d2.catalog import MetadataCatalog
from nvidia_tao_deploy.utils.image_batcher import ImageBatcher
from nvidia_tao_deploy.cv.mask2former.scripts.inference import get_metadata

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="infer", schema=ExperimentConfig
)
@monitor_status(name='oneformer', mode='inference')
def main(cfg: ExperimentConfig) -> None:
    """Oneformer TRT Inference."""
    if not os.path.exists(cfg.inference.trt_engine):
        raise FileNotFoundError(f"Provided inference.trt_engine at {cfg.inference.trt_engine} does not exist!")

    metadata = get_metadata(cfg)
    MetadataCatalog.get("custom").set(
        thing_classes=metadata["thing_classes"],
        thing_colors=metadata["thing_colors"],
        stuff_classes=metadata["stuff_classes"],
        stuff_colors=metadata["stuff_colors"],
        thing_dataset_id_to_contiguous_id=metadata["thing_dataset_id_to_contiguous_id"],
        stuff_dataset_id_to_contiguous_id=metadata["stuff_dataset_id_to_contiguous_id"],
    )

    trt_infer = OneformerInferencer(
        cfg.inference.trt_engine,
        batch_size=cfg.dataset.test.batch_size,
        is_inference=True)

    profile_idx = getattr(trt_infer, "profile_idx", 0)
    trt_infer.context.set_optimization_profile_async(profile_idx, trt_infer.stream.handle)

    hh, ww = trt_infer.input_tensors[0].tensor_shape[2:]  # IMAGES: [B,C,H,W], TASK_TOKENS: [B,D]
    batcher = ImageBatcher(
        cfg.dataset.test.images,
        (cfg.dataset.test.batch_size, 3, hh, ww),
        trt.nptype(trt_infer.input_tensors[0].tensor_dtype),
        preprocessor="OneFormer")

    task_tokenizer = Tokenize(max_seq_len=cfg.dataset.max_seq_len)
    task_text = f"The task is {cfg.export.task}"

    for batches, img_paths, _ in tqdm(batcher.get_batch(), total=batcher.num_batches, desc="Producing predictions"):
        if len(img_paths) != len(batches):
            batches = batches[:len(img_paths)]

        task_texts = [task_text] * len(batches)
        task_tokens = np.stack([task_tokenizer(task) for task in task_texts])
        task_tokens = np.asarray(task_tokens, dtype=trt.nptype(trt_infer.input_tensors[1].tensor_dtype))
        batch_mask_cls, batch_mask_pred = trt_infer.infer(batches, task_tokens=task_tokens)

        for i, (mask_cls, mask_pred) in enumerate(zip(batch_mask_cls, batch_mask_pred)):

            curr_path = img_paths[i]
            raw_img = load_image(curr_path, target_size=(ww, hh))

            img_for_vis = batches[i].transpose(1, 2, 0).astype(np.uint8)
            visualizer = Visualizer(img_for_vis, MetadataCatalog.get("custom"), instance_mode=ColorMode.IMAGE)

            semseg = trt_infer.postprocess_semseg(mask_cls, mask_pred, output_size=raw_img.shape[:2])

            vis_output = visualizer.draw_sem_seg(
                semseg[0],
            )

            cv2.imwrite(
                os.path.join(cfg.results_dir, os.path.basename(curr_path)[:-4] + ".jpg"),
                vis_output.get_image()
            )

    logging.info("Inference results were saved at %s.", cfg.results_dir)


def load_image(file_name, root_dir=None, target_size=None):
    """Load image.

    Args:
        file_name (str): relative path to an image file (.png).
    Return:
        image (PIL image): loaded image
    """
    root_dir = root_dir or ""
    image = Image.open(os.path.join(root_dir, file_name)).convert('RGB')
    if target_size:
        image = image.resize(target_size)
    image = np.array(image)
    return image


if __name__ == '__main__':
    main()
