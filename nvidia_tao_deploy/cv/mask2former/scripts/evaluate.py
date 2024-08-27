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

"""Standalone TensorRT inference."""

import os
import logging
import numpy as np
from tqdm.auto import tqdm

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.dataloader.ade import ADELoader
from nvidia_tao_deploy.dataloader.coco_panoptic import COCOPanopticLoader
from nvidia_tao_deploy.cv.mask2former.dataloader import Mask2formerCOCOLoader
from nvidia_tao_deploy.cv.mask2former.inferencer import Mask2formerInferencer
from nvidia_tao_deploy.cv.mask2former.hydra_config.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.mask2former.metrics import total_intersect_over_union

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="infer", schema=ExperimentConfig
)
@monitor_status(name='mask2former', mode='trt_evaluate')
def main(cfg: ExperimentConfig) -> None:
    """Mask2former TRT evaluation."""
    if not os.path.exists(cfg.evaluate.trt_engine):
        raise FileNotFoundError(f"Provided inference.trt_engine at {cfg.evaluate.trt_engine} does not exist!")
    trt_infer = Mask2formerInferencer(
        cfg.evaluate.trt_engine,
        batch_size=cfg.dataset.val.batch_size)

    assert len(trt_infer.outputs) == 1, "[Experimental] Only support engines exported in the `semantic` mode."
    shape = [trt_infer.max_batch_size] + list(trt_infer._input_shape)

    if cfg.dataset.val.type == 'ade':
        dl = ADELoader(
            cfg.dataset.val.annot_file,
            batch_size=trt_infer.max_batch_size,
            data_format="channels_first",
            shape=shape,
            dtype=trt_infer.inputs[0].host.dtype,
            root_dir=cfg.dataset.val.root_dir,
            eval_samples=None)
    elif cfg.dataset.val.type == 'coco_panoptic':
        dl = COCOPanopticLoader(
            cfg.dataset.val.panoptic_json,
            cfg.dataset.val.img_dir,
            cfg.dataset.val.panoptic_dir,
            shape,
            dtype=trt_infer.inputs[0].host.dtype,
            contiguous_id=False,
            batch_size=trt_infer.max_batch_size,
            data_format='channels_first',
            eval_samples=None)
    else:
        dl = Mask2formerCOCOLoader(
            cfg.dataset.val.instance_json,
            shape,
            dtype=trt_infer.inputs[0].host.dtype,
            batch_size=trt_infer.max_batch_size,
            data_format='channels_first',
            image_dir=cfg.dataset.val.img_dir,
            eval_samples=None)
    total_area_intersect, total_area_union = 0, 0
    total_area_pred_label, total_area_label = 0, 0
    for imgs, segms in tqdm(dl, total=len(dl), desc="Producing predictions"):
        pred_masks = trt_infer.infer(imgs)
        mask_img = np.argmax(pred_masks, axis=1)
        area_intersect, area_union, area_pred_label, area_label = \
            total_intersect_over_union(mask_img,
                                       segms,
                                       cfg.model.sem_seg_head.num_classes,
                                       ignore_index=255,
                                       reduce_zero_label=True)

        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label

    iou = total_area_intersect / total_area_union
    miou = np.nanmean(iou)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    logger.info("Evaluation metrics")
    logger.info("==================")
    logger.info("mIOU: %2.2f" % miou)
    logger.info("Acc: %2.2f" % all_acc)
    logger.info("Finished evaluation.")


if __name__ == '__main__':
    main()
