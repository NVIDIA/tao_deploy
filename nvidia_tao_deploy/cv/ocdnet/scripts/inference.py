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

"""Inference module."""
import os
import sys
import pathlib
import time
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from nvidia_tao_deploy.cv.ocdnet.config.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.ocdnet.post_processing.seg_detector_representer import get_post_processing
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.ocdnet.utils.utils import show_img, draw_bbox, save_result, get_file_list
from nvidia_tao_deploy.cv.ocdnet.tensorrt_utils.tensorrt_model import TrtModel

__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))


def resize_image(img, image_size):
    """Resize image"""
    resized_img = cv2.resize(img, image_size)
    return resized_img


class Inferencer:
    """Infer class."""

    def __init__(self, model_path, config, post_p_thre=0.7, gpu_id=None):
        """Init model."""
        self.gpu_id = gpu_id
        self.post_process = get_post_processing(config['inference']['post_processing'])
        self.post_process.box_thresh = post_p_thre
        self.img_mode = config['inference']['img_mode']
        self.model = TrtModel(model_path, 1)
        self.model.build_or_load_trt_engine()
        self.is_trt = True

    def predict(self, img_path: str, image_size, is_output_polygon=False):
        """Run prediction."""
        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0).astype(np.float32)
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize_image(img, image_size)
        rgb_mean = np.array([122.67891434, 116.66876762, 104.00698793])
        image = img
        image -= rgb_mean
        image /= 255.
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        batch = {'img': image}
        start = time.time()
        if self.is_trt:
            preds = self.model.predict({"input": image})["pred"]
        box_list, score_list = self.post_process(batch, preds, is_output_polygon=is_output_polygon)
        box_list, score_list = box_list[0], score_list[0]
        if len(box_list) > 0:
            if is_output_polygon:
                idx = [x.sum() > 0 for x in box_list]
                box_list = [box_list[i] for i, v in enumerate(idx) if v]
                score_list = [score_list[i] for i, v in enumerate(idx) if v]
            else:
                idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0
                box_list, score_list = box_list[idx], score_list[idx]
        else:
            box_list, score_list = [], []
        t = time.time() - start
        return preds[0, 0, :, :], box_list, score_list, t


def run_experiment(experiment_config, model_path, post_p_thre, input_folder, output_folder,
                   width, height, polygon, show):
    """Run experiment."""
    experiment_config['model']['pretrained'] = False
    # Init the network
    infer_model = Inferencer(
        model_path,
        experiment_config,
        post_p_thre,
        gpu_id=0
    )
    for img_path in tqdm(get_file_list(input_folder, p_postfix=['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG', '.bmp'])):
        preds, boxes_list, score_list, _ = infer_model.predict(
            img_path,
            (width, height),
            is_output_polygon=polygon
        )
        im = cv2.imread(img_path)
        h_scale = im.shape[0] / float(height)
        w_scale = im.shape[1] / float(width)
        if np.shape(boxes_list) != (0,):
            boxes_list[:, :, 0] = boxes_list[:, :, 0] * w_scale
            boxes_list[:, :, 1] = boxes_list[:, :, 1] * h_scale
        img = draw_bbox(im[:, :, ::-1], boxes_list)
        if show:
            show_img(preds)
            show_img(img, title=os.path.basename(img_path))
            plt.show()
        # save result
        img_path = pathlib.Path(img_path)
        output_path = os.path.join(output_folder, img_path.stem + '_result.jpg')
        pred_path = os.path.join(output_folder, img_path.stem + '_pred.jpg')
        cv2.imwrite(output_path, img[:, :, ::-1])
        cv2.imwrite(pred_path, preds * 255)
        save_result(output_path.replace('_result.jpg', '.txt'), boxes_list, score_list, polygon)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"), config_name="inference", schema=ExperimentConfig
)
@monitor_status(name="ocdnet", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process."""
    if cfg.inference.results_dir is not None:
        results_dir = cfg.inference.results_dir
    else:
        results_dir = os.path.join(cfg.results_dir, "inference")
    os.makedirs(results_dir, exist_ok=True)

    run_experiment(experiment_config=cfg,
                   model_path=cfg.inference.trt_engine,
                   post_p_thre=cfg.inference.post_processing.args.box_thresh,
                   input_folder=cfg.inference.input_folder,
                   output_folder=results_dir,
                   width=cfg.inference.width,
                   height=cfg.inference.height,
                   polygon=cfg.inference.polygon,
                   show=cfg.inference.show
                   )


if __name__ == "__main__":
    main()
