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

"""Standalone TensorRT Inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import logging

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.retinanet.dataloader import RetinaNetKITTILoader
from nvidia_tao_deploy.cv.retinanet.inferencer import RetinaNetInferencer
from nvidia_tao_deploy.cv.retinanet.proto.utils import load_proto


logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


@monitor_status(name='retinanet', mode='inference')
def main(args):
    """RetinaNet TRT inference."""
    # Load from proto-based spec file
    es = load_proto(args.experiment_spec)

    conf_thres = es.nms_config.confidence_threshold if es.nms_config.confidence_threshold else 0.01
    batch_size = args.batch_size if args.batch_size else es.eval_config.batch_size
    trt_infer = RetinaNetInferencer(args.model_path, batch_size=batch_size)

    c, h, w = trt_infer._input_shape

    img_mean = es.augmentation_config.image_mean
    if c == 3:
        if img_mean:
            img_mean = [img_mean['b'], img_mean['g'], img_mean['r']]
        else:
            img_mean = [103.939, 116.779, 123.68]
    else:
        if img_mean:
            img_mean = [img_mean['l']]
        else:
            img_mean = [117.3786]

    # Override path if provided through command line args
    if args.image_dir:
        image_dirs = [args.image_dir]
    else:
        image_dirs = [d.image_directory_path for d in es.dataset_config.validation_data_sources]

    # Load mapping_dict from the spec file
    mapping_dict = dict(es.dataset_config.target_class_mapping)

    dl = RetinaNetKITTILoader(
        shape=(c, h, w),
        image_dirs=image_dirs,
        label_dirs=[None],
        mapping_dict=mapping_dict,
        exclude_difficult=True,
        batch_size=batch_size,
        is_inference=True,
        image_mean=img_mean,
        keep_aspect_ratio=False,
        dtype=trt_infer.inputs[0].host.dtype)

    inv_classes = {v: k for k, v in dl.classes.items()}

    if args.results_dir is None:
        results_dir = os.path.dirname(args.model_path)
    else:
        results_dir = args.results_dir
        os.makedirs(results_dir, exist_ok=True)

    output_annotate_root = os.path.join(results_dir, "images_annotated")
    output_label_root = os.path.join(results_dir, "labels")

    os.makedirs(output_annotate_root, exist_ok=True)
    os.makedirs(output_label_root, exist_ok=True)

    for i, (imgs, _) in tqdm(enumerate(dl), total=len(dl), desc="Producing predictions"):
        y_pred = trt_infer.infer(imgs)
        image_paths = dl.image_paths[np.arange(args.batch_size) + args.batch_size * i]

        for i in range(len(y_pred)):
            # Load image
            img = Image.open(image_paths[i])

            orig_width, orig_height = img.size
            width_scale = orig_width / trt_infer.width
            height_scale = orig_height / trt_infer.height

            # Filter and scale back to original image resolution
            y_pred_valid = y_pred[i][y_pred[i][:, 1] > conf_thres]
            y_pred_valid[..., 2] = np.clip(y_pred_valid[..., 2].round(), 0.0, w) * width_scale
            y_pred_valid[..., 3] = np.clip(y_pred_valid[..., 3].round(), 0.0, h) * height_scale
            y_pred_valid[..., 4] = np.clip(y_pred_valid[..., 4].round(), 0.0, w) * width_scale
            y_pred_valid[..., 5] = np.clip(y_pred_valid[..., 5].round(), 0.0, h) * height_scale

            # Store images
            bbox_img, label_strings = trt_infer.draw_bbox(img, y_pred_valid, inv_classes, args.threshold)
            img_filename = os.path.basename(image_paths[i])
            bbox_img.save(os.path.join(output_annotate_root, img_filename))

            # Store labels
            filename, _ = os.path.splitext(img_filename)
            label_file_name = os.path.join(output_label_root, filename + ".txt")
            with open(label_file_name, "w", encoding="utf-8") as f:
                for l_s in label_strings:
                    f.write(l_s)

    logging.info("Finished inference.")


def build_command_line_parser(parser=None):
    """Build the command line parser using argparse.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
                parser mechanism.
    Returns:
        parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(prog='infer', description='Inference with a RetinaNet TRT model.')

    parser.add_argument(
        '-i',
        '--image_dir',
        type=str,
        required=False,
        default=None,
        help='Input directory of images')
    parser.add_argument(
        '-e',
        '--experiment_spec',
        type=str,
        required=True,
        help='Path to the experiment spec file.'
    )
    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        required=True,
        help='Path to the RetinaNet TensorRT engine.'
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        required=False,
        default=1,
        help='Batch size.')
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        default=0.3,
        help='Confidence threshold for inference.')
    parser.add_argument(
        '-r',
        '--results_dir',
        type=str,
        required=True,
        default=None,
        help='Output directory where the log is saved.'
    )
    return parser


def parse_command_line_arguments(args=None):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser(args)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_command_line_arguments()
    main(args)
