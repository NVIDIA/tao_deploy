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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from PIL import Image
import logging

import numpy as np
from tqdm.auto import tqdm

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.efficientdet_tf1.inferencer import EfficientDetInferencer
from nvidia_tao_deploy.cv.efficientdet_tf1.proto.utils import load_proto

from nvidia_tao_deploy.utils.image_batcher import ImageBatcher


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


def get_label_dict(label_txt):
    """Create label dict from txt file."""
    with open(label_txt, 'r', encoding="utf-8") as f:
        labels = f.readlines()
        result = {i + 1: label.strip() for i, label in enumerate(labels)}
        result[-1] = "background"
        return result


@monitor_status(name='efficientdet_tf1', mode='inference')
def main(args):
    """EfficientDet TRT inference."""
    # Load from proto-based spec file
    es = load_proto(args.experiment_spec)

    max_detections_per_image = es.eval_config.max_detections_per_image if es.eval_config.max_detections_per_image else 100
    trt_infer = EfficientDetInferencer(args.model_path, max_detections_per_image=max_detections_per_image)

    # Inference may not have labels. Hence, use image batcher
    batcher = ImageBatcher(args.image_dir,
                           tuple(trt_infer._input_shape),
                           trt_infer.inputs[0]['dtype'],
                           preprocessor="EfficientDet")

    # Create results directories
    if args.results_dir is None:
        results_dir = os.path.dirname(args.model_path)
    else:
        results_dir = args.results_dir
        os.makedirs(results_dir, exist_ok=True)

    output_annotate_root = os.path.join(results_dir, "images_annotated")
    output_label_root = os.path.join(results_dir, "labels")

    os.makedirs(output_annotate_root, exist_ok=True)
    os.makedirs(output_label_root, exist_ok=True)

    if args.class_map and not os.path.exists(args.class_map):
        raise FileNotFoundError(f"Class map at {args.class_map} does not exist.")

    if args.class_map:
        inv_classes = get_label_dict(args.class_map)
    else:
        inv_classes = None
        logger.debug("label_map was not provided. Hence, class predictions will not be displayed on the visualization.")

    for batch, img_paths, scales in tqdm(batcher.get_batch(), total=batcher.num_batches, desc="Producing predictions"):
        detections = trt_infer.infer(batch, scales)

        y_pred_valid = np.concatenate([detections['detection_classes'][..., None],
                                      detections['detection_scores'][..., None],
                                      detections['detection_boxes']], axis=-1)

        for img_path, pred in zip(img_paths, y_pred_valid):
            # Load Image
            img = Image.open(img_path)
            orig_width, orig_height = img.size

            # Convert xywh to xyxy
            pred[:, 4:] += pred[:, 2:4]
            pred[..., 2::4] = np.clip(pred[..., 2::4], 0.0, orig_width)
            pred[..., 3::5] = np.clip(pred[..., 3::5], 0.0, orig_height)

            # Scale back the predictions
            # pred[:, 2:6] *= sc

            bbox_img, label_strings = trt_infer.draw_bbox(img, pred, inv_classes, args.threshold)
            img_filename = os.path.basename(img_path)
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
        parser = argparse.ArgumentParser(prog='infer', description='Inference with an EfficientDet TRT model.')

    parser.add_argument(
        '-e',
        '--experiment_spec',
        type=str,
        required=True,
        help='Path to the experiment spec file.'
    )
    parser.add_argument(
        '-i',
        '--image_dir',
        type=str,
        required=True,
        default=None,
        help='Input directory of images')
    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        required=True,
        help='Path to the EfficientDet TensorRT engine.'
    )
    parser.add_argument(
        '-c',
        '--class_map',
        type=str,
        default=None,
        required=False,
        help='The path to the class label file.'
    )
    parser.add_argument(
        '-r',
        '--results_dir',
        type=str,
        required=True,
        default=None,
        help='Output directory where the log is saved.'
    )
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for inference.'
    )
    return parser


def parse_command_line_arguments(args=None):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser(args)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_command_line_arguments()
    main(args)
