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
import json

from tqdm.auto import tqdm

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.mask_rcnn.inferencer import MRCNNInferencer
from nvidia_tao_deploy.cv.mask_rcnn.proto.utils import load_proto
from nvidia_tao_deploy.utils.image_batcher import ImageBatcher


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


def resize_pad(image, model_width, model_height, pad_color=(0, 0, 0)):
    """Resize and Pad.

    A subroutine to implement padding and resizing. This will resize the image to fit
    fully within the input size, and pads the remaining bottom-right portions with
    the value provided.

    Args:
        image (PIL.Image): The PIL image object
        pad_color (list): The RGB values to use for the padded area. Default: Black/Zeros.

    Returns:
        pad (PIL.Image): The PIL image object already padded and cropped,
        scale (list): the resize scale used.
    """
    width, height = image.size
    width_scale = width / model_width
    height_scale = height / model_height
    scale = 1.0 / max(width_scale, height_scale)
    image = image.resize(
        (round(width * scale), round(height * scale)),
        resample=Image.BILINEAR)
    pad = Image.new("RGB", (model_width, model_height))
    pad.paste(pad_color, [0, 0, model_width, model_height])
    pad.paste(image)
    padded = (abs(round(width * scale) - model_width), abs(round(height * scale) - model_height))
    return pad, scale, padded


def get_label_dict(label_txt):
    """Create label dict from txt file."""
    with open(label_txt, 'r', encoding="utf-8") as f:
        labels = f.readlines()
        result = {i + 1: label.strip() for i, label in enumerate(labels)}
        result[-1] = "background"
        return result


@monitor_status(name='mask_rcnn', mode='inference')
def main(args):
    """MRCNN TRT inference."""
    # Load from proto-based spec file
    es = load_proto(args.experiment_spec)
    mask_size = es.maskrcnn_config.mrcnn_resolution if es.maskrcnn_config.mrcnn_resolution else 28
    nms_size = es.maskrcnn_config.test_detections_per_image if es.maskrcnn_config.test_detections_per_image else 100

    trt_infer = MRCNNInferencer(args.model_path,
                                nms_size=nms_size,
                                n_classes=es.data_config.num_classes,
                                mask_size=mask_size)

    # Inference may not have labels. Hence, use image batcher
    batch_size = trt_infer.max_batch_size
    batcher = ImageBatcher(args.image_dir,
                           (batch_size,) + trt_infer._input_shape,
                           trt_infer.inputs[0].host.dtype,
                           preprocessor="MRCNN")

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

    if not os.path.exists(args.class_map):
        raise FileNotFoundError(f"Class map is required for inference! {args.class_map} does not exist.")

    inv_classes = get_label_dict(args.class_map)

    for batch, img_paths, scales in tqdm(batcher.get_batch(), total=batcher.num_batches, desc="Producing predictions"):
        detections = trt_infer.infer(batch, scales)
        for idx, img_path in enumerate(img_paths):
            # Load Image
            img = Image.open(img_path)
            orig_width, orig_height = img.size
            img, sc, padding = resize_pad(img, trt_infer.width, trt_infer.height)
            detections['detection_boxes'][idx] /= sc

            bbox_img, label_strings = trt_infer.draw_bbox_and_segm(img,
                                                                   detections['detection_classes'][idx],
                                                                   detections['detection_scores'][idx],
                                                                   detections['detection_boxes'][idx],
                                                                   detections['detection_masks'][idx],
                                                                   inv_classes,
                                                                   args.threshold)

            # Crop out padded region and resize to original image
            bbox_img = bbox_img.crop((0, 0, trt_infer.width - padding[0], trt_infer.height - padding[1]))
            bbox_img = bbox_img.resize((orig_width, orig_height))

            img_filename = os.path.basename(img_path)
            bbox_img.save(os.path.join(output_annotate_root, img_filename))

            # Store labels
            filename, _ = os.path.splitext(img_filename)
            label_file_name = os.path.join(output_label_root, filename + ".json")

            # Add image path in label dump
            for i in range(len(label_strings)):
                label_strings[i]['image_id'] = img_path

            with open(label_file_name, "w", encoding="utf-8") as f:
                json.dump(label_strings, f, indent=4, sort_keys=True)

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
        parser = argparse.ArgumentParser(prog='infer', description='Inference with a MRCNN TRT model.')

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
        help='Path to the MRCNN TensorRT engine.'
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
        '-c',
        '--class_map',
        type=str,
        default=None,
        required=True,
        help='The path to the class label file.'
    )
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        default=0.6,
        help='Confidence threshold for inference.'
    )
    parser.add_argument(
        '--include_mask',
        action='store_true',
        required=False,
        default=None,
        help=argparse.SUPPRESS
    )
    return parser


def parse_command_line_arguments(args=None):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser(args)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_command_line_arguments()
    main(args)
