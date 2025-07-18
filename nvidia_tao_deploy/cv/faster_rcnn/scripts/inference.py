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

import argparse
import os
from PIL import Image
import numpy as np
import tensorrt as trt
from tqdm.auto import tqdm

import logging

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.faster_rcnn.dataloader import FRCNNKITTILoader, aug_letterbox_resize
from nvidia_tao_deploy.cv.faster_rcnn.inferencer import FRCNNInferencer
from nvidia_tao_deploy.cv.faster_rcnn.proto.utils import load_proto


logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


@monitor_status(name='faster_rcnn', mode='inference', hydra=False)
def main(args):
    """FRCNN TRT inference."""
    # Load from proto-based spec file
    es = load_proto(args.experiment_spec)
    infer_config = es.inference_config
    dataset_config = es.dataset_config

    batch_size = args.batch_size if args.batch_size else infer_config.batch_size
    if batch_size <= 0:
        raise ValueError(f"Inference batch size should be >=1, got {batch_size}, please check inference_config.batch_size")
    trt_infer = FRCNNInferencer(args.model_path, batch_size=batch_size)
    c, h, w = trt_infer.input_tensors[0].shape
    img_mean = es.model_config.input_image_config.image_channel_mean
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

    # Load mapping_dict from the spec file
    mapping_dict = dict(dataset_config.target_class_mapping)

    # Override eval_dataset_path from spec file if image directory is provided
    if args.image_dir:
        image_dirs = args.image_dir
    else:
        image_dirs = infer_config.images_dir

    dl = FRCNNKITTILoader(
        shape=(c, h, w),
        image_dirs=[image_dirs],
        label_dirs=[None],
        mapping_dict=mapping_dict,
        exclude_difficult=True,
        batch_size=batch_size,
        is_inference=True,
        image_mean=img_mean,
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype))

    inv_classes = {v: k for k, v in dl.classes.items()}
    inv_classes[-1] = "background"  # Dummy class to filter backgrounds

    if args.results_dir is None:
        results_dir = os.path.dirname(args.model_path)
    else:
        results_dir = args.results_dir
        os.makedirs(results_dir, exist_ok=True)

    if infer_config.detection_image_output_dir:
        output_annotate_root = infer_config.detection_image_output_dir
    else:
        output_annotate_root = os.path.join(results_dir, "images_annotated")
    os.makedirs(output_annotate_root, exist_ok=True)

    if infer_config.labels_dump_dir:
        output_label_root = infer_config.labels_dump_dir
    else:
        output_label_root = os.path.join(results_dir, "labels")
    os.makedirs(output_label_root, exist_ok=True)

    for i, (imgs, _) in tqdm(enumerate(dl), total=len(dl), desc="Producing predictions"):
        image_paths = dl.image_paths[np.arange(batch_size) + batch_size * i]

        y_pred = trt_infer.infer(imgs)
        for i in range(len(y_pred)):
            y_pred_valid = y_pred[i]
            target_size = np.array([w, h, w, h])

            # Scale back bounding box coordinates
            y_pred_valid[:, 2:6] *= target_size[None, :]

            # Load image
            img = Image.open(image_paths[i])
            orig_width, orig_height = img.size
            img, _, crop_coord = aug_letterbox_resize(img,
                                                      y_pred_valid[:, 2:6],
                                                      num_channels=c,
                                                      resize_shape=(trt_infer.width, trt_infer.height))
            img = Image.fromarray(img.astype('uint8'))

            # Store images
            bbox_img, label_strings = trt_infer.draw_bbox(img, y_pred_valid, inv_classes, infer_config.bbox_visualize_threshold)
            bbox_img = bbox_img.crop((crop_coord[0], crop_coord[1], crop_coord[2], crop_coord[3]))
            bbox_img = bbox_img.resize((orig_width, orig_height))

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
        parser = argparse.ArgumentParser(prog='infer', description='Inference with a FRCNN TRT model.')

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
        help='Path to the FRCNN TensorRT engine.'
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        required=False,
        default=None,
        help='Batch size.')
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
