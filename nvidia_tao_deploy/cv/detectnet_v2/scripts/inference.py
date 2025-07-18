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
from nvidia_tao_deploy.cv.detectnet_v2.proto.utils import load_proto
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    level='INFO')
from nvidia_tao_deploy.cv.detectnet_v2.dataloader import DetectNetKITTILoader  # noqa: E402
from nvidia_tao_deploy.cv.detectnet_v2.inferencer import DetectNetInferencer  # noqa: E402
from nvidia_tao_deploy.cv.detectnet_v2.postprocessor import BboxHandler  # noqa: E402


@monitor_status(name='detectnet_v2', mode='inference', hydra=False)
def main(args):
    """DetectNetv2 TRT inference."""
    inferencer_spec = load_proto(args.experiment_spec, "inference")

    # Load target classes from label file
    target_classes = inferencer_spec.inferencer_config.target_classes

    # Load mapping_dict from the spec file
    mapping_dict = {c: c for c in target_classes}
    batch_size = args.batch_size if args.batch_size else inferencer_spec.inferencer_config.batch_size

    trt_infer = DetectNetInferencer(args.model_path,
                                    batch_size=batch_size,
                                    target_classes=target_classes)
    if batch_size != trt_infer.max_batch_size and trt_infer.etlt_type == "uff":
        logging.warning("Using deprecated UFF format. Overriding provided batch size "
                        "%d to engine's batch size %d", batch_size, trt_infer.max_batch_size)
        batch_size = trt_infer.max_batch_size

    c, h, w = trt_infer.input_tensors[0].shape

    dl = DetectNetKITTILoader(
        shape=(c, h, w),
        image_dirs=[args.image_dir],
        label_dirs=[None],
        mapping_dict=mapping_dict,
        exclude_difficult=True,
        batch_size=batch_size,
        is_inference=True,
        image_mean=None,
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype))

    bboxer = BboxHandler(
        batch_size=batch_size,
        frame_height=h,
        frame_width=w,
        target_classes=target_classes,
        postproc_classes=target_classes,
        classwise_cluster_params=inferencer_spec.bbox_handler_config.classwise_bbox_handler_config,
    )

    # Override class mapping with the class order specified by target_classes
    dl.classes = {c: i + 1 for i, c in enumerate(target_classes)}
    dl.class_mapping = {key.lower(): dl.classes[str(val.lower())]
                        for key, val in mapping_dict.items()}

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

    # Get classwise edge color
    box_color = {}
    for k, v in inferencer_spec.bbox_handler_config.classwise_bbox_handler_config.items():
        box_color[k] = (0, 255, 0)
        if v.bbox_color:
            box_color[k] = (v.bbox_color.R, v.bbox_color.G, v.bbox_color.B)

    for i, (imgs, _) in tqdm(enumerate(dl), total=len(dl), desc="Producing predictions"):
        y_pred = trt_infer.infer(imgs)
        processed_inference = bboxer.bbox_preprocessing(y_pred)
        classwise_detections = bboxer.cluster_detections(processed_inference)
        y_pred_valid = bboxer.postprocess(classwise_detections, batch_size, dl.image_size[i], (w, h), dl.classes)

        image_paths = dl.image_paths[np.arange(batch_size) + batch_size * i]

        for img_path, pred in zip(image_paths, y_pred_valid):
            # Load image
            img = Image.open(img_path)

            # No need to rescale here as rescaling was done in bboxer.postprocess
            bbox_img, label_strings = trt_infer.draw_bbox(img, pred, inv_classes, bboxer.state['confidence_th'], box_color)
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
        parser = argparse.ArgumentParser(prog='infer', description='Inference with a DetectNetv2 TRT model.')

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
