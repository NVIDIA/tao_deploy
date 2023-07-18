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

"""Standalone TensorRT evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging

import os
import json
import numpy as np

from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score

from nvidia_tao_deploy.cv.classification_tf1.inferencer import ClassificationInferencer
from nvidia_tao_deploy.cv.classification_tf1.dataloader import ClassificationLoader
from nvidia_tao_deploy.cv.common.decorators import monitor_status

from nvidia_tao_deploy.cv.classification_tf1.proto.utils import load_proto

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


@monitor_status(name='classification_tf1', mode='evaluation')
def main(args):
    """Classification TRT evaluation."""
    # Load from proto-based spec file
    es = load_proto(args.experiment_spec)
    interpolation = es.model_config.resize_interpolation_method if es.model_config.resize_interpolation_method else 0
    interpolation_map = {
        0: "bilinear",
        1: "bicubic"
    }
    interpolation_method = interpolation_map[interpolation]
    mode = es.train_config.preprocess_mode if es.train_config.preprocess_mode else "caffe"
    crop = "center" if es.eval_config.enable_center_crop else None
    image_mean = es.train_config.image_mean
    if image_mean:
        assert all(c in image_mean for c in ['r', 'g', 'b']), (
            "'r', 'g', 'b' should all be present in image_mean "
            "for images with 3 channels."
        )
        image_mean = [image_mean['b'], image_mean['g'], image_mean['r']]
    else:
        image_mean = [103.939, 116.779, 123.68]

    top_k = es.eval_config.top_k if es.eval_config.top_k else 5
    data_format = "channels_first"  # TF1 is always channels first
    batch_size = es.eval_config.batch_size if args.batch_size is None else args.batch_size

    # Override eval_dataset_path from spec file if image directory is provided
    image_dirs = args.image_dir if args.image_dir else es.eval_config.eval_dataset_path

    if args.classmap:
        # if classmap is provided, we explicitly set the mapping from the json file
        if not os.path.exists(args.classmap):
            raise FileNotFoundError(f"{args.classmap} does not exist!")

        with open(args.classmap, "r", encoding="utf-8") as f:
            mapping_dict = json.load(f)
    else:
        # If not, the order of the classes are alphanumeric as defined by Keras
        # Ref: https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/preprocessing/image.py#L507
        mapping_dict = {}
        for idx, subdir in enumerate(sorted(os.listdir(image_dirs))):
            if os.path.isdir(os.path.join(image_dirs, subdir)):
                mapping_dict[subdir] = idx

    target_names = [c[0] for c in sorted(mapping_dict.items(), key=lambda x:x[1])]

    trt_infer = ClassificationInferencer(args.model_path, data_format=data_format, batch_size=batch_size)

    if trt_infer.etlt_type == "uff" and batch_size != 1:
        logger.warning("The etlt file was in deprecated UFF format which does not support dynmaic batch size. "
                       "Overriding the batch size to 1")
        batch_size = 1

    dl = ClassificationLoader(
        trt_infer._input_shape,
        [image_dirs],
        mapping_dict,
        data_format=data_format,
        interpolation_method=interpolation_method,
        mode=mode,
        crop=crop,
        batch_size=batch_size,
        image_mean=image_mean,
        dtype=trt_infer.inputs[0].host.dtype)

    gt_labels = []
    pred_labels = []
    for imgs, labels in tqdm(dl, total=len(dl), desc="Producing predictions"):
        gt_labels.extend(labels)
        y_pred = trt_infer.infer(imgs)
        pred_labels.extend(y_pred)

    # Check output classes
    output_num_classes = pred_labels[0].shape[0]
    if len(mapping_dict) != output_num_classes:
        raise ValueError(f"Provided class map has {len(mapping_dict)} classes while the engine expects {output_num_classes} classes.")

    gt_labels = np.array(gt_labels)
    pred_labels = np.array(pred_labels)

    # Metric calculation
    if pred_labels.shape[-1] == 2:
        # If there are only two classes, sklearn perceive the problem as binary classification
        # and requires predictions to be in (num_samples, ) rather than (num_samples, num_classes)
        scores = top_k_accuracy_score(gt_labels, pred_labels[:, 1], k=top_k)
    else:
        scores = top_k_accuracy_score(gt_labels, pred_labels, k=top_k)
    logging.info("Top %s scores: %s", top_k, scores)

    logging.info("Confusion Matrix")
    y_predictions = np.argmax(pred_labels, axis=1)
    print(confusion_matrix(gt_labels, y_predictions))
    logging.info("Classification Report")
    target_names = [c[0] for c in sorted(mapping_dict.items(), key=lambda x:x[1])]
    print(classification_report(gt_labels, y_predictions, target_names=target_names))

    # Store evaluation results into JSON
    eval_results = {"top_k_accuracy": scores}
    if args.results_dir is None:
        results_dir = os.path.dirname(args.model_path)
    else:
        results_dir = args.results_dir
    with open(os.path.join(results_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f)
    logging.info("Finished evaluation.")


def build_command_line_parser(parser=None):
    """Build the command line parser using argparse.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
                parser mechanism.
    Returns:
        parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(prog='eval', description='Evaluate with a Classification TRT model.')

    parser.add_argument(
        '-i',
        '--image_dir',
        type=str,
        required=False,
        default=None,
        help='Input directory of images')
    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        required=True,
        help='Path to the Classification TensorRT engine.'
    )
    parser.add_argument(
        '-e',
        '--experiment_spec',
        type=str,
        required=True,
        help='Path to the experiment spec file.'
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        required=False,
        default=1,
        help='Batch size.')
    parser.add_argument(
        '-c',
        '--classmap',
        type=str,
        required=False,
        default=None,
        help='File with class mapping.'
    )
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
