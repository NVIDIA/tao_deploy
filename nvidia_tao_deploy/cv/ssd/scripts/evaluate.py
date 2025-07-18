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

import argparse
import os
import json
import numpy as np
import tensorrt as trt
from tqdm.auto import tqdm

import logging

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.ssd.dataloader import SSDKITTILoader
from nvidia_tao_deploy.cv.ssd.inferencer import SSDInferencer
from nvidia_tao_deploy.cv.ssd.proto.utils import load_proto
from nvidia_tao_deploy.metrics.kitti_metric import KITTIMetric


logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


@monitor_status(name='ssd', mode='evaluate', hydra=False)
def main(args):
    """SSD TRT evaluation."""
    # Load from proto-based spec file
    es = load_proto(args.experiment_spec)

    matching_iou_threshold = es.eval_config.matching_iou_threshold if es.eval_config.matching_iou_threshold else 0.5
    conf_thres = es.nms_config.confidence_threshold if es.nms_config.confidence_threshold else 0.01
    batch_size = args.batch_size if args.batch_size else es.eval_config.batch_size
    trt_infer = SSDInferencer(args.model_path, batch_size=batch_size)

    c, h, w = trt_infer.input_tensors[0].shape

    ap_mode = es.eval_config.average_precision_mode
    ap_mode_dict = {0: "sample", 1: "integrate"}

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

    if args.label_dir:
        label_dirs = [args.label_dir]
    else:
        label_dirs = [d.label_directory_path for d in es.dataset_config.validation_data_sources]

    # Load mapping_dict from the spec file
    mapping_dict = dict(es.dataset_config.target_class_mapping)

    dl = SSDKITTILoader(
        shape=(c, h, w),
        image_dirs=image_dirs,
        label_dirs=label_dirs,
        mapping_dict=mapping_dict,
        exclude_difficult=True,
        batch_size=batch_size,
        image_mean=img_mean,
        keep_aspect_ratio=False,
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype))

    eval_metric = KITTIMetric(n_classes=len(dl.classes) + 1,
                              matching_iou_threshold=matching_iou_threshold,
                              conf_thres=conf_thres,
                              average_precision_mode=ap_mode_dict[ap_mode])

    gt_labels = []
    pred_labels = []
    for i, (imgs, labels) in tqdm(enumerate(dl), total=len(dl), desc="Producing predictions"):
        gt_labels.extend(labels)

        y_pred = trt_infer.infer(imgs)
        for i in range(len(y_pred)):
            y_pred_valid = y_pred[i][y_pred[i][:, 1] > eval_metric.conf_thres]
            y_pred_valid[..., 2] = np.clip(y_pred_valid[..., 2].round(), 0.0, w)
            y_pred_valid[..., 3] = np.clip(y_pred_valid[..., 3].round(), 0.0, h)
            y_pred_valid[..., 4] = np.clip(y_pred_valid[..., 4].round(), 0.0, w)
            y_pred_valid[..., 5] = np.clip(y_pred_valid[..., 5].round(), 0.0, h)
            pred_labels.append(y_pred_valid)

    m_ap, ap = eval_metric(gt_labels, pred_labels, verbose=True)
    m_ap = np.mean(ap[1:])

    logging.info("*******************************")
    class_mapping = {v: k for k, v in dl.classes.items()}
    eval_results = {}
    for i in range(len(dl.classes)):
        eval_results['AP_' + class_mapping[i + 1]] = np.float64(ap[i + 1])
        logging.info("{:<14}{:<6}{}".format(class_mapping[i + 1], 'AP', round(ap[i + 1], 5)))  # noqa pylint: disable=C0209

    eval_results['mAP'] = round(m_ap, 3)
    logging.info("{:<14}{:<6}{}".format('', 'mAP', round(m_ap, 3)))  # noqa pylint: disable=C0209
    logging.info("*******************************")

    # Store evaluation results into JSON
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
        parser = argparse.ArgumentParser(prog='eval', description='Evaluate with a SSD TRT model.')

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
        help='Path to the SSD TensorRT engine.'
    )
    parser.add_argument(
        '-l',
        '--label_dir',
        type=str,
        required=False,
        help='Label directory.')
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        required=False,
        default=1,
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
