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
from nvidia_tao_deploy.cv.detectnet_v2.proto.utils import load_proto
from nvidia_tao_deploy.cv.detectnet_v2.proto.postprocessing_config import build_postprocessing_config

from nvidia_tao_deploy.cv.detectnet_v2.dataloader import DetectNetKITTILoader
from nvidia_tao_deploy.cv.detectnet_v2.inferencer import DetectNetInferencer
from nvidia_tao_deploy.cv.detectnet_v2.postprocessor import BboxHandler

from nvidia_tao_deploy.metrics.kitti_metric import KITTIMetric


logger = logging.getLogger(__name__)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    level='INFO')


@monitor_status(name='detectnet_v2', mode='evaluate', hydra=False)
def main(args):
    """DetectNetv2 TRT evaluation."""
    experiment_spec = load_proto(args.experiment_spec)
    pproc_config = build_postprocessing_config(experiment_spec.postprocessing_config)

    # Load mapping_dict from the spec file
    mapping_dict = dict(experiment_spec.dataset_config.target_class_mapping)

    # Load target classes from label file
    target_classes = [target_class.name for target_class in experiment_spec.cost_function_config.target_classes]
    batch_size = args.batch_size if args.batch_size else experiment_spec.training_config.batch_size_per_gpu

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
        label_dirs=[args.label_dir],
        mapping_dict=mapping_dict,
        exclude_difficult=True,
        batch_size=batch_size,
        image_mean=None,
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype))

    bboxer = BboxHandler(batch_size=batch_size,
                         frame_height=h,
                         frame_width=w,
                         target_classes=target_classes,
                         postproc_classes=target_classes,
                         classwise_cluster_params=pproc_config,
                         )

    # Override class mapping with the class order specified by target_classes
    dl.classes = {c: i + 1 for i, c in enumerate(target_classes)}
    dl.class_mapping = {key.lower(): dl.classes[str(val.lower())]
                        for key, val in mapping_dict.items()}

    eval_metric = KITTIMetric(n_classes=len(dl.classes) + 1)
    gt_labels = []
    pred_labels = []
    for i, (imgs, labels) in tqdm(enumerate(dl), total=len(dl), desc="Producing predictions"):
        gt_labels.extend(labels)
        y_pred = trt_infer.infer(imgs)
        processed_inference = bboxer.bbox_preprocessing(y_pred)
        classwise_detections = bboxer.cluster_detections(processed_inference)
        y_pred_valid = bboxer.postprocess(classwise_detections, batch_size, dl.image_size[i], (w, h), dl.classes)
        pred_labels.extend(y_pred_valid)

    m_ap, ap = eval_metric(gt_labels, pred_labels, verbose=True)
    m_ap = np.mean(ap[1:])

    logging.info("*******************************")
    class_mapping = {v: k for k, v in dl.classes.items()}
    eval_results = {}
    for i in range(len(dl.classes)):
        eval_results['AP_' + class_mapping[i + 1]] = np.float64(ap[i + 1])
        logging.info("{:<14}{:<6}{}".format(class_mapping[i + 1], 'AP', round(ap[i + 1], 5)))  # noqa pylint: disable=C0209

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
        parser = argparse.ArgumentParser(prog='eval', description='Evaluate with a RetinaNet TRT model.')

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
