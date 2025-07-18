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

"""TAO Deploy Config Base Utilities."""

import os
import logging
import numpy as np
from google.protobuf.text_format import Merge as merge_text_proto
from nvidia_tao_deploy.cv.unet.proto.experiment_pb2 import Experiment

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


def load_proto(config):
    """Load the experiment proto."""
    proto = Experiment()

    def _load_from_file(filename, pb2):
        if not os.path.exists(filename):
            raise IOError(f"Specfile not found at: {filename}")
        with open(filename, "r", encoding="utf-8") as f:
            merge_text_proto(f.read(), pb2)
    _load_from_file(config, proto)

    return proto


def initialize_params(experiment_spec, phase="val"):
    """Initialization of the params object to the estimator runtime config.

    Args:
        experiment_spec: Loaded Unet Experiment spec.
        phase: Data source to load from.
    """
    training_config = experiment_spec.training_config
    dataset_config = experiment_spec.dataset_config
    model_config = experiment_spec.model_config

    target_classes = build_target_class_list(dataset_config.data_class_config)
    num_classes = get_num_unique_train_ids(target_classes)

    if not model_config.activation:
        activation = "softmax"
    elif model_config.activation == 'sigmoid' and num_classes > 2:
        logging.warning("Sigmoid activation can only be used for binary segmentation. \
                         Defaulting to softmax activation.")
        activation = 'softmax'
    elif model_config.activation == 'sigmoid' and num_classes == 2:
        num_classes = 1
        activation = model_config.activation
    else:
        activation = model_config.activation

    if phase == "val":
        data_sources = dataset_config.val_data_sources.data_source
        if data_sources:
            images_list = []
            masks_list = []
            for data_source in data_sources:
                image_path = data_source.image_path if data_source.image_path else None
                mask_path = data_source.masks_path if data_source.masks_path else None
                images_list.append(image_path)
                masks_list.append(mask_path)
        else:
            images_list = [dataset_config.val_images_path]
            masks_list = [dataset_config.val_masks_path]
    else:
        data_sources = dataset_config.train_data_sources.data_source
        if data_sources:
            images_list = []
            masks_list = []
            for data_source in data_sources:
                image_path = data_source.image_path if data_source.image_path else None
                mask_path = data_source.masks_path if data_source.masks_path else None
                images_list.append(image_path)
                masks_list.append(mask_path)
        else:
            images_list = [dataset_config.train_images_path]
            masks_list = [dataset_config.train_masks_path]

    return {
        'batch_size': training_config.batch_size if training_config.batch_size else 1,
        'resize_padding': dataset_config.resize_padding if
        dataset_config.resize_padding else False,
        'resize_method': dataset_config.resize_method.upper() if
        dataset_config.resize_method else 'BILINEAR',
        'activation': activation,
        'augment': dataset_config.augment if dataset_config.augment else False,
        'filter_data': dataset_config.filter_data if dataset_config.filter_data else False,
        'num_classes': num_classes,
        'num_conf_mat_classes': num_classes,
        'train_id_name_mapping': get_train_class_mapping(target_classes),
        'label_id_train_id_mapping': get_label_train_dic(target_classes),
        'preprocess': dataset_config.preprocess if dataset_config.preprocess else "min_max_-1_1",
        'input_image_type': dataset_config.input_image_type if dataset_config.input_image_type else "color",
        'images_list': images_list,
        'masks_list': masks_list,
        'arch': model_config.arch if model_config.arch else "resnet",
        'enable_qat': model_config.enable_qat if model_config.enable_qat else False,
    }


def get_label_train_dic(target_classes):
    """Function to get mapping between class and train ids."""
    label_train_dic = {}
    for target in target_classes:
        label_train_dic[target.label_id] = target.train_id

    return label_train_dic


def get_train_class_mapping(target_classes):
    """Utility function that returns the mapping of the train id to orig class."""
    train_id_name_mapping = {}
    for target_class in target_classes:
        if target_class.train_id not in train_id_name_mapping.keys():
            train_id_name_mapping[target_class.train_id] = [target_class.name]
        else:
            train_id_name_mapping[target_class.train_id].append(target_class.name)
    return train_id_name_mapping


def get_num_unique_train_ids(target_classes):
    """Return the final number classes used for training.

    Arguments:
        target_classes: The target classes object that contain the train_id and
        label_id.
    Returns:
        Number of classes to be segmented.
    """
    train_ids = [target.train_id for target in target_classes]
    train_ids = np.array(train_ids)
    train_ids_unique = np.unique(train_ids)
    return len(train_ids_unique)


def build_target_class_list(data_class_config):
    """Build a list of TargetClasses based on proto.

    Arguments:
        cost_function_config: CostFunctionConfig.
    Returns:
        A list of TargetClass instances.
    """
    target_classes = []
    orig_class_label_id_map = {}
    for target_class in data_class_config.target_classes:
        orig_class_label_id_map[target_class.name] = target_class.label_id

    class_label_id_calibrated_map = orig_class_label_id_map.copy()
    for target_class in data_class_config.target_classes:
        label_name = target_class.name
        train_name = target_class.mapping_class
        class_label_id_calibrated_map[label_name] = orig_class_label_id_map[train_name]

    train_ids = sorted(list(set(class_label_id_calibrated_map.values())))
    train_id_calibrated_map = {}
    for idx, tr_id in enumerate(train_ids):
        train_id_calibrated_map[tr_id] = idx

    class_train_id_calibrated_map = {}
    for label_name, train_id in class_label_id_calibrated_map.items():
        class_train_id_calibrated_map[label_name] = train_id_calibrated_map[train_id]

    for target_class in data_class_config.target_classes:
        target_classes.append(
            TargetClass(target_class.name, label_id=target_class.label_id,
                        train_id=class_train_id_calibrated_map[target_class.name]))

    for target_class in target_classes:
        logging.debug("Label Id %d: Train Id %d", target_class.label_id, target_class.train_id)

    return target_classes


class TargetClass(object):
    """Target class parameters."""

    def __init__(self, name, label_id, train_id=None):
        """Constructor.

        Args:
            name (str): Name of the target class.
            label_id (str):original label id of every pixel of the mask
            train_id (str): The mapped train id of every pixel in the mask
        Raises:
            ValueError: On invalid input args.
        """
        self.name = name
        self.train_id = train_id
        self.label_id = label_id
