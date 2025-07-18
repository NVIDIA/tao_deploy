# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Generic PointPillars data loader."""
import copy
import os
import pickle

from pathlib import Path
import numpy as np

from nvidia_tao_deploy.utils.path_utils import expand_path
from nvidia_tao_deploy.cv.pointpillars.dataloader.utils import (
    keep_arrays_by_name,
    DataProcessor,
    PointFeatureEncoder,
)
from nvidia_tao_deploy.cv.pointpillars.utils import eval as kitti_eval
from nvidia_tao_deploy.cv.pointpillars.utils import object3d_general


def build_dataloader(dataset_cfg, class_names, root_path=None, info_path=None,
                     logger=None):
    """Build data loader."""
    dataset = GeneralPCDataset(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        info_path=info_path,
        training=False,
        logger=logger,
    )
    return dataset


class DatasetTemplate():
    """Dataset Template class."""

    def __init__(self, dataset_cfg=None, class_names=None,
                 training=True, root_path=None, info_path=None, logger=None):
        """Initialize."""
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(expand_path(self.dataset_cfg.data_path))
        self.info_path = Path(expand_path(info_path))
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.point_cloud_range, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.point_feature_encoding,
            point_cloud_range=self.point_cloud_range
        )
        self.data_processor = DataProcessor(
            self.dataset_cfg.data_processor,
            point_cloud_range=self.point_cloud_range,
            training=self.training,
            num_point_features=self.dataset_cfg.data_augmentor.aug_config_list[0].num_point_features
        )
        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size

    @property
    def mode(self):
        """mode of training."""
        return 'train' if self.training else 'test'

    def __getstate__(self):
        """Get state."""
        d = dict(self.__dict__)
        if "logger" in d:
            del d['logger']
        return d

    def __setstate__(self, d):
        """Set state."""
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def __len__(self):
        """Length of dataset."""
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if data_dict.get('gt_boxes', None) is not None:
            selected = keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )
        data_dict.pop('gt_names', None)

        return data_dict


class GeneralPCDataset(DatasetTemplate):
    """Generic data loader."""

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, info_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training,
            root_path=root_path, info_path=info_path, logger=logger
        )
        self.num_point_features = self.dataset_cfg.data_augmentor.aug_config_list[0].num_point_features
        self.split = self.dataset_cfg.data_split[self.mode]
        self.root_split_path = self.root_path / self.split
        lidar_path = self.root_split_path / "lidar"
        sample_id_list = os.listdir(lidar_path)
        assert len(sample_id_list), "lidar directory is empty"
        # strip .bin suffix
        self.sample_id_list = [x[:-4] for x in sample_id_list]
        for sid in self.sample_id_list:
            if len(self.get_label(sid)) == 0:
                raise IOError(
                    f"Got empty label for sample {sid} in {self.split} split"
                    ", please check the dataset"
                )
        self.infos = []
        self.include_data(self.mode)

    def include_data(self, mode):
        """Inlcude data files."""
        if self.logger is not None:
            self.logger.info('Loading point cloud dataset')
        pc_infos = []
        for info_path in self.dataset_cfg.info_path[mode]:
            info_path = self.info_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                pc_infos.extend(infos)
        self.infos.extend(pc_infos)
        if self.logger is not None:
            self.logger.info('Total samples for point cloud dataset: %d' % (len(pc_infos)))

    def set_split(self, split):
        """Setup train/val split."""
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, info_path=self.info_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / self.split
        lidar_path = self.root_split_path / "lidar"
        sample_id_list = []

        if os.path.isdir(lidar_path):
            sample_id_list = os.listdir(lidar_path)
        else:
            raise NotADirectoryError(f"{lidar_path} is not a directory")

        assert len(sample_id_list), "lidar directory is empty"
        # strip .bin suffix
        self.sample_id_list = [x[:-4] for x in sample_id_list]
        for sid in self.sample_id_list:
            if len(self.get_label(sid)) == 0:
                raise IOError(
                    f"Got empty label for sample {sid} in {split} split"
                    ", please check the dataset"
                )

    def get_lidar(self, idx):
        """Get LIDAR points."""
        lidar_file = self.root_split_path / 'lidar' / ('%s.bin' % idx)
        if not lidar_file.exists():
            raise FileNotFoundError(f"File not exist: {lidar_file}")
        points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, self.num_point_features)
        return points

    def get_label(self, idx):
        """Get KITTI labels."""
        label_file = self.root_split_path / 'label' / ('%s.txt' % idx)
        if not label_file.exists():
            raise FileNotFoundError(f"File not exist: {label_file}")
        return object3d_general.get_objects_from_label(label_file)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            """Get template for prediction result."""
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            """Get single prediction result."""
            pred_scores = box_dict['pred_scores']
            pred_boxes = box_dict['pred_boxes']
            pred_labels = box_dict['pred_labels']
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict
            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w', encoding="utf-8") as f:
                    box_lidar = single_pred_dict['boxes_lidar']
                    for idx in range(len(box_lidar)):
                        x, y, z, l, w, h, rt = box_lidar[idx]
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], 0,
                                 0, 0, 0, 0,
                                 h, w, l, x, y, z, rt,
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        """Evaluation of prediction results."""
        if 'annos' not in self.infos[0].keys():
            return None, {}
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]
        ap_result_str, ap_dict = kitti_eval.get_kitti_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        """Length."""
        return len(self.infos)

    def __getitem__(self, index):
        """Get item."""
        info = copy.deepcopy(self.infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        input_dict = {
            'points': points,
            'frame_id': sample_idx,
        }
        if 'annos' in info:
            annos = info['annos']
            gt_names = annos['name']
            gt_boxes_lidar = annos["gt_boxes_lidar"]
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
