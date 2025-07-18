"""Data loader utils."""
import numpy as np
from functools import partial


def keep_arrays_by_name(gt_names, used_classes):
    """Keep arrays by name."""
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def mask_points_by_range(points, limit_range):
    """Mask points by range."""
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
        & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


class PointFeatureEncoder(object):
    """Point Feature Encoder class."""

    def __init__(self, config, point_cloud_range=None):
        """Initialize."""
        super().__init__()
        self.point_encoding_config = config
        assert list(self.point_encoding_config.src_feature_list[0:3]) == ['x', 'y', 'z'], "src_feature_list[0:3] is not ['x', 'y', 'z']"
        self.used_feature_list = self.point_encoding_config.used_feature_list
        self.src_feature_list = self.point_encoding_config.src_feature_list
        self.point_cloud_range = point_cloud_range

    @property
    def num_point_features(self):
        """Number of point features."""
        return getattr(self, self.point_encoding_config.encoding_type)(points=None)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        func_map = {key: getattr(PointFeatureEncoder, key) for key in vars(PointFeatureEncoder) if not key.startswith("__") and key.endswith("_encoding")}
        if self.point_encoding_config.encoding_type in func_map:
            data_dict['points'], use_lead_xyz = func_map[self.point_encoding_config.encoding_type](self, data_dict['points'])
            data_dict['use_lead_xyz'] = use_lead_xyz
            return data_dict
        return None

    def absolute_coordinates_encoding(self, points=None):
        """Absolute coordinates encoding."""
        if points is None:
            num_output_features = len(self.used_feature_list)
            return num_output_features

        point_feature_list = [points[:, 0:3]]
        for x in self.used_feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.src_feature_list.index(x)
            point_feature_list.append(points[:, idx:idx + 1])
        point_features = np.concatenate(point_feature_list, axis=1)
        return point_features, True


class DataProcessor(object):
    """Data Processor."""

    def __init__(
        self, processor_configs,
        point_cloud_range, training,
        num_point_features
    ):
        """Initialize."""
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        self.num_point_features = num_point_features
        func_map = {key: getattr(DataProcessor, key) for key in vars(DataProcessor) if not key.startswith("__")}
        for cur_cfg in processor_configs:
            if cur_cfg.name in func_map:
                cur_processor = func_map[cur_cfg.name](self, config=cur_cfg)
                self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        """Mask points and boxes that are out of range."""
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        mask = mask_points_by_range(data_dict['points'], self.point_cloud_range)
        data_dict['points'] = data_dict['points'][mask]
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
