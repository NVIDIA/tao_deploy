# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""TAO Deploy Clustering Config."""

from nvidia_tao_deploy.cv.detectnet_v2.proto.postprocessing_config_pb2 import ClusteringConfig as ClusteringProto

CLUSTERING_ALGORITHM = {
    0: "dbscan",
    1: "nms",
    2: "hybrid"
}


def build_clustering_config(clustering_config):
    """Build ClusteringConfig from a proto.

    Args:
        clustering_config: clustering_config proto message.

    Returns:
        ClusteringConfig object.
    """
    return ClusteringConfig(clustering_config.coverage_threshold,
                            clustering_config.dbscan_eps,
                            clustering_config.dbscan_min_samples,
                            clustering_config.minimum_bounding_box_height,
                            clustering_config.clustering_algorithm,
                            clustering_config.nms_iou_threshold,
                            clustering_config.nms_confidence_threshold,
                            clustering_config.dbscan_confidence_threshold)


def build_clustering_proto(clustering_config):
    """Build proto from ClusteringConfig.

    Args:
        clustering_config: ClusteringConfig object.

    Returns:
        clustering_config: clustering_config proto message.
    """
    proto = ClusteringProto()
    proto.coverage_threshold = clustering_config.coverage_threshold
    proto.dbscan_eps = clustering_config.dbscan_eps
    proto.dbscan_min_samples = clustering_config.dbscan_min_samples
    proto.minimum_bounding_box_height = clustering_config.minimum_bounding_box_height
    proto.clustering_algorithm = clustering_config.clustering_algorithm
    proto.nms_iou_threshold = clustering_config.nms_iou_threshold
    proto.nms_confidence_threshold = clustering_config.nms_confidence_threshold
    proto.dbscan_confidence_threshold = clustering_config.dbscan_confidence_threshold
    return proto


class ClusteringConfig(object):
    """Hold the parameters for clustering detections."""

    def __init__(self, coverage_threshold, dbscan_eps, dbscan_min_samples,
                 minimum_bounding_box_height, clustering_algorithm,
                 nms_iou_threshold, nms_confidence_threshold,
                 dbscan_confidence_threshold):
        """Constructor.

        Args:
            coverage_threshold (float): Grid cells with coverage lower than this
                threshold will be ignored. Valid range [0.0, 1.0].
            dbscan_eps (float): DBSCAN eps parameter. The maximum distance between two samples
                for them to be considered as in the same neighborhood. Valid range [0.0, 1.0].
            dbscan_min_samples (float): DBSCAN min samples parameter. The number of samples (or
                total weight) in a neighborhood for a point to be considered as a core point.
                This includes the point itself. Must be >= 0.0.
            minimum_bounding_box_height (int): Minimum bbox height. Must be >= 0.
            clustering_algorithm (clustering_config.clustering_algorithm): The type of clustering
                algorithm.
            nms_iou_threshold (float): The iou threshold for NMS.
            dbscan_confidence_threshold (float): The dbscan confidence threshold.
            nms_confidence_threshold (float): The nms confidence threshold.

        Raises:
            ValueError: If the input arg is not within the accepted range.
        """
        if coverage_threshold < 0.0 or coverage_threshold > 1.0:
            raise ValueError("ClusteringConfig.coverage_threshold must be in [0.0, 1.0]")
        clustering_algorithm = CLUSTERING_ALGORITHM[clustering_algorithm]
        if clustering_algorithm not in ["dbscan", "nms"]:
            raise NotImplementedError(
                f"Invalid clustering algorithm: {clustering_algorithm}"
            )
        if clustering_algorithm == "dbscan":
            if dbscan_eps < 0.0 or dbscan_eps > 1.0:
                raise ValueError("ClusteringConfig.dbscan_eps must be in [0.0, 1.0]")
            if dbscan_min_samples < 0.0:
                raise ValueError("ClusteringConfig.dbscan_min_samples must be >= 0.0")
        if minimum_bounding_box_height < 0:
            raise ValueError(
                "ClusteringConfig.minimum_bounding_box_height must be >= 0"
            )
        if clustering_algorithm == "nms":
            if nms_iou_threshold < 0.0 or nms_iou_threshold > 1.0:
                raise ValueError(
                    "ClusteringConfig.nms_iou_threshold must be in [0.0, 1.0]"
                )

        self.coverage_threshold = coverage_threshold
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.dbscan_confidence_threshold = dbscan_confidence_threshold
        self.minimum_bounding_box_height = minimum_bounding_box_height
        self.clustering_algorithm = clustering_algorithm
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_confidence_threshold = nms_confidence_threshold
