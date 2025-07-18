"""NMS on CPU"""
import numpy as np
from collections import namedtuple
from shapely.geometry import Polygon

Point2d = namedtuple('Point', ['x', 'y'])


def nms_cpu(
    box_scores,
    box_preds,
    nms_config,
    score_thresh
):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    src_box_scores = box_scores
    scores_mask = box_scores >= score_thresh
    box_scores = box_scores[scores_mask]
    box_preds = box_preds[scores_mask]
    selected = []
    if box_scores.shape[0] > 0:
        indices = np.argsort(box_scores)[::-1]
        box_scores_nms = box_scores[indices]
        boxes_for_nms = box_preds[indices]
        keep_idx = nms_cpu_core(boxes_for_nms[:, 0:7], box_scores_nms, nms_config.nms_thresh)
        selected = indices[keep_idx[:nms_config.nms_post_max_size]]

    original_idxs = np.nonzero(scores_mask)[0].reshape(-1)
    selected = original_idxs[selected]
    return selected, src_box_scores[selected]


def nms_cpu_core(boxes_for_nms, box_scores_nms, nms_thresh):
    """NMS CPU core"""
    suppressed = np.zeros(box_scores_nms.shape)
    keep_idx = []
    for k in range(boxes_for_nms.shape[0]):
        if suppressed[k] == 1:
            continue
        keep_idx.append(k)
        for j in range(k + 1, boxes_for_nms.shape[0]):
            if suppressed[j] == 1:
                continue
            sa = boxes_for_nms[k, 3] * boxes_for_nms[k, 4]
            sb = boxes_for_nms[j, 3] * boxes_for_nms[j, 4]
            s_overlap = box_overlap2(boxes_for_nms[k], boxes_for_nms[j])
            iou = s_overlap / max((sa + sb - s_overlap), 1e-6)
            if iou >= nms_thresh:
                suppressed[j] = 1
    return np.array(keep_idx, dtype=np.int32)


def rotate_around_center(center, angle_cos, angle_sin, p):
    """Roate around center."""
    new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x
    new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y
    return Point2d(new_x, new_y)


def box_overlap2(box_a, box_b):
    """Box overlap."""
    # (x, y, z, l, w, h, rt, id, score)
    a_angle = box_a[6]
    b_angle = box_b[6]
    a_dx_half = box_a[3] / 2
    b_dx_half = box_b[3] / 2
    a_dy_half = box_a[4] / 2
    b_dy_half = box_b[4] / 2
    a_x1 = box_a[0] - a_dx_half
    a_y1 = box_a[1] - a_dy_half
    a_x2 = box_a[0] + a_dx_half
    a_y2 = box_a[1] + a_dy_half
    b_x1 = box_b[0] - b_dx_half
    b_y1 = box_b[1] - b_dy_half
    b_x2 = box_b[0] + b_dx_half
    b_y2 = box_b[1] + b_dy_half
    box_a_corners = [Point2d(0, 0) for _ in range(5)]
    box_b_corners = [Point2d(0, 0) for _ in range(5)]
    center_a = Point2d(box_a[0], box_a[1])
    center_b = Point2d(box_b[0], box_b[1])

    box_a_corners[0] = Point2d(a_x1, a_y1)
    box_a_corners[1] = Point2d(a_x2, a_y1)
    box_a_corners[2] = Point2d(a_x2, a_y2)
    box_a_corners[3] = Point2d(a_x1, a_y2)

    box_b_corners[0] = Point2d(b_x1, b_y1)
    box_b_corners[1] = Point2d(b_x2, b_y1)
    box_b_corners[2] = Point2d(b_x2, b_y2)
    box_b_corners[3] = Point2d(b_x1, b_y2)

    a_angle_cos = np.cos(a_angle)
    a_angle_sin = np.sin(a_angle)
    b_angle_cos = np.cos(b_angle)
    b_angle_sin = np.sin(b_angle)

    for k in range(4):
        box_a_corners[k] = rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k])
        box_b_corners[k] = rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k])

    box_a_corners[4] = box_a_corners[0]
    box_b_corners[4] = box_b_corners[0]
    pa = Polygon(box_a_corners)
    pb = Polygon(box_b_corners)
    return pa.intersection(pb).area
