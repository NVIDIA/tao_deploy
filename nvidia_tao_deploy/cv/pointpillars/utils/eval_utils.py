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

"""Evaluation utils for PointPillars."""
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt


def statistics_info(cfg, ret_dict, metric, disp_dict):
    """Statistics infomation."""
    for cur_thresh in cfg.model.post_processing.recall_thresh_list:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.model.post_processing.recall_thresh_list[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.
    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """
    assert view.shape[0] <= 4, "view.shape[0] should be no more than 4"
    assert view.shape[1] <= 4, "view.shape[1] should be no more than 4"
    assert points.shape[0] == 3, "points.shape[0] should be equal to 3"
    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view
    nbr_points = points.shape[1]
    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]
    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
    return points


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])
    rot_matrix = np.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    template = np.array((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2
    corners3d = np.tile(boxes3d[:, None, 3:6], (1, 8, 1)) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.reshape((-1, 8, 3)), boxes3d[:, 6]).reshape((-1, 8, 3))
    corners3d += boxes3d[:, None, 0:3]

    return corners3d


def draw_box(box, axis, view, colors, linewidth):
    """Draw box."""
    # box: (3, 4), append first point to form a loop
    x = np.concatenate((box[0, :], box[0, :1]), axis=-1)
    y = np.concatenate((box[1, :], box[1, :1]), axis=-1)
    axis.plot(
        x, y,
        color=colors[0],
        linewidth=linewidth
    )


def visual(points, gt_anno, det, det_scores, frame_id, eval_range=35, conf_th=0.1):
    """Visualization."""
    _, ax = plt.subplots(1, 1, figsize=(9, 9), dpi=200)
    # points
    points = view_points(points[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / eval_range)
    ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)
    # (B, 8, 3)
    boxes_gt = boxes_to_corners_3d(gt_anno)
    # Show GT boxes.
    for box in boxes_gt:
        # (8, 3)
        bev = box[4:, :]
        bev = view_points(bev.transpose(), np.eye(4), normalize=False)
        draw_box(bev, ax, view=np.eye(4), colors=('r', 'r', 'r'), linewidth=2)
    # Show EST boxes.
    if len(det) == 0:
        plt.axis('off')
        plt.savefig(frame_id + ".png")
        plt.close()
        return
    boxes_est = boxes_to_corners_3d(det)
    for idx, box in enumerate(boxes_est):
        if det_scores[idx] < conf_th:
            continue
        bev = box[4:, :]
        bev = view_points(bev.transpose(), np.eye(4), normalize=False)
        draw_box(bev, ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=1)
    axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    plt.axis('off')
    plt.savefig(frame_id + ".png")
    plt.close()


def sparse_to_dense(points, batch_size):
    """Convert sparse points to dense format."""
    points_dense = []
    num_points_dense = []
    points_per_frame = np.copy(points)
    num_points_ = points_per_frame.shape[0]
    points_dense.append(points_per_frame)
    num_points_dense.append(num_points_)
    return points_dense, num_points_dense


def eval_one_epoch_trt(cfg, model, dataloader, save_to_file=False, result_dir=None):
    """Do evaluation on one epoch with TensorRT engine."""
    result_dict = {}
    result_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir = result_dir / "detected_labels"
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)
    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.model.post_processing.recall_thresh_list:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0
    dataset = dataloader
    class_names = dataset.class_names
    det_annos = []
    print('*************** EVALUATION *****************')
    total_time = 0
    start_time = time.time()
    for _, batch_dict in enumerate(dataloader):
        points = batch_dict['points']
        batch_size = 1
        start = time.time()
        points_np, num_points_np = sparse_to_dense(points, batch_size)
        # Do infer
        outputs_final = model(
            {
                "points": points_np,
                "num_points": num_points_np,
            }
        )
        end = time.time()
        total_time += end - start
        pred_dicts = []
        for output_final in outputs_final:
            pred_dict = {'pred_boxes': [], 'pred_scores': [], 'pred_labels': []}
            for box in output_final:
                if box[-1] > -0.5:
                    pred_dict['pred_boxes'].append(box[:7])
                    pred_dict['pred_scores'].append(np.array([box[7]])[0])
                    pred_dict['pred_labels'].append(np.array([box[8]])[0])
                else:
                    break
            if len(pred_dict['pred_boxes']) > 0:
                pred_dict['pred_boxes'] = np.stack(pred_dict['pred_boxes'])
                pred_dict['pred_scores'] = np.stack(pred_dict['pred_scores'])
                pred_dict['pred_labels'] = (np.stack(pred_dict['pred_labels']) + 0.01).astype(np.int32)
            else:
                pred_dict['pred_boxes'] = np.zeros((0, 7))
                pred_dict['pred_scores'] = np.zeros((0, ))
                pred_dict['pred_labels'] = np.zeros((0,))
            pred_dicts.append(pred_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
    print('*************** Performance *****************')
    sec_per_example = (time.time() - start_time) / len(dataset)
    print('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)
    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.model.post_processing.recall_thresh_list:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        print('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        print('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += len(anno['name'])
    print('Average predicted number of objects(%d samples): %.3f'
          % (len(det_annos), total_pred_objects / max(1, len(det_annos))))
    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)
    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.model.post_processing.eval_metric,
        output_path=final_output_dir
    )
    print(result_str)
    print('**********Eval time per frame: %.3f ms**********' % (total_time / len(dataloader) * 1000))
    print('Result is save to %s' % result_dir)
    print('****************Evaluation done.*****************')
    return result_dict


def infer_one_epoch_trt(cfg, model, dataloader, save_to_file=False, result_dir=None):
    """Do inference on one epoch with TensorRT engine."""
    result_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir = result_dir / "detected_labels"
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)
        image_output_dir = result_dir / "detected_boxes"
        image_output_dir.mkdir(parents=True, exist_ok=True)
    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.model.post_processing.recall_thresh_list:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0
    dataset = dataloader
    class_names = dataset.class_names
    det_annos = []
    print('*************** INFERENCE *****************')
    total_time = 0
    for _, batch_dict in enumerate(dataloader):
        points = batch_dict['points']
        batch_size = 1
        start = time.time()
        points_np, num_points_np = sparse_to_dense(points, batch_size)
        # Do infer
        outputs_final = model(
            {
                "points": points_np,
                "num_points": num_points_np,
            }
        )
        end = time.time()
        total_time += end - start
        pred_dicts = []
        for output_final in outputs_final:
            pred_dict = {'pred_boxes': [], 'pred_scores': [], 'pred_labels': []}
            for box in output_final:
                if box[-1] > -0.5:
                    pred_dict['pred_boxes'].append(box[:7])
                    pred_dict['pred_scores'].append(np.array([box[7]])[0])
                    pred_dict['pred_labels'].append(np.array([box[8]])[0])
                else:
                    break
            if len(pred_dict['pred_boxes']) > 0:
                pred_dict['pred_boxes'] = np.stack(pred_dict['pred_boxes'])
                pred_dict['pred_scores'] = np.stack(pred_dict['pred_scores'])
                pred_dict['pred_labels'] = (np.stack(pred_dict['pred_labels']) + 0.01).astype(np.int32)
            else:
                pred_dict['pred_boxes'] = np.zeros((0, 7))
                pred_dict['pred_scores'] = np.zeros((0, ))
                pred_dict['pred_labels'] = np.zeros((0,))
            pred_dicts.append(pred_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        for pdi, pd in enumerate(pred_dicts):
            visual(
                points_np[pdi].transpose(),
                batch_dict["gt_boxes"][:, 0:7],
                pd['pred_boxes'],
                pd['pred_scores'],
                str(image_output_dir / batch_dict['frame_id']),
                eval_range=60,
                conf_th=cfg.inference.viz_conf_thresh
            )
    ret_dict = {}
    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)
    print('Result is save to %s' % result_dir)
    print('****************Inference done.*****************')
    return ret_dict
