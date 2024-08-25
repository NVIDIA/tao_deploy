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

"""Utility functions to be used for CenterPose."""

import os
import cv2
import json
import logging
import numpy as np
from enum import IntEnum
from pyrr import Quaternion
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


def get_dir(src_point, rot_rad):
    """Get the direction of the keypoints"""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    """Get the 3rd keypoints"""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    """Get the affine transform through the parameters"""
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    """Affine transformation"""
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size):
    """Affine transformation through the transform matrix"""
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        if coords[p, 0] == -10000 and coords[p, 1] == -10000:
            # Still give it a very small number
            target_coords[p, 0:2] = [-10000, -10000]
        else:
            target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def soft_nms(src_boxes, sigma=0.5, Nt=0.3, threshold=0.001, method=0):
    """Soft non-maximum Suppression for bounding boxes"""
    N = src_boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0

    for i in range(N):
        maxscore = src_boxes[i]['score']
        maxpos = i

        tx1 = src_boxes[i]['bbox'][0]
        ty1 = src_boxes[i]['bbox'][1]
        tx2 = src_boxes[i]['bbox'][2]
        ty2 = src_boxes[i]['bbox'][3]
        ts = src_boxes[i]['score']

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < src_boxes[pos]['score']:
                maxscore = src_boxes[pos]['score']
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        src_boxes[i]['bbox'] = src_boxes[maxpos]['bbox']
        src_boxes[i]['score'] = src_boxes[maxpos]['score']

        # swap ith box with position of max box
        src_boxes[maxpos]['bbox'] = [tx1, ty1, tx2, ty2]
        src_boxes[maxpos]['score'] = ts

        for key in src_boxes[0]:
            if key not in ('bbox', 'score'):
                tmp = src_boxes[i][key]
                src_boxes[i][key] = src_boxes[maxpos][key]
                src_boxes[maxpos][key] = tmp

        tx1 = src_boxes[i]['bbox'][0]
        ty1 = src_boxes[i]['bbox'][1]
        tx2 = src_boxes[i]['bbox'][2]
        ty2 = src_boxes[i]['bbox'][3]
        ts = src_boxes[i]['score']

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:

            x1 = src_boxes[pos]['bbox'][0]
            y1 = src_boxes[pos]['bbox'][1]
            x2 = src_boxes[pos]['bbox'][2]
            y2 = src_boxes[pos]['bbox'][3]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # iou between max box and detection box

                    if method == 1:  # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov) / sigma)
                    else:  # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    src_boxes[pos]['score'] = weight * src_boxes[pos]['score']

                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if src_boxes[pos]['score'] < threshold:

                        src_boxes[pos]['bbox'] = src_boxes[N - 1]['bbox']
                        src_boxes[pos]['score'] = src_boxes[N - 1]['score']

                        for key in src_boxes[0]:
                            if key not in ('bbox', 'score'):
                                tmp = src_boxes[pos][key]
                                src_boxes[pos][key] = src_boxes[N - 1][key]
                                src_boxes[N - 1][key] = tmp

                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = list(range(N))
    return keep


def transform_outputs(dets, c, s, output_res):
    """This module transform the outputs to the format expected by pnp solver.

    Args:
        dets: detection results
        c: principle points
        s: the maximum axis of the orginal images
    """
    w = output_res
    h = output_res

    # Scale bbox & pts and Regroup
    if not ('scores' in dets):
        return [[{}]]

    ret = []

    for i in range(dets['scores'].shape[0]):

        preds = []

        for j in range(len(dets['scores'][i])):
            item = {}
            item['score'] = float(dets['scores'][i][j])
            item['cls'] = int(dets['clses'][i][j])
            item['obj_scale'] = dets['obj_scale'][i][j]

            # from w,h to c[i], s[i]
            bbox = transform_preds(dets['bboxes'][i, j].reshape(-1, 2), c[i], s[i], (w, h))
            item['bbox'] = bbox.reshape(-1, 4).flatten()

            item['ct'] = [(item['bbox'][0] + item['bbox'][2]) / 2, (item['bbox'][1] + item['bbox'][3]) / 2]

            kps = transform_preds(dets['kps'][i, j].reshape(-1, 2), c[i], s[i], (w, h))
            item['kps'] = kps.reshape(-1, 16).flatten()

            kps_displacement_mean = transform_preds(dets['kps_displacement_mean'][i, j].reshape(-1, 2), c[i], s[i],
                                                    (w, h))
            item['kps_displacement_mean'] = kps_displacement_mean.reshape(-1, 16).flatten()

            kps_heatmap_mean = transform_preds(dets['kps_heatmap_mean'][i, j].reshape(-1, 2), c[i], s[i], (w, h))
            item['kps_heatmap_mean'] = kps_heatmap_mean.reshape(-1, 16).flatten()

            preds.append(item)

        ret.append(preds)

    return ret


def merge_outputs(detections, vis_threshold=0.3, nms=True):
    """This module group all the detection result from different scales on a single image. Merge the detection results according to the score and nms.

    Args:
        detections: detection results of the model
        nms: Non-maximum Suppression for removing the redundunt bbox
    """
    ret = []
    for k in range(len(detections)):
        results = []
        for det in detections[k]:
            if det['score'] > vis_threshold:
                results.append(det)
        results = np.array(results)
        if nms:
            keep = soft_nms(results, Nt=0.5, method=2, threshold=vis_threshold)
            results = results[keep]
        ret.append(results)
    return ret


def pnp_shell(points_filtered, scale, cam_intrinsic, opencv_return=True):
    """Initialize a 3D cuboid and process the PnP calcualation to get the 2D/3D keypoints"""
    # Initial a 3d cuboid
    cuboid3d = Cuboid3d(1 * np.array(scale))

    pnp_solver = \
        CuboidPNPSolver(
            cuboid3d=cuboid3d
        )
    pnp_solver.set_camera_intrinsic_matrix(cam_intrinsic)

    # Process the 3D cuboid, 2D keypoints and intrinsic matrix to solve the pnp
    location, quaternion, projected_points, _ = pnp_solver.solve_pnp(
        points_filtered, opencv_return=opencv_return)

    # Calculate the actual 3D keypoints by using the location and quaternion from pnp solver
    if location is not None:

        ori = R.from_quat(quaternion).as_matrix()
        pose_pred = np.identity(4)
        pose_pred[:3, :3] = ori
        pose_pred[:3, 3] = location
        point_3d_obj = cuboid3d.get_vertices()

        point_3d_cam = pose_pred @ np.hstack(
            (np.array(point_3d_obj), np.ones((np.array(point_3d_obj).shape[0], 1)))).T

        point_3d_cam = point_3d_cam[:3, :].T

        # Add the centroid
        point_3d_cam = np.insert(point_3d_cam, 0, np.mean(point_3d_cam, axis=0), axis=0)

        # Add the center
        projected_points = np.insert(projected_points, 0, np.mean(projected_points, axis=0), axis=0)

        return projected_points, point_3d_cam, location, quaternion

    return None


def add_obj_order(img, keypoints2d):
    """Draw the 2D keypoints on the image"""
    bbox = np.array(keypoints2d, dtype=np.int32)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(bbox)):
        txt = '{:d}'.format(i)
        cat_size = cv2.getTextSize(txt, font, 1, 2)[0]
        cv2.putText(img, txt, (bbox[i][0], bbox[i][1] + cat_size[1]),
                    font, 1, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)


def add_axes(img, box, cam_intrinsic, axis_size=1):
    """Draw the 6-DoF on the image"""
    # box 9x3 array
    # OpenCV way
    N = axis_size
    # Centroid, top, front, right
    axes_point_list = [0, box[3] - box[1], box[2] - box[1], box[5] - box[1]]

    viewport_point_list = []
    for axes_point in axes_point_list:
        vector = axes_point
        vector = vector / np.linalg.norm(vector) * N if np.linalg.norm(vector) != 0 else 0
        vector = vector + box[0]
        vector = vector.flatten()

        k_3d = np.array([vector[0], vector[1], vector[2]])
        pp = np.matmul(cam_intrinsic, k_3d.reshape(3, 1))
        viewport_point = [pp[0] / pp[2], pp[1] / pp[2]]
        viewport_point_list.append((int(viewport_point[0]), int(viewport_point[1])))

    # BGR space
    cv2.line(img, viewport_point_list[0], viewport_point_list[1], (0, 255, 0), 5)  # y-> green
    cv2.line(img, viewport_point_list[0], viewport_point_list[2], (255, 0, 0), 5)  # z-> blue
    cv2.line(img, viewport_point_list[0], viewport_point_list[3], (0, 0, 255), 5)  # x-> red


def add_coco_hp(img, points):
    """Draw the projected 3D Bounding Box on the image"""
    # objectron
    edges = [[2, 4], [2, 6], [6, 8], [4, 8],
             [1, 2], [3, 4], [5, 6], [7, 8],
             [1, 3], [1, 5], [3, 7], [5, 7]]

    num_joints = 8
    points = np.array(points, dtype=np.int32).reshape(num_joints, 2)
    # Draw edges
    for e in edges:
        temp = [e[0] - 1, e[1] - 1]
        edge_color = (0, 0, 255)  # bgr
        cv2.line(img, (points[temp[0], 0], points[temp[0], 1]),
                      (points[temp[1], 0], points[temp[1], 1]), edge_color, 2)


def add_obj_scale(img, bbox, scale):
    """Draw the relative dimension numbers in a small region"""
    bbox = np.array(bbox, dtype=np.int32)
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt = '{:.3f}/{:.3f}/{:.3f}'.format(scale[0], scale[1], scale[2])
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    cv2.rectangle(img,
                  (bbox[0], bbox[1] + 2),
                  (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 6), (0, 0, 0), -1)
    cv2.putText(img, txt, (bbox[0], bbox[1] + cat_size[1]),
                font, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


def rotation_y_matrix(theta):
    """Rotate the bounding box along with the y axis"""
    M_R = np.array([[np.cos(theta), 0, np.sin(theta), 0],
                    [0, 1, 0, 0],
                    [-np.sin(theta), 0, np.cos(theta), 0], [0, 0, 0, 1]])
    return M_R


def safe_divide(i1, i2):
    """Get the average percision"""
    divisor = float(i2) if i2 > 0 else 1e-6
    return i1 / divisor


def save_inference_prediction(outputs, output_dir, img_path, infer_config):
    """Save the visualization results to the required folder"""
    # Camera Intrinsic matrix
    cx = infer_config['principle_point_x']
    cy = infer_config['principle_point_y']
    fx = infer_config['focal_length_x']
    fy = infer_config['focal_length_y']
    skew = infer_config['skew']
    axis_size = infer_config['axis_size']
    cam_intrinsic = np.array([[fx, skew, cx], [0, fy, cy], [0, 0, 1]])

    save_json = infer_config['save_json']
    save_visualization = infer_config['save_visualization']

    for idx in range(len(img_path)):
        out = outputs[idx]
        img = cv2.imread(img_path[idx])
        _, tail = os.path.split(img_path[idx])

        output_image_name = os.path.join(output_dir, tail)
        output_json_name = os.path.join(output_dir, os.path.splitext(tail)[0] + '.json')

        dict_results = {'image_name': tail, "objects": []}
        for k in range(len(out['projected_points'])):

            projected_points = out['projected_points'][k]
            point_3d_cam = out['point_3d_cam'][k]
            bbox = out['bbox'][k]
            obj_scale = out['obj_scale'][k]
            obj_translations = out['location'][k]
            obj_rotations = out['quaternion'][k]
            keypoints_2d = out['keypoints_2d'][k]

            if save_json:
                if obj_translations is not None:
                    dict_obj = {
                        'id': f'object_{k}',
                        'location': obj_translations,
                        'quaternion_xyzw': obj_rotations,
                        'projected_keypoints_2d': projected_points.tolist(),
                        'keypoints_3d': point_3d_cam.tolist(),
                        'relative_scale': obj_scale.tolist(),
                        'keypoints_2d': keypoints_2d.tolist()}
                else:
                    dict_obj = {
                        'id': f'object_{k}',
                        'location': [],
                        'quaternion_xyzw': [],
                        'projected_keypoints_2d': [],
                        'keypoints_3d': [],
                        'relative_scale': obj_scale.tolist(),
                        'keypoints_2d': keypoints_2d.tolist()}
                dict_results['objects'].append(dict_obj)

            if save_visualization is True and obj_translations is not None:
                # visualize the points order
                add_obj_order(img, projected_points)
                # visualize the bounding box
                add_coco_hp(img, projected_points[1:])
                # visualize the 6-DoF
                add_axes(img, point_3d_cam, cam_intrinsic, axis_size)
                # visualize the relative dimension of the object
                add_obj_scale(img, bbox, obj_scale)

        if save_visualization is True:
            cv2.imwrite(output_image_name, img)

        if save_json is True:
            with open(output_json_name, 'w+', encoding='UTF-8') as fp:
                json.dump(dict_results, fp, indent=4, sort_keys=True)


class CuboidVertexType(IntEnum):
    """This class contains a 3D cuboid vertex type"""

    FrontTopRight = 0
    FrontTopLeft = 1
    FrontBottomLeft = 2
    FrontBottomRight = 3
    RearTopRight = 4
    RearTopLeft = 5
    RearBottomLeft = 6
    RearBottomRight = 7
    Center = 8
    TotalCornerVertexCount = 8
    TotalVertexCount = 9


class Cuboid3d():
    """This class initialize a 3D cuboid according to the scale."""

    def __init__(self, size3d=[1.0, 1.0, 1.0],
                 coord_system=None, parent_object=None):
        """This local coordinate system is similar to the intrinsic transform matrix of a 3d object. Create a box with a certain size."""
        self.center_location = [0, 0, 0]
        self.coord_system = coord_system
        self.size3d = size3d
        self._vertices = [0, 0, 0] * CuboidVertexType.TotalCornerVertexCount
        self.generate_vertexes()

    def get_vertex(self, vertex_type):
        """Returns the location of a vertex.

        Args:
            vertex_type: enum of type CuboidVertexType

        Returns:
            Numpy array(3) - Location of the vertex type in the cuboid
        """
        return self._vertices[vertex_type]

    def get_vertices(self):
        """Return the 3D cuboid vertices"""
        return self._vertices

    def generate_vertexes(self):
        """Generate the 3D cuboid vertices"""
        width, height, depth = self.size3d

        # By default use the normal OpenCV coordinate system
        if (self.coord_system is None):
            cx, cy, cz = self.center_location
            # X axis point to the right
            right = cx + width / 2.0
            left = cx - width / 2.0
            # Y axis point upward
            top = cy + height / 2.0
            bottom = cy - height / 2.0
            # Z axis point forward
            front = cz + depth / 2.0
            rear = cz - depth / 2.0

            # List of 8 vertices of the box
            self._vertices = [
                # self.center_location,   # Center
                [left, bottom, rear],  # Rear Bottom Left
                [left, bottom, front],  # Front Bottom Left
                [left, top, rear],  # Rear Top Left
                [left, top, front],  # Front Top Left

                [right, bottom, rear],  # Rear Bottom Right
                [right, bottom, front],  # Front Bottom Right
                [right, top, rear],  # Rear Top Right
                [right, top, front],  # Front Top Right

            ]


class CuboidPNPSolver(object):
    """This class is used to find the 6-DoF pose of a cuboid given its projected vertices. Runs perspective-n-point (PNP) algorithm."""

    # Class variables
    cv2version = cv2.__version__.split('.')
    cv2majorversion = int(cv2version[0])

    def __init__(self, scaling_factor=1,
                 camera_intrinsic_matrix=None,
                 cuboid3d=None,
                 dist_coeffs=np.zeros((4, 1)),
                 min_required_points=4
                 ):
        """Initialize the 3D cuboid and camera parameters"""
        self.min_required_points = max(4, min_required_points)
        self.scaling_factor = scaling_factor

        if camera_intrinsic_matrix is not None:
            self._camera_intrinsic_matrix = camera_intrinsic_matrix
        else:
            self._camera_intrinsic_matrix = np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ])
        self._cuboid3d = cuboid3d

        self._dist_coeffs = dist_coeffs

    def set_camera_intrinsic_matrix(self, new_intrinsic_matrix):
        """Sets the camera intrinsic matrix"""
        self._camera_intrinsic_matrix = new_intrinsic_matrix

    def set_dist_coeffs(self, dist_coeffs):
        """Sets the camera intrinsic matrix"""
        self._dist_coeffs = dist_coeffs

    def solve_pnp(self, cuboid2d_points, pnp_algorithm=cv2.SOLVEPNP_ITERATIVE, opencv_return=True):
        """Detects the rotation and traslation of a cuboid object from its vertexes' 2D location in the image

        Inputs:
        - cuboid2d_points:  list of XY tuples
        - pnp_algorithm: algorithm of the Perspective-n-Point (PnP) pose computation
        - opencv_return: if ture, return the OpenCV coordinate; else, return the OpenGL coordinate.
        OpenCV coordiate is used to demo the visualization results and the OpenGL coordinate is used to calculate the 3D IoU.

        Outputs:
        - location in 3D
        - pose in 3D (as quaternion)
        - projected points:  np.ndarray of np.ndarrays

        """
        location = None
        quaternion = None
        location_new = None
        quaternion_new = None
        loc = None
        quat = None
        reprojectionError = None
        projected_points = cuboid2d_points
        cuboid3d_points = np.array(self._cuboid3d.get_vertices())

        obj_2d_points = []
        obj_3d_points = []

        # 8*n points
        for i in range(len(cuboid2d_points)):
            check_point_2d = cuboid2d_points[i]
            # Ignore invalid points
            if (check_point_2d is None or check_point_2d[0] < -5000 or check_point_2d[1] < -5000):
                continue
            obj_2d_points.append(check_point_2d)
            obj_3d_points.append(cuboid3d_points[int(i // (len(cuboid2d_points) / CuboidVertexType.TotalCornerVertexCount))])

        obj_2d_points = np.array(obj_2d_points, dtype=float)
        obj_3d_points = np.array(obj_3d_points, dtype=float)
        valid_point_count = len(obj_2d_points)

        # Can only do PNP if we have more than 3 valid points
        is_points_valid = valid_point_count >= self.min_required_points

        if is_points_valid:

            # Heatmap representation may have less than 6 points, in which case we have to use another pnp algorithm
            if valid_point_count < 6:
                pnp_algorithm = cv2.SOLVEPNP_EPNP

            # Usually, we use this one
            ret, rvec, tvec, reprojectionError = cv2.solvePnPGeneric(
                obj_3d_points,
                obj_2d_points,
                self._camera_intrinsic_matrix,
                self._dist_coeffs,
                flags=pnp_algorithm
            )

            if ret:

                rvec = np.array(rvec[0])
                tvec = np.array(tvec[0])

                reprojectionError = reprojectionError.flatten()[0]

                # Convert OpenCV coordinate system to OpenGL coordinate system
                transformation = np.identity(4)
                r = R.from_rotvec(rvec.reshape(1, 3))
                transformation[:3, :3] = r.as_matrix()
                transformation[:3, 3] = tvec.reshape(1, 3)
                M = np.zeros((4, 4))
                M[0, 1] = 1
                M[1, 0] = 1
                M[3, 3] = 1
                M[2, 2] = -1
                transformation = np.matmul(M, transformation)

                rvec_new = R.from_matrix(transformation[:3, :3]).as_rotvec()
                tvec_new = transformation[:3, 3]

                # OpenGL result, to be compared against GT
                location_new = list(x for x in tvec_new)
                quaternion_new = list(self.convert_rvec_to_quaternion(rvec_new))

                # OpenCV result
                location = list(x[0] for x in tvec)
                quaternion = list(self.convert_rvec_to_quaternion(rvec))

                # Still use OpenCV way to project 3D points
                projected_points, _ = cv2.projectPoints(cuboid3d_points, rvec, tvec, self._camera_intrinsic_matrix,
                                                        self._dist_coeffs)

                projected_points = np.squeeze(projected_points)

                # Currently, we assume pnp fails if z<0
                _, _, z = location
                if z < 0:
                    location = None
                    quaternion = None
                    location_new = None
                    quaternion_new = None

                    logger.debug("PNP solution is behind the camera (Z < 0) => Fail")
                else:
                    logger.debug("solvePNP found good results - location: {} - rotation: {} !!!".format(location, quaternion))
            else:
                logger.debug('solvePnP return false')
        else:
            logger.debug("Need at least 4 valid points in order to run PNP. Currently: {}".format(valid_point_count))

        if opencv_return:
            # Return OpenCV result for demo
            loc = location
            quat = quaternion
        else:
            # Return OpenGL result for eval
            loc = location_new
            quat = quaternion_new
        return loc, quat, projected_points, reprojectionError

    def convert_rvec_to_quaternion(self, rvec):
        """Convert rvec (which is log quaternion) to quaternion"""
        theta = np.sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2])  # in radians
        raxis = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta]

        # pyrr's Quaternion (order is XYZW), https://pyrr.readthedocs.io/en/latest/oo_api_quaternion.html
        return Quaternion.from_axis_rotation(raxis, theta)


class PnPProcess:
    """This module is to get 2d projection of keypoints & 6-DoF & 3d keypoint in camera frame."""

    def __init__(self, config, evaluate=False):
        """PostProcess constructor.

        Args:
            config: configuration file that includes the camera calibration info.
            camera_intrinsic : camera intrinsic matrix used for the pnp solver
        """
        super().__init__()
        self.evaluate = evaluate
        if not self.evaluate:
            cx = config['principle_point_x']
            cy = config['principle_point_y']
            fx = config['focal_length_x']
            fy = config['focal_length_y']
            skew = config['skew']
            self.camera_intrinsic = np.array([[fx, skew, cx], [0, fy, cy], [0, 0, 1]])

        self.opencv = config.opencv

    def set_intrinsic_matrix(self, intrinsic):
        """Set the intrinsic matrix manually"""
        self.camera_intrinsic = intrinsic

    def get_process(self, det):
        """PnP solver for getting 2d projection of keypoints & 6-DoF & 3d keypoint"""
        # cv2 pnp solver can not batch processing.
        ret = []
        for idx in range(len(det)):
            results = {'keypoints_2d': [], 'projected_points': [], 'point_3d_cam': [], 'bbox': [], 'obj_scale': [], 'location': [], 'quaternion': [], 'score': []}
            outputs = det[idx]
            camera_intrinsic = self.camera_intrinsic[idx] if len(self.camera_intrinsic.shape) == 3 else self.camera_intrinsic

            for bbox in outputs:
                # 16 representation
                points_1 = np.array(bbox['kps_displacement_mean']).reshape(-1, 2)
                points_1 = [(x[0], x[1]) for x in points_1]
                points_2 = np.array(bbox['kps_heatmap_mean']).reshape(-1, 2)
                points_2 = [(x[0], x[1]) for x in points_2]
                points = np.hstack((points_1, points_2)).reshape(-1, 2)
                points_filtered = np.array(points)

                pnp_out = pnp_shell(points_filtered, np.array(bbox['obj_scale']), camera_intrinsic, opencv_return=self.opencv)

                if pnp_out is not None:
                    projected_points, point_3d_cam, location, quaternion = pnp_out
                    results['projected_points'].append(projected_points)
                    results['point_3d_cam'].append(point_3d_cam)
                    results['location'].append(location)
                    results['quaternion'].append(quaternion)
                    results['bbox'].append(bbox['bbox'])
                    results['obj_scale'].append(bbox['obj_scale'])
                    results['keypoints_2d'].append(np.array(points_1).reshape(-1, 2))
                    results['score'].append(bbox['score'])

            ret.append(results)
        return ret
