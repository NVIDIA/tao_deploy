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

"""Dataset module."""
import copy
import cv2
import json
import multiprocessing
import numpy as np
import os
import pathlib
from nvidia_tao_deploy.cv.ocdnet.data_loader import icdar_uber
from nvidia_tao_deploy.utils.path_utils import expand_path


def get_dataset(data_path, module_name, transform, dataset_args):
    """Get dataset.

    Args:
        data_path: dataset file list.
        module_name: custom dataset name，Supports data_loaders.ImageDataset
        dataset_args: module_name args
    Returns:
        ConcatDataset object
    """
    s_dataset = getattr(icdar_uber, module_name)(transform=transform, data_path=data_path,
                                                 **dataset_args)
    return s_dataset


class ICDARCollateFN:
    """ICDAR Collation."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        pass

    def __call__(self, batch):
        """Call fn."""
        pass


def get_dataloader(module_config, distributed=False):
    """Get dataloader."""
    if module_config is None:
        return None
    config = copy.deepcopy(module_config)
    dataset_args = config['args']
    dataset_name = config['data_name']
    data_path = config['data_path']
    if data_path is None:
        return None

    data_path = [x for x in data_path if x is not None]
    if len(data_path) == 0:
        return None
    if 'collate_fn' not in config['loader'] or config['loader']['collate_fn'] is None or len(config['loader']['collate_fn']) == 0:
        config['loader']['collate_fn'] = None
    else:
        config['loader']['collate_fn'] = globals()[config['loader']['collate_fn']]()
    _dataset = get_dataset(data_path=data_path, module_name=dataset_name, transform=None, dataset_args=dataset_args)

    return _dataset


def _load_txt(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = [x.strip().strip('\ufeff').strip('\xef\xbb\xbf') for x in f.readlines()]
    return content


def _load_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


def load(file_path: str):
    """load file."""
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': _load_txt, '.json': _load_json, '.list': _load_txt}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](file_path)


def order_points_clockwise(pts):
    """order points clockwise."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_datalist(train_data_path):
    """Get train data list and val data list"""
    train_data = []
    for p in train_data_path:
        # use list file
        if os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                    if len(line) > 1:
                        img_path = pathlib.Path(expand_path(line[0].strip(' ')))
                        label_path = pathlib.Path(expand_path(line[1].strip(' ')))
                        if img_path.exists() and img_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
                            train_data.append((str(img_path), str(label_path)))
        # use standard directory structure
        else:
            img_dir = os.path.join(p, "img")
            label_dir = os.path.join(p, "gt")
            for img in os.listdir(img_dir):
                img_file = os.path.join(img_dir, img)
                label = "gt_" + img.split('.')[0] + ".txt"
                label_file = os.path.join(label_dir, label)
                assert os.path.exists(label_file), (
                    f"Cannot find label file for image: {img_file}"
                )
                train_data.append((img_file, label_file))
    return sorted(train_data)


def get_datalist_uber(train_data_path):
    """Get uber train data list and val data list"""
    train_data = []
    for p in train_data_path:
        # use list file
        if os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                    if len(line) > 1:
                        img_path = pathlib.Path(expand_path(line[0].strip(' ')))
                        label_path = pathlib.Path(expand_path(line[1].strip(' ')))
                        if img_path.exists() and img_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
                            train_data.append((str(img_path), str(label_path)))
        # use standard directory structure
        else:
            img_dir = os.path.join(p, "img")
            label_dir = os.path.join(p, "gt")
            for img in os.listdir(img_dir):
                img_file = os.path.join(img_dir, img)
                label = "truth_" + img.split('.')[0] + ".txt"
                label_file = os.path.join(label_dir, label)
                assert os.path.exists(label_file), (
                    f"Cannot find label file for image: {img_file}"
                )
                train_data.append((img_file, label_file))
    return sorted(train_data)


def expand_polygon(polygon):
    """expand bbox which has only one character."""
    (x, y), (w, h), angle = cv2.minAreaRect(np.float32(polygon))
    if angle < -45:
        w, h = h, w
        angle += 90
    new_w = w + h
    box = ((x, y), (new_w, h), angle)
    points = cv2.boxPoints(box)
    return order_points_clockwise(points)


class Resize2D:
    """Resize 2D."""

    def __init__(self, short_size, resize_text_polys=True):
        """Initialize."""
        self.short_size = short_size
        self.resize_text_polys = resize_text_polys

    def __call__(self, data: dict) -> dict:
        """Resize images and texts"""
        im = data['img']
        text_polys = data['text_polys']

        h, w, _ = im.shape
        if isinstance(self.short_size, (list, tuple)):
            target_width = self.short_size[0]
            target_height = self.short_size[1]
            scale = (target_width / w, target_height / h)
            im = cv2.resize(im, dsize=None, fx=scale[0], fy=scale[1])
            if self.resize_text_polys:
                text_polys[:, :, 0] *= scale[0]
                text_polys[:, :, 1] *= scale[1]
        else:
            short_edge = min(h, w)
            if short_edge < self.short_size:
                # make sure shorter edge >= short_size
                scale = self.short_size / short_edge
                im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
                scale = (scale, scale)
                if self.resize_text_polys:
                    text_polys[:, :, 0] *= scale[0]
                    text_polys[:, :, 1] *= scale[1]

        data['img'] = im
        data['text_polys'] = text_polys
        return data


class BaseDataSet():
    """BaseDataSet class."""

    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None,
                 target_transform=None):
        """Initialize."""
        assert img_mode in ['RGB', 'BGR', 'GRAY']
        self.ignore_tags = ignore_tags
        self.data_list = self.load_data(data_path)
        item_keys = ['img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags']
        for item in item_keys:
            assert item in self.data_list[0], f'data_list from load_data must contains {item_keys}'
        self.img_mode = img_mode
        self.filter_keys = filter_keys
        self.transform = transform
        self.target_transform = target_transform
        self._init_pre_processes(pre_processes)

    def _init_pre_processes(self, pre_processes):
        self.aug = []
        if pre_processes is not None:
            for aug in pre_processes:
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']

                if isinstance(args, dict):
                    cls = globals()[aug['type']](**args)
                else:
                    cls = globals()[aug['type']](args)

                self.aug.append(cls)

    def load_data(self, data_path: str) -> list:
        """Load data to a list

        Args:
            data_path (str): file or folder

        Returns:
            A dict (dict): contains 'img_path','img_name','text_polys','texts','ignore_tags'
        """
        raise NotImplementedError

    def apply_pre_processes(self, data):
        """Apply pre_processing."""
        for aug in self.aug:
            data = aug(data)
        return data

    def __getitem__(self, index):
        """getitem function."""
        try:
            data = copy.deepcopy(self.data_list[index])
            im = cv2.imread(data['img_path'], 1 if self.img_mode != 'GRAY' else 0).astype("float32")
            if self.img_mode == 'RGB':
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            data['img'] = im
            data['shape'] = [im.shape[0], im.shape[1]]
            data = self.apply_pre_processes(data)
            rgb_mean = np.array([122.67891434, 116.66876762, 104.00698793])
            image = data['img']
            image -= rgb_mean
            image /= 255.
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            data['img'] = image
            data['text_polys'] = data['text_polys'].tolist()
            if len(self.filter_keys):
                data_dict = {}
                for k, v in data.items():
                    if k not in self.filter_keys:
                        data_dict[k] = v
                return data_dict
            return data
        except Exception:
            return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        """len functin."""
        return len(self.data_list)


class UberDataset(BaseDataSet):
    """Uber Dataset class."""

    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        """Initialize."""
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        """Load data."""
        pool = multiprocessing.Pool(processes=4)  # pylint: disable=R1732
        data_list = pool.apply_async(get_datalist_uber, args=(data_path,)).get()
        pool.close()
        pool.join()

        t_data_list = []
        pool = multiprocessing.Pool(processes=4)  # pylint: disable=R1732
        for img_path, label_path in data_list:
            tmp = pool.apply_async(self._get_annotation, args=(label_path,))
            data = tmp.get()
            if len(data['text_polys']) > 0:
                item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                item.update(data)
                t_data_list.append(item)
            else:
                print(f'there is no suit bbox in {label_path}')
        pool.close()
        pool.join()

        return t_data_list

    def _get_annotation(self, label_path: str) -> dict:
        polys = []
        texts = []
        ignores = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f:
                content = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split('\t')
                params = content[0].split(" ")[:-2]
                try:
                    poly = np.array(list(map(float, params))).reshape(-1, 2).astype(np.float32)
                    if cv2.contourArea(poly) > 0:
                        polys.append(poly)
                        label = content[1]
                        if len(label.split(" ")) > 1:
                            label = "###"
                        texts.append(label)
                        ignores.append(label in self.ignore_tags)
                except Exception:
                    print(f'load label failed on {label_path}')
        data = {
            'text_polys': np.array(polys),
            'texts': texts,
            'ignore_tags': ignores,
        }

        return data


class ICDAR2015Dataset(BaseDataSet):
    """ICDAR2015 Dataset."""

    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        """Initialize."""
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        """Load data."""
        data_list = get_datalist(data_path)
        t_data_list = []
        for img_path, label_path in data_list:
            data = self._get_annotation(label_path)
            if len(data['text_polys']) > 0:
                item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                item.update(data)
                t_data_list.append(item)
            else:
                print(f'there is no suit bbox in {label_path}')

        return t_data_list

    def _get_annotation(self, label_path: str) -> dict:
        boxes = []
        texts = []
        ignores = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    box = order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2))
                    if cv2.contourArea(box) > 0:
                        boxes.append(box)
                        label = params[8]
                        texts.append(label)
                        ignores.append(label in self.ignore_tags)
                except Exception:
                    print(f'load label failed on {label_path}')
        data = {
            'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores,
        }

        return data
