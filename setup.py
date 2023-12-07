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

"""Setup script to build the TAO Toolkit package."""

import os
import setuptools
import sys

from release.python.utils import utils

version_locals = utils.get_version_details()
PACKAGE_LIST = [
    "nvidia_tao_deploy"
]

__python_version__ = "=={}.{}.*".format(sys.version_info.major, sys.version_info.minor)


# Getting dependencies.
def get_requirements():
    """Simple function to get packages."""
    package_root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(package_root, "docker/requirements.txt"), 'r') as req_file:
        requirements = [r.replace('\n', '')for r in req_file.readlines()]
    return requirements


setuptools_packages = []
for package_name in PACKAGE_LIST:
    setuptools_packages.extend(utils.find_packages(package_name))

setuptools.setup(
    name=version_locals['__package_name__'],
    version=version_locals['__version__'],
    description=version_locals['__description__'],
    author='NVIDIA Corporation',
    classifiers=[
        'Environment :: Console',
        'License :: Other/Proprietary License',
        'Natural Language :: English',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    license=version_locals['__license__'],
    keywords=version_locals['__keywords__'],
    packages=setuptools_packages,
    package_data={
        '': ['*.pyc', "*.yaml", "*.so", '*.pdf']
    },
    include_package_data=True,
    python_requires=__python_version__,
    install_requires=get_requirements(),
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'centerpose=nvidia_tao_deploy.cv.centerpose.entrypoint.centerpose:main',
            'classification_pyt=nvidia_tao_deploy.cv.classification_pyt.entrypoint.classification_pyt:main',
            'classification_tf1=nvidia_tao_deploy.cv.classification_tf1.entrypoint.classification_tf1:main',
            'classification_tf2=nvidia_tao_deploy.cv.classification_tf2.entrypoint.classification_tf2:main',
            'deformable_detr=nvidia_tao_deploy.cv.deformable_detr.entrypoint.deformable_detr:main',
            'detectnet_v2=nvidia_tao_deploy.cv.detectnet_v2.entrypoint.detectnet_v2:main',
            'dino=nvidia_tao_deploy.cv.dino.entrypoint.dino:main',
            'dssd=nvidia_tao_deploy.cv.ssd.entrypoint.ssd:main',
            'efficientdet_tf1=nvidia_tao_deploy.cv.efficientdet_tf1.entrypoint.efficientdet_tf1:main',
            'efficientdet_tf2=nvidia_tao_deploy.cv.efficientdet_tf2.entrypoint.efficientdet_tf2:main',
            'faster_rcnn=nvidia_tao_deploy.cv.faster_rcnn.entrypoint.faster_rcnn:main',
            'lprnet=nvidia_tao_deploy.cv.lprnet.entrypoint.lprnet:main',
            'mask_rcnn=nvidia_tao_deploy.cv.mask_rcnn.entrypoint.mask_rcnn:main',
            'ml_recog=nvidia_tao_deploy.cv.metric_learning_recognition.entrypoint.metric_learning_recognition:main',
            'multitask_classification=nvidia_tao_deploy.cv.multitask_classification.entrypoint.multitask_classification:main',
            'ocdnet=nvidia_tao_deploy.cv.ocdnet.entrypoint.ocdnet:main',
            'ocrnet=nvidia_tao_deploy.cv.ocrnet.entrypoint.ocrnet:main',
            'optical_inspection=nvidia_tao_deploy.cv.optical_inspection.entrypoint.optical_inspection:main',
            'retinanet=nvidia_tao_deploy.cv.retinanet.entrypoint.retinanet:main',
            'ssd=nvidia_tao_deploy.cv.ssd.entrypoint.ssd:main',
            'segformer=nvidia_tao_deploy.cv.segformer.entrypoint.segformer:main',
            'unet=nvidia_tao_deploy.cv.unet.entrypoint.unet:main',
            'visual_changenet=nvidia_tao_deploy.cv.visual_changenet.entrypoint.visual_changenet:main',
            'yolo_v3=nvidia_tao_deploy.cv.yolo_v3.entrypoint.yolo_v3:main',
            'yolo_v4=nvidia_tao_deploy.cv.yolo_v4.entrypoint.yolo_v4:main',
            'yolo_v4_tiny=nvidia_tao_deploy.cv.yolo_v4.entrypoint.yolo_v4:main',
        ]
    }
)
