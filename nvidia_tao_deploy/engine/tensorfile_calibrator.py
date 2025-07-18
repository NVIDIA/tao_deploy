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

"""Base calibrator class for TensorRT INT8 Calibration."""

import logging
import os

import numpy as np

import pycuda.autoinit  # noqa pylint: disable=unused-import
import pycuda.driver as cuda

import tensorrt as trt

# Simple helper class for calibration.
from nvidia_tao_deploy.engine.tensorfile import TensorFile

logger = logging.getLogger(__name__)


class TensorfileCalibrator(trt.IInt8EntropyCalibrator2):
    """Calibrator class."""

    def __init__(self, data_filename, cache_filename,
                 n_batches, batch_size,
                 *args, **kwargs):
        """Init routine.

        This inherits from ``trt.IInt8EntropyCalibrator2``
        to implement the calibration interface that TensorRT needs to
        calibrate the INT8 quantization factors. The data source here is assumed
        to be a Tensorfile as defined in engine.tensorfile.Tensorfile(), which
        was pre-generated using the dataloader

        Args:
            data_filename (str): ``TensorFile`` data file to use.
            cache_filename (str): name of calibration file to read/write to.
            n_batches (int): number of batches for calibrate for.
            batch_size (int): batch size to use for calibration data.
        """
        super().__init__(*args, **kwargs)
        self._data_source = None
        self._cache_filename = cache_filename

        self._batch_size = batch_size
        self._n_batches = n_batches

        self._batch_count = 0
        self._data_mem = None

        self.instantiate_data_source(data_filename)

    def instantiate_data_source(self, data_filename):
        """Simple function to instantiate the data_source of the dataloader.

        Args:
            data_filename (str): The path to the data file.

        Returns:
            No explicit returns.
        """
        if os.path.exists(data_filename):
            self._data_source = TensorFile(data_filename, "r")
        else:
            logger.info(
                "A valid data source wasn't provided to the calibrator. "
                "The calibrator will attempt to read from a cache file if provided."
            )

    def get_data_from_source(self):
        """Simple function to get data from the defined data_source."""
        batch = np.array(self._data_source.read())
        if batch is not None:
            # <@vpraveen>: Disabling pylint error check on line below
            # because of a python3 linting error. To be reverted when
            # pylint/issues/3139 gets fixed.
            batch_size = batch.shape[0]  # pylint: disable=E1136
            if batch_size < self._batch_size:
                raise ValueError(
                    f"Batch size yielded from data source {batch_size} < requested batch size "
                    "from calibrator {self._batch_size}"
                )
            batch = batch[:self._batch_size]
        else:
            raise ValueError(
                "Batch wasn't yielded from the data source. You may have run "
                "out of batches. Please set the num batches accordingly")
        return batch

    def get_batch_size(self):
        """Return batch size."""
        return self._batch_size

    def get_batch(self, names):
        """Return one batch.

        Args:
            names (list): list of memory bindings names.
        """
        if self._batch_count < self._n_batches:
            batch = self.get_data_from_source()
            if batch is not None:
                if self._data_mem is None:
                    # 4 bytes per float32.
                    self._data_mem = cuda.mem_alloc(batch.size * 4)

                self._batch_count += 1

                # Transfer input data to device.
                cuda.memcpy_htod(self._data_mem, np.ascontiguousarray(
                    batch, dtype=np.float32))
                return [int(self._data_mem)]

        if self._data_mem is not None:
            self._data_mem.free()
        return None

    def read_calibration_cache(self):
        """Read calibration from file."""
        logger.debug("read_calibration_cache - no-op")
        if os.path.isfile(self._cache_filename):
            logger.warning("Calibration file exists at %s."
                           " Reading this cache.", self._cache_filename)
            with open(self._cache_filename, "rb") as cal_file:
                return cal_file.read()
        return None

    def write_calibration_cache(self, cache):
        """Write calibration to file.

        Args:
            cache (memoryview): buffer to read calibration data from.
        """
        logger.info("Saving calibration cache (size %d) to %s",
                    len(cache), self._cache_filename)
        with open(self._cache_filename, 'wb') as f:
            f.write(cache)
