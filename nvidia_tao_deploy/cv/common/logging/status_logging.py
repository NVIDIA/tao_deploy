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

"""Logger class for TAO Deploy models."""

from abc import abstractmethod
import atexit
from datetime import datetime
import json
import logging
import os

logger = logging.getLogger(__name__)


class Verbosity():
    """Verbosity levels."""

    DISABLE = 0
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


# Defining a log level to name dictionary.
log_level_to_name = {
    Verbosity.DISABLE: "DISABLE",
    Verbosity.DEBUG: 'DEBUG',
    Verbosity.INFO: 'INFO',
    Verbosity.WARNING: 'WARNING',
    Verbosity.ERROR: 'ERROR',
    Verbosity.CRITICAL: 'CRITICAL'
}


class Status():
    """Status levels."""

    SUCCESS = 0
    FAILURE = 1
    STARTED = 2
    RUNNING = 3
    SKIPPED = 4


status_level_to_name = {
    Status.SUCCESS: 'SUCCESS',
    Status.FAILURE: 'FAILURE',
    Status.STARTED: 'STARTED',
    Status.RUNNING: 'RUNNING',
    Status.SKIPPED: 'SKIPPED'
}


class BaseLogger(object):
    """File logger class."""

    def __init__(self, is_master=False, verbosity=Verbosity.DISABLE):
        """Base logger class."""
        self.is_master = is_master
        self.verbosity = verbosity
        self.categorical = {}
        self.graphical = {}
        self.kpi = {}

    @property
    def date(self):
        """Get date from the status."""
        date_time = datetime.now()
        date_object = date_time.date()
        return "{}/{}/{}".format(  # noqa pylint: disable=C0209
            date_object.month,
            date_object.day,
            date_object.year
        )

    @property
    def time(self):
        """Get date from the status."""
        date_time = datetime.now()
        time_object = date_time.time()
        return "{}:{}:{}".format(  # noqa pylint: disable=C0209
            time_object.hour,
            time_object.minute,
            time_object.second
        )

    @property
    def categorical(self):
        """Categorical data to be logged."""
        return self._categorical

    @categorical.setter
    def categorical(self, value: dict):
        """Set categorical data to be logged."""
        self._categorical = value

    @property
    def graphical(self):
        """Graphical data to be logged."""
        return self._graphical

    @graphical.setter
    def graphical(self, value: dict):
        """Set graphical data to be logged."""
        self._graphical = value

    @property
    def kpi(self):
        """Set KPI data."""
        return self._kpi

    @kpi.setter
    def kpi(self, value: dict):
        """Set KPI data."""
        self._kpi = value

    def flush(self):
        """Flush the logger."""
        pass

    def format_data(self, data: dict):
        """Format the data."""
        if isinstance(data, dict):
            data_string = []
            for key, value in data.items():
                data_string.append(
                    f"{key}: {self.format_data(value)}"
                    if isinstance(value, dict) else value
                )
        return ", ".join(data_string)

    def log(self, level, string):
        """Log the data string."""
        if level >= self.verbosity:
            logging.log(level, string)

    @abstractmethod
    def write(self, data=None,
              status_level=Status.RUNNING,
              verbosity_level=Verbosity.INFO,
              message=None):
        """Write data out to the log file."""
        if self.verbosity > Verbosity.DISABLE:
            if not data:
                data = {}
            # Define generic data.
            data["date"] = self.date
            data["time"] = self.time
            data["status"] = status_level_to_name.get(status_level, "RUNNING")
            data["verbosity"] = log_level_to_name.get(verbosity_level, "INFO")

            if message:
                data["message"] = message
            logging.log(verbosity_level, message)

            if self.categorical:
                data["categorical"] = self.categorical

            if self.graphical:
                data["graphical"] = self.graphical

            if self.kpi:
                data["kpi"] = self.kpi

            data_string = self.format_data(data)
            if self.is_master:
                self.log(verbosity_level, data_string)
            self.flush()


class StatusLogger(BaseLogger):
    """Simple logger to save the status file."""

    def __init__(self, filename=None,
                 is_master=False,
                 verbosity=Verbosity.INFO,
                 append=True):
        """Logger to write out the status."""
        super().__init__(is_master=is_master, verbosity=verbosity)
        self.log_path = os.path.realpath(filename)
        if os.path.exists(self.log_path):
            logger.info("Log file already exists at %s", self.log_path)
        if is_master:
            self.l_file = open(self.log_path, "a" if append else "w", encoding='utf-8')  # noqa pylint: disable=R1732
            atexit.register(self.l_file.close)

    def log(self, level, string):
        """Log the data string."""
        if level >= self.verbosity:
            self.l_file.write(string + "\n")

    def flush(self):
        """Flush contents of the log file."""
        if self.is_master:
            self.l_file.flush()

    @staticmethod
    def format_data(data):
        """Format the dictionary data."""
        if not isinstance(data, dict):
            raise TypeError(f"Data must be a dictionary and not type {type(data)}.")
        data_string = json.dumps(data)
        return data_string


# Define the logger here so it's static.
_STATUS_LOGGER = BaseLogger()


def set_status_logger(status_logger):
    """Set the status logger.

    Args:
        status_logger: An instance of the logger class.
    """
    global _STATUS_LOGGER  # pylint: disable=W0603
    _STATUS_LOGGER = status_logger


def get_status_logger():
    """Get the status logger."""
    global _STATUS_LOGGER  # pylint: disable=W0602,W0603
    return _STATUS_LOGGER
