# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Original source taken from https://github.com/NVIDIA/NeMo
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

"""Common utility functions."""


def remove_none_empty_fields(json_schema):
    """
    Recursively remove all None and empty string values and their corresponding keys from a dictionary.

    Parameters:
    json_schema (dict): The input dictionary from which None and empty string values should be removed.

    Returns:
    dict: A new dictionary with all None and empty string values removed.
    """
    if not isinstance(json_schema, dict):
        return json_schema

    new_dict = {}
    for key, value in json_schema.items():
        if isinstance(value, dict):
            nested_dict = remove_none_empty_fields(value)
            if nested_dict:  # only add if nested_dict is not empty
                new_dict[key] = nested_dict
        elif isinstance(value, list):
            new_list = [
                remove_none_empty_fields(item)
                for item in value
                if item is not None and item != ""
            ]
            if new_list:  # only add if new_list is not empty
                new_dict[key] = new_list
        elif value is not None and value != "":
            new_dict[key] = value

    return new_dict
