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

"""This script is responsible for generating default experiment.yaml files from dataclasses."""

from __future__ import annotations

import os
import logging
import importlib
from os import makedirs, listdir
from os.path import abspath, dirname, exists, join

from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass

from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner

# Usage example:
# ==============
"""
python default_specs \
    results_dir=/results/classification_tf2/ \
    module_name=classification_tf2
"""


# Get the config root from tao-core
try:
    import nvidia_tao_core
    TAO_CORE_ROOT = dirname(dirname(abspath(nvidia_tao_core.__file__)))
    CONFIG_ROOT = join(TAO_CORE_ROOT, "nvidia_tao_core/config")
except ImportError:
    # Fallback: try to find tao-core relative to tao-deploy
    # __file__ is in nvidia_tao_deploy/cv/common/utils/default_specs.py
    # Need to go up 5 levels to get to tao-deploy root
    TAO_DEPLOY_ROOT = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))
    TAO_CORE_ROOT = join(dirname(TAO_DEPLOY_ROOT), "tao-core")
    CONFIG_ROOT = join(TAO_CORE_ROOT, "nvidia_tao_core/config")


def get_supported_modules():
    """
    Get list of supported modules from config directory that are also implemented in nvidia_tao_deploy.

    This function checks both:
    1. Modules defined in nvidia_tao_core/config/
    2. Modules actually implemented in nvidia_tao_deploy/cv

    Returns:
        List[str]: List of module names that have both config definitions and deploy implementations
    """
    if not exists(CONFIG_ROOT):
        logging.warning("Config root not found at %s", CONFIG_ROOT)
        return []

    # Get all config modules from tao-core
    config_modules = [
        item for item in listdir(CONFIG_ROOT)
        if item not in ["utils", "__pycache__", "common"] and os.path.isdir(join(CONFIG_ROOT, item))
    ]

    # Get TAO Deploy path from current file location without importing
    # __file__ is in nvidia_tao_deploy/cv/common/utils/default_specs.py
    # Go up 4 levels to get to nvidia_tao_deploy/, then get cv directory
    nvidia_tao_deploy_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
    cv_dir = join(nvidia_tao_deploy_dir, "cv")

    # Get all implemented modules from tao-deploy (directories with entrypoints in cv/)
    deploy_modules = set()
    if exists(cv_dir):
        for item in listdir(cv_dir):
            item_path = join(cv_dir, item)
            if os.path.isdir(item_path) and item not in ["__pycache__", "common"]:
                # Check if it has an entrypoint (indicates it's a network module)
                entrypoint_path = join(item_path, "entrypoint")
                if exists(entrypoint_path):
                    deploy_modules.add(item)

    # Return only modules that exist in both places
    supported = [module for module in config_modules if module in deploy_modules]

    if not supported:
        logging.warning(
            "No matching modules found between config (%d modules) "
            "and deploy implementation (%d modules)",
            len(config_modules), len(deploy_modules)
        )

    return sorted(supported)


def import_module_from_path(module_name):
    """
    Import a module from its full path.

    Args:
        module_name (str): Full module path (e.g., 'nvidia_tao_core.config.classification_tf2.default_config')

    Returns:
        module: The imported module
    """
    return importlib.import_module(module_name)


def dataclass_to_yaml(dataclass_obj, yaml_file_path):
    """
    Convert a dataclass object to a YAML file using omegaconf.

    Args:
        dataclass_obj (object): The dataclass object to convert.
        yaml_file_path (str): The path to the output YAML file.

    Returns:
        None
    """
    if not hasattr(dataclass_obj, "__dataclass_fields__"):
        raise ValueError("Provided object is not a dataclass instance.")

    # Convert dataclass to OmegaConf structured object
    conf = OmegaConf.structured(dataclass_obj)

    # Save as YAML
    output_dir = dirname(yaml_file_path)
    if output_dir and not exists(output_dir):
        makedirs(output_dir, exist_ok=True)
    with open(yaml_file_path, 'w', encoding='utf-8') as yaml_file:
        yaml_file.write(OmegaConf.to_yaml(conf))
        logging.info("Generated default spec: %s", yaml_file_path)


@dataclass
class DefaultConfig:
    """This is a structured config for generating default specs."""

    # Minimalistic experiment manager.
    results_dir: str = MISSING
    module_name: str = MISSING


spec_path = dirname(abspath(__file__))


@hydra_runner(config_path=spec_path, config_name="default_specs", schema=DefaultConfig)
def main(cfg: DefaultConfig) -> None:
    """Script to generate default experiment YAML from dataclasses.

    Args:
        cfg (OmegaConf.DictConf): Hydra parsed config object.
    """
    logging.info("Generating default spec for module: %s", cfg.module_name)

    # Validate module name
    supported_modules = get_supported_modules()
    if cfg.module_name not in supported_modules:
        error_msg = (f"Module '{cfg.module_name}' is not supported.\n"
                     f"Supported modules: {', '.join(supported_modules)}")
        logging.error(error_msg)
        raise ValueError(error_msg)

    # Create results directory if it doesn't exist
    if not exists(cfg.results_dir):
        makedirs(cfg.results_dir, exist_ok=True)
        logging.info("Created results directory: %s", cfg.results_dir)

    # Set output file path
    output_filename = "experiment.yaml"
    output_path = join(cfg.results_dir, output_filename)
    if exists(output_path):
        logging.warning("Output file already exists and will be overwritten: %s", output_path)

    # Import the module and get the ExperimentConfig dataclass
    module_path = f"nvidia_tao_core.config.{cfg.module_name}.default_config"
    try:
        imported_module = import_module_from_path(module_path)
        if not hasattr(imported_module, 'ExperimentConfig'):
            raise AttributeError(f"Module '{module_path}' does not have 'ExperimentConfig' dataclass")

        # Generate YAML from dataclass
        dataclass_to_yaml(imported_module.ExperimentConfig, output_path)

        # Success logging
        logging.info("Default specification file for %s generated at '%s'", cfg.module_name, output_path)

    except ImportError as e:
        error_msg = f"Failed to import module '{module_path}': {str(e)}"
        logging.error(error_msg)
        raise
    except Exception as e:
        error_msg = f"Failed to generate spec for {cfg.module_name}: {str(e)}"
        logging.error(error_msg)
        raise


if __name__ == "__main__":
    main()
