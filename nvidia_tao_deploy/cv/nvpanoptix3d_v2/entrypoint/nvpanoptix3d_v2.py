# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entrypoint for NVPanoptix3Dv2 deployment tasks."""

import argparse

from nvidia_tao_deploy.cv.common.entrypoint.entrypoint_hydra import (
    command_line_parser,
    get_subtasks,
    launch,
)
from nvidia_tao_deploy.cv.nvpanoptix3d_v2 import scripts


def get_subtask_list():
    """Return deployment subtasks discovered in the scripts package."""
    return get_subtasks(scripts)


def main():
    """Parse the command line and launch an NVPanoptix3Dv2 subtask."""
    parser = argparse.ArgumentParser(
        "nvpanoptix3d_v2",
        add_help=True,
        description="Deploy the NVPanoptix3Dv2 panoptic model",
    )
    subtasks = get_subtask_list()
    args, unknown_args = command_line_parser(parser, subtasks)
    launch(vars(args), unknown_args, subtasks, network="nvpanoptix3d_v2")


if __name__ == "__main__":
    main()
