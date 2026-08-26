#!/usr/bin/env bash
# DTC-Aria (Digital Twin Catalog object-capture sequences) -- via explorer.projectaria.com (no email gate).
# Docs: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/digital_twin_catalog  License: https://www.projectaria.com/datasets/dtc/license
# 200 sequences (100 objects x active/passive), 1.6 h. Full set 307 GB; DEFAULT_GROUPS ~278 GB.
# Groups (GB): main_vrs 273 | mps_slam_points 4.4 | mps_slam_trajectories 0.65 | main_groundtruth ~0 |
#   skipped: video_main_rgb 24 (preview), mps_artifacts 5 (= mps_* bundled). Object meshes: download_dtc_objects.sh
DATASET_NAME="dtc"
SLUG="dtc"
MANIFEST_NAME="DTC-Aria_download_urls.json"
DEFAULT_GROUPS="main_vrs,main_groundtruth,mps_slam_trajectories,mps_slam_calibration,mps_slam_points,mps_slam_summary"
source "$(dirname "$0")/aria_common.sh"
aria_dataset_main "$@"
