#!/usr/bin/env bash
# HOT3D-Aria (hand-object tracking, Aria glasses) -- via explorer.projectaria.com (no email gate).
# Docs: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/hot3d  License: https://www.projectaria.com/datasets/hot3d/license/
# 198 sequences, 6.5 h. Full set 526 GB; DEFAULT_GROUPS ~420 GB.
# Groups (GB): main_vrs 329 | mps_slam_points 84 | mps_slam_trajectories 5 | hand_data 1 | ground_truth 0.7 |
#   skipped: video_main_rgb 18 (preview), mps_artifacts 89 (= mps_* bundled), mps_eye_gaze 0.05
# Object models (33 HOT3D objects, Hot3DAssets_download_urls.json) are NOT served by the explorer: see download_hot3d.sh.
DATASET_NAME="hot3d_aria"
SLUG="hot3d-aria"
MANIFEST_NAME="HOT3D-Aria_download_urls.json"
DEFAULT_GROUPS="main_vrs,ground_truth,hand_data,mps_slam_trajectories,mps_slam_calibration,mps_slam_points,mps_slam_summary"
source "$(dirname "$0")/aria_common.sh"
aria_dataset_main "$@"
