#!/usr/bin/env bash
# Aria Scenes (photoreal reconstruction benchmark, 12 indoor/outdoor scenes) -- via explorer.projectaria.com.
# Docs: https://www.projectaria.com/photoreal-reconstruction/  License: CC BY-NC 4.0
# 12 sequences, 0.5 h. Full set 133 GB; DEFAULT_GROUPS ~112 GB (main_vrs 108 | mps_slam_points 3 | trajectories 0.3;
#   skipped: video_main_rgb 18 (preview), mps_artifacts 3.4 (= mps_* bundled)).
DATASET_NAME="aria_scenes"
SLUG="aria-scenes"
MANIFEST_NAME="Aria_Scenes_download_urls.json"
DEFAULT_GROUPS="main_vrs,mps_slam_trajectories,mps_slam_calibration,mps_slam_points,mps_slam_summary"
source "$(dirname "$0")/aria_common.sh"
aria_dataset_main "$@"
