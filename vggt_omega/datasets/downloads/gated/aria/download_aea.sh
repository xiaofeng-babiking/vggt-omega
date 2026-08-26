#!/usr/bin/env bash
# Aria Everyday Activities (AEA, re-release of the Aria Pilot Dataset) -- via explorer.projectaria.com (no email gate).
# Docs: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_activities_dataset  License: https://www.projectaria.com/datasets/aea/license
# 143 sequences, 7.6 h. Full set 397 GB; DEFAULT_GROUPS ~343 GB.
# Groups (GB): main_vrs 309 | mps_slam_points 31 | mps_slam_trajectories 3 | mps_slam_calibration 0.2 | annotations ~0 |
#   skipped: video_main_rgb 20 (preview), mps_artifacts 34 (= mps_* bundled), mps_eye_gaze 0.01
DATASET_NAME="aea"
SLUG="aea"
MANIFEST_NAME="Aria_Everyday_Activities_download_urls.json"
DEFAULT_GROUPS="main_vrs,annotations,mps_slam_trajectories,mps_slam_calibration,mps_slam_points,mps_slam_summary"
source "$(dirname "$0")/aria_common.sh"
aria_dataset_main "$@"
