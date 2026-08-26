#!/usr/bin/env bash
# Aria Digital Twin (ADT) -- Meta Project Aria, via explorer.projectaria.com (no email gate).
# Docs: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset
# License: https://www.projectaria.com/datasets/adt/license
# 236 sequences, 8.1 h. Full set 2.28 TB; DEFAULT_GROUPS ~2.15 TB.
# Groups (GB): depth 1301 | main_vrs 433 | synthetic 208 | mps_slam_points 102 | segmentation 94 |
#   main_groundtruth 11 | mps_slam_trajectories 3 | mps_slam_calibration 0.05 | mps_slam_summary ~0 |
#   skipped: video_main_rgb 24 (browser preview mp4), mps_artifacts 105 (= the mps_* zips bundled), mps_eye_gaze 0.03
# Examples: DATA_GROUPS=all ./download_adt.sh | SEQUENCES=Apartment_release_clean_seq131_M1292 ./download_adt.sh
DATASET_NAME="adt"
SLUG="adt"
MANIFEST_NAME="Aria_Digital_Twin_download_urls.json"
DEFAULT_GROUPS="main_vrs,main_groundtruth,depth,segmentation,synthetic,mps_slam_trajectories,mps_slam_calibration,mps_slam_points,mps_slam_summary"
source "$(dirname "$0")/aria_common.sh"
aria_dataset_main "$@"
