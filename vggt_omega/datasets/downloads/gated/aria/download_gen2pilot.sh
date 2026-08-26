#!/usr/bin/env bash
# Aria Gen 2 Pilot Dataset -- via explorer.projectaria.com (no email gate).
# Docs: https://facebookresearch.github.io/projectaria_tools/gen2/research-tools/dataset/pilot/content  License: CC BY-NC 4.0
# 12 sequences, 1.1 h. Full set 159 GB; DEFAULT_GROUPS ~105 GB.
# Groups (GB): mps_slam_points 44 | main_vrs 35 | depth 25 | mps_slam_trajectories 0.4 | scene/hand_*/heart_rate/diarization <0.1 |
#   skipped: video_main_rgb 10 (preview), mps_artifacts 45 (= mps_* bundled)
DATASET_NAME="gen2pilot"
SLUG="gen2pilot"
MANIFEST_NAME="Aria_Gen_2_Pilot_Dataset_download_urls.json"
DEFAULT_GROUPS="main_vrs,depth,scene,mps_slam_trajectories,mps_slam_calibration,mps_slam_points,mps_slam_summary,mps_hand_tracking,hand_object_interaction,heart_rate,diarization"
source "$(dirname "$0")/aria_common.sh"
aria_dataset_main "$@"
