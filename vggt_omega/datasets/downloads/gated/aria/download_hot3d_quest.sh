#!/usr/bin/env bash
# HOT3D-Quest (hand-object tracking, Quest 3 headset) -- via explorer.projectaria.com (no email gate).
# Docs: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/hot3d  License: https://www.projectaria.com/datasets/hot3d/license/
# 226 sequences. Full set 394 GB; DEFAULT_GROUPS ~387 GB (main_vrs 386 | hand_data 0.5 | ground_truth 0.4; skipped: video_main_rgb 7).
DATASET_NAME="hot3d_quest"
SLUG="hot3d-quest"
MANIFEST_NAME="HOT3D-Quest_download_urls.json"
DEFAULT_GROUPS="main_vrs,ground_truth,hand_data"
source "$(dirname "$0")/aria_common.sh"
aria_dataset_main "$@"
