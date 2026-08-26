#!/usr/bin/env bash
# Nymeria v0.0 (CVPR'24 original release) -- Meta Project Aria, via explorer.projectaria.com.
# Docs: https://www.projectaria.com/datasets/nymeria/  License: CC BY-NC 4.0
# 1,100 sequences, 340 h. Full set 51.3 TB; DEFAULT_GROUPS ~35.2 TB.
# Groups (TB): recording_head_data_data_vrs 17.1 | recording_observer_data_data_vrs 16.3 | recording_head 0.76 (motion/et vrs + mps) |
#   recording_observer 0.62 | skipped: semidense_observations 11.6 | body_xdata_mvnx 2.0 | body_motion 1.1 |
#   recording_lwrist/rwrist 0.45+0.47 | video_main_rgb 1.1 (preview) | narration_* ~0
# NOTE: superseded by download_nymeria_plus.sh (same 1,100 sequences, re-packaged + more annotations); not in launch_all.sh lanes.
DATASET_NAME="nymeria"
SLUG="nymeria"
MANIFEST_NAME="Nymeria_download_urls.json"
DEFAULT_GROUPS="recording_head_data_data_vrs,recording_observer_data_data_vrs,recording_head,recording_observer,metadata_json,LICENSE"
source "$(dirname "$0")/aria_common.sh"
aria_dataset_main "$@"
