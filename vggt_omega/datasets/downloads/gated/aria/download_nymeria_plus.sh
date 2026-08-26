#!/usr/bin/env bash
# NymeriaPlus (CVPR'26 re-release of Nymeria) -- Meta Project Aria, via explorer.projectaria.com.
# Docs: https://github.com/facebookresearch/nymeria_dataset  (License: CC BY-NC 4.0 per upstream)
# 1,100 sequences, 340 h, 4 Aria devices each (head, observer, 2 wrists). Full set 67.7 TB; DEFAULT_GROUPS ~37.8 TB.
# Groups (TB): recording_head_data_data_vrs 18.4 | recording_observer_data_data_vrs 17.5 | slam 1.7 |
#   object_bounding_box 0.125 | object_mesh 0.04 | metadata_json/LICENSE ~0 |
#   skipped: slam_semidense_observations 11.6 | recording_lwrist/rwrist_data_data_vrs 6.3+6.1 | audio 2.5 |
#   body_raw 1.3 | body_processed 0.35 | timesync_and_imu 0.35 | eye_tracking 0.44 | narration ~0 | video_main_rgb 1.1 (preview)
# NOTE: supersedes download_nymeria.sh (v0.0); files differ byte-wise, so do not download both unless you need v0.0.
# Examples: DATA_GROUPS=all ./download_nymeria_plus.sh | EXCLUDE_GROUPS=recording_observer_data_data_vrs ./download_nymeria_plus.sh
DATASET_NAME="nymeria_plus"
SLUG="nymeria_plus"
MANIFEST_NAME="NymeriaPlus_download_urls.json"
DEFAULT_GROUPS="recording_head_data_data_vrs,recording_observer_data_data_vrs,slam,object_bounding_box,object_mesh,metadata_json,LICENSE"
source "$(dirname "$0")/aria_common.sh"
aria_dataset_main "$@"
