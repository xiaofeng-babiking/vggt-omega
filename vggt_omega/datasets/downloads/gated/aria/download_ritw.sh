#!/usr/bin/env bash
# Reading In The Wild (RITW) -- via explorer.projectaria.com (no email gate).
# Docs: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/ritw  License: CC BY-NC 4.0
# 987 recordings, 78.7 h (reading / non-reading activities, eye tracking). Full set 6.83 TB; DEFAULT_GROUPS ~6.6 TB.
# Groups (TB): recording_vrs 4.4 | mps 2.2 | metadata_json ~0 | skipped: video_main_rgb 0.19 (preview)
DATASET_NAME="ritw"
SLUG="ritw"
MANIFEST_NAME="Reading_In_The_Wild_download_urls.json"
DEFAULT_GROUPS="recording_vrs,metadata_json,mps"
source "$(dirname "$0")/aria_common.sh"
aria_dataset_main "$@"
