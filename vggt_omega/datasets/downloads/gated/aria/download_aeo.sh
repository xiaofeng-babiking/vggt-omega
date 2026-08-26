#!/usr/bin/env bash
# Aria Everyday Objects (AEO) -- via explorer.projectaria.com (no email gate).
# Docs: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_objects  License: https://www.projectaria.com/datasets/aeo/license
# 25 sequences, 0.8 h. Full set 19 GB; DEFAULT_GROUPS ~16 GB (main_vrs 13 | mps 2.4 | main_groundtruth 0.03; skipped: video_main_rgb 3).
DATASET_NAME="aeo"
SLUG="aeo"
MANIFEST_NAME="Aria_Everyday_Objects_download_urls.json"
DEFAULT_GROUPS="main_vrs,mps,main_groundtruth"
source "$(dirname "$0")/aria_common.sh"
aria_dataset_main "$@"
