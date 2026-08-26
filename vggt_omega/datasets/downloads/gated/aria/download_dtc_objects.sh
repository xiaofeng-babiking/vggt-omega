#!/usr/bin/env bash
# DTC Object Explorer -- the Digital Twin Catalog object library (GLB meshes), via dtc.projectaria.com (no email gate).
# Docs: https://www.projectaria.com/datasets/dtc/  License: CC BY-SA (per object)
# 2,399 objects (1,999 DTC + 400 ADT), 5 files each. Full set 152 GB; DEFAULT_GROUPS ~118 GB.
# Groups (GB): 3d-asset_glb 118 | 3d-asset-manifold_glb 33 | 3d-asset-manifold-10k_glb 0.7 | metadata/license ~0
# Layout: $DEST/<release>/<object_uid>/<files>
EXPLORER_BASE="${EXPLORER_BASE:-https://dtc.projectaria.com}"
DATASET_NAME="dtc_objects"
SLUG="objects"
DEST_SUBDIR="dtc_objects"
MANIFEST_NAME="DTC_Objects_download_urls.json"
DEFAULT_GROUPS="3d-asset_glb,metadata,license"
source "$(dirname "$0")/aria_common.sh"
aria_dataset_main "$@"
