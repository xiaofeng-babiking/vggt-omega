#!/usr/bin/env bash
# TartanGround. Open (CC-BY-4.0) via the same 'tartanair' Python package. ~16 TB full.
# Storage: ~16 TB full (all envs/modalities); a single env+modality subset is tens of GB.
# Homepage: https://tartanair.org/tartanground.html  Repo: https://github.com/castacks/tartanairpy
DATASET_NAME="tartanground"
source "$(dirname "$0")/common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "tartanground")"
log "Target: $DEST_DIR"

uv_sync   # isolated uv env: envs/tartanground (tartanair)

log "Downloading via tartanairpy.download_ground (edit env/modality lists as needed)"
uv_py - "$DEST_DIR" <<'PY'
import sys
import tartanair as ta
root = sys.argv[1]
ta.init(root)
ta.download_ground(
    env=[],                                   # [] = all environments (~16 TB)
    version=['omni'],                         # 'omni' | 'diff' | 'anymal'
    modality=['image', 'depth', 'seg', 'lidar', 'imu'],
    camera_name=['lcam_front'],
    unzip=False,
)
PY

log "Done. TARTANGROUND_DIR=$DEST_DIR"
