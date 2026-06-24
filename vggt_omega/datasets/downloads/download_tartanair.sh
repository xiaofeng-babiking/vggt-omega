#!/usr/bin/env bash
# TartanAir V2. Open (CC-BY-4.0) via the official 'tartanair' Python package.
# Homepage: https://tartanair.org/  Repo: https://github.com/castacks/tartanairpy
# Storage: V2 ~3 TB+ for all envs x both difficulties x all modalities; one env+modality
#          subset is tens of GB. Narrow env/modality/camera lists below to control size.
DATASET_NAME="tartanair"
source "$(dirname "$0")/common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "tartanair")"
log "Target: $DEST_DIR"

uv_sync   # isolated uv env: envs/tartanair (tartanair)

log "Downloading via tartanairpy (edit env/modality/camera lists below as needed)"
uv_py - "$DEST_DIR" <<'PY'
import sys
import tartanair as ta
root = sys.argv[1]
ta.init(root)
ta.download(
    env=[],                                   # [] = all environments
    difficulty=['easy', 'hard'],
    modality=['image', 'depth', 'seg', 'pose'],
    camera_name=['lcam_front'],
    unzip=True,
)
PY

log "Done. TARTANAIR_DIR=$DEST_DIR"
