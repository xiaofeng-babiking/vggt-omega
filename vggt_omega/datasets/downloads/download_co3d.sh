#!/usr/bin/env bash
# CO3Dv2 (Common Objects in 3D v2) -- Meta / FAIR.
# Open download via the official repo's batch downloader (FAIR CDN). ~5.5 TB full.
# Storage: ~5.5 TB full; --single_sequence_subset ~8.9 GB.
# Repo: https://github.com/facebookresearch/co3d
DATASET_NAME="co3d"
source "$(dirname "$0")/common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "co3d")"
log "Target: $DEST_DIR"

require_cmd git
uv_sync   # isolated uv env: envs/co3d (requests, tqdm)

REPO_DIR="$DEST_DIR/_co3d_repo"
[ -d "$REPO_DIR/.git" ] || git clone https://github.com/facebookresearch/co3d "$REPO_DIR"

# --single_sequence_subset (~8.9 GB) for a quick subset; drop it for the full set.
log "Running official downloader (add --single_sequence_subset for the 8.9 GB subset)"
uv_py "$REPO_DIR/co3d/download_dataset.py" \
    --download_folder "$DEST_DIR" \
    "${@:2}"

log "Done. Expected layout matches datasets/vendors/co3d.py (CO3D_DIR=$DEST_DIR)."
