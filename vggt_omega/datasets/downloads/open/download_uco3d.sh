#!/usr/bin/env bash
# uCO3D (UnCommon Objects in 3D) -- Meta / FAIR. Public (ungated). ~19.3 TB full.
# Storage: ~19.3 TB full; --download_small_subset ~9.6 GB (52 videos).
# Repo: https://github.com/facebookresearch/uco3d  HF: facebook/uco3d
DATASET_NAME="uco3d"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "uco3d")"
log "Target: $DEST_DIR"

require_cmd git
uv_sync   # isolated uv env: envs/uco3d (requests, tqdm, huggingface_hub)

REPO_DIR="$DEST_DIR/_uco3d_repo"
[ -d "$REPO_DIR/.git" ] || git clone https://github.com/facebookresearch/uco3d.git "$REPO_DIR"
# Editable-install the cloned repo (and its deps) INTO the uco3d uv env.
uv_pip install -e "$REPO_DIR"

# Use --download_small_subset (~9.6 GB) for a quick test, or --use_huggingface
# to pull from facebook/uco3d instead of the CDN. Pass extra flags after the dest.
log "Running official downloader (add --download_small_subset for the 9.6 GB subset)"
# --use_huggingface: the signed fbcdn URLs in Meta's published links json
# expire and are (as of 2026-07) stale upstream -> every CDN download 403s.
# The HF mirror route (added upstream in the hf-download PR) works and
# honors HF_ENDPOINT.
uv_py "$REPO_DIR/dataset_download/download_dataset.py" \
    --download_folder "$DEST_DIR" \
    --checksum_check \
    --use_huggingface \
    "${@:2}"

log "Done."
