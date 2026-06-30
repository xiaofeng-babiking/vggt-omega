#!/usr/bin/env bash
# Hypersim -- Apple. Photorealistic synthetic indoor. Open (CC BY-SA 3.0). ~1.9 TB.
# Storage: ~1.9 TB downloads + similar decompressed (budget ~4 TB total during extract).
# Repo: https://github.com/apple/ml-hypersim
DATASET_NAME="hypersim"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "hypersim")"
log "Target: $DEST_DIR"

require_cmd git
uv_sync   # isolated uv env: envs/hypersim (pandas, requests, tqdm)

REPO_DIR="$DEST_DIR/_ml-hypersim"
[ -d "$REPO_DIR/.git" ] || git clone https://github.com/apple/ml-hypersim "$REPO_DIR"

log "Running official image downloader (~1.9 TB; downloads + decompresses)"
uv_py "$REPO_DIR/code/python/tools/dataset_download_images.py" \
    --downloads_dir "$DEST_DIR/downloads" \
    --decompress_dir "$DEST_DIR/scenes"

log "Done. HYPERSIM_DIR=$DEST_DIR/scenes"
