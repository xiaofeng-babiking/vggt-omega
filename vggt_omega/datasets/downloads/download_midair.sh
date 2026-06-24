#!/usr/bin/env bash
# Mid-Air -- aerial synthetic, University of Liege. Open (Creative Commons).
# Homepage: https://midair.ulg.ac.be/download.html
# Storage: ~1.5 TB full (all trajectories x RGB/depth/normals/segmentation/stereo);
#          a single climate-set RGB+depth subset is ~50-100 GB.
DATASET_NAME="midair"
source "$(dirname "$0")/common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "midair")"
CONFIG="${DOWNLOAD_CONFIG:-$DEST_DIR/download_config.txt}"

if [ ! -f "$CONFIG" ]; then
    manual_gate "https://midair.ulg.ac.be/download.html" <<EOF
Mid-Air uses a checkbox-generated config of direct URLs (no account needed):
1. Open the download page and tick the trajectories/modalities you want.
2. It generates a 'download_config.txt' with the direct file URLs.
3. Save it as: $CONFIG   (or set DOWNLOAD_CONFIG=<path>).
4. Re-run this script.
EOF
fi

require_cmd git
uv_sync   # isolated uv env: envs/midair (requests, tqdm)
REPO_DIR="$DEST_DIR/_midair_repo"
[ -d "$REPO_DIR/.git" ] || git clone https://github.com/montefiore-ai/midair-dataset "$REPO_DIR" \
    || warn "Could not clone helper repo; download_dataset.py may ship with the config instead."

DL_SCRIPT="$REPO_DIR/download_dataset.py"
[ -f "$DL_SCRIPT" ] || DL_SCRIPT="$DEST_DIR/download_dataset.py"
[ -f "$DL_SCRIPT" ] || die "download_dataset.py not found; download it from the Mid-Air page and place it in $DEST_DIR."

log "Downloading per download_config.txt"
uv_py "$DL_SCRIPT" --download_config "$CONFIG" --output_dir "$DEST_DIR" --max_threads 4

log "Done. MIDAIR_DIR=$DEST_DIR"
