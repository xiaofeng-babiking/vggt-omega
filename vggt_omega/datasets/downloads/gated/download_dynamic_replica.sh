#!/usr/bin/env bash
# Dynamic Replica -- Meta. Requires accepting the license to obtain links.json.
# Project: https://dynamic-stereo.github.io/
# Storage: ~2.3 TB total -- train 1.8 TB, test 328 GB, valid 106 GB, real 152 MB.
DATASET_NAME="dynamic_replica"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "dynamic_replica")"
LINKS="${LINKS_JSON:-$DEST_DIR/links.json}"

if [ ! -f "$LINKS" ]; then
    manual_gate "https://dynamic-stereo.github.io/" <<EOF
1. Open the project page and go to the "data" tab.
2. Accept the license agreement; this yields a 'links.json' file.
3. Save it as: $LINKS
   (or point LINKS_JSON=<path> at it).
4. Re-run this script.
EOF
fi

require_cmd git
uv_sync   # isolated uv env: envs/dynamic_replica (requests, tqdm)

REPO_DIR="$DEST_DIR/_dynamic_stereo"
[ -d "$REPO_DIR/.git" ] || git clone https://github.com/facebookresearch/dynamic_stereo "$REPO_DIR"

log "Downloading splits (train 1.8TB, test 328GB, valid 106GB, real 152MB)"
uv_py "$REPO_DIR/scripts/download_dynamic_replica.py" \
    --link_list_file "$LINKS" \
    --download_folder "$DEST_DIR" \
    --download_splits "${@:2}"   # e.g. real valid test train

log "Done. DYNAMIC_REPLICA_DIR=$DEST_DIR"
