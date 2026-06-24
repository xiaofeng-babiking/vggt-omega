#!/usr/bin/env bash
# HOT3D -- Meta hand-object tracking. Email-gated (Project Aria) for the full VRS set.
# Homepage: https://facebookresearch.github.io/hot3d/
# Storage: full VRS dataset ~1.5 TB (Aria + Quest3 sequences); HOT3D-Clips subset ~80 GB.
DATASET_NAME="hot3d"
source "$(dirname "$0")/common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "hot3d")"
URLS_JSON="${URLS_JSON:-$DEST_DIR/Hot3DAssets_download_urls.json}"

if [ ! -f "$URLS_JSON" ]; then
    manual_gate "https://www.projectaria.com/datasets/hot3D/" <<EOF
HOT3D (full VRS dataset) is email-gated; links last ~14 days:
1. Accept the HOT3D license and submit your email on the dataset page.
2. Save the served 'Hot3DAssets_download_urls.json' (and Aria/Quest JSONs) as:
   $URLS_JSON   (or set URLS_JSON=<path>).
3. Re-run this script (its uv env pins projectaria_tools==1.5.1 + vrs automatically).
4. (Subset alternative -- HOT3D-Clips, ungated: HF dataset 'bop-benchmark/hot3d'.)
EOF
fi

require_cmd git
uv_sync   # isolated uv env: envs/hot3d (projectaria_tools==1.5.1, vrs)
REPO_DIR="$DEST_DIR/_hot3d_repo"
[ -d "$REPO_DIR/.git" ] || git clone https://github.com/facebookresearch/hot3d "$REPO_DIR"

log "Downloading HOT3D via the toolkit downloader (--sequence_name all)"
uv_py "$REPO_DIR/hot3d/hot3d/data_downloader/dataset_downloader_base_main.py" \
    -c "$URLS_JSON" \
    -o "$DEST_DIR" \
    --sequence_name "${SEQUENCE_NAME:-all}"

log "Done. HOT3D_DIR=$DEST_DIR"
