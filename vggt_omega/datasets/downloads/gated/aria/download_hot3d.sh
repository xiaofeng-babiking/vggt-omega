#!/usr/bin/env bash
# HOT3D -- Meta hand-object tracking (Aria + Quest 3). Umbrella script:
#   1. sequences: HOT3D-Aria + HOT3D-Quest straight from explorer.projectaria.com
#      (no email gate) via ./download_hot3d_aria.sh and ./download_hot3d_quest.sh
#      -> $DOWNLOAD_ROOT/aria/hot3d-aria, $DOWNLOAD_ROOT/aria/hot3d-quest
#   2. object models (33 objects, Hot3DAssets_download_urls.json): NOT served by
#      the explorer; still needs the email form on https://www.projectaria.com/datasets/hot3D/
#      Save that JSON as $HOT3D_ASSETS_JSON and this script fetches the assets with
#      the official toolkit downloader (uv env envs/hot3d) into $DOWNLOAD_ROOT/aria/hot3d.
# Homepage: https://facebookresearch.github.io/hot3d/   License: https://www.projectaria.com/datasets/hot3d/license/
# Storage: sequences ~0.8 TB with the default groups (see the two scripts), assets ~1 GB.
# Subset alternative (ungated, ~80 GB): HF dataset 'bop-benchmark/hot3d' (HOT3D-Clips).
DATASET_NAME="hot3d"
source "$(dirname "$0")/../../common.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

DEST_DIR="$(resolve_dest "${1:-}" "aria/hot3d")"
HOT3D_ASSETS_JSON="${HOT3D_ASSETS_JSON:-$DEST_DIR/Hot3DAssets_download_urls.json}"

log "sequences: HOT3D-Aria then HOT3D-Quest via the explorer API"
"$HERE/download_hot3d_aria.sh"
"$HERE/download_hot3d_quest.sh"

if [ ! -f "$HOT3D_ASSETS_JSON" ]; then
    warn "object models skipped: no $HOT3D_ASSETS_JSON"
    warn "  -> accept the HOT3D license at https://www.projectaria.com/datasets/hot3D/, save the served"
    warn "     'Hot3DAssets_download_urls.json' at that path (or HOT3D_ASSETS_JSON=<path>) and re-run."
    log "Done (sequences only). HOT3D_DIR=$DEST_DIR"
    exit 0
fi

require_cmd git
uv_sync   # isolated uv env: envs/hot3d (projectaria_tools==1.5.1, vrs)
REPO_DIR="$DEST_DIR/_hot3d_repo"
[ -d "$REPO_DIR/.git" ] || git clone https://github.com/facebookresearch/hot3d "$REPO_DIR"
log "Downloading HOT3D object assets via the toolkit downloader"
uv_py "$REPO_DIR/hot3d/hot3d/data_downloader/dataset_downloader_base_main.py" \
    -c "$HOT3D_ASSETS_JSON" -o "$DEST_DIR" --sequence_name "${SEQUENCE_NAME:-all}"
log "Done. HOT3D_DIR=$DEST_DIR"
