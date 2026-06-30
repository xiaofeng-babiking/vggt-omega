#!/usr/bin/env bash
# WildRGB-D. Public HuggingFace repo (ungated). ~3.37 TB zipped.
# Repo: https://github.com/wildrgbd/wildrgbd  HF: hongchi/wildrgbd
DATASET_NAME="wildrgbd"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "wildrgbd")"
log "Target: $DEST_DIR"

require_cmd git
uv_sync   # isolated uv env: envs/wildrgbd (requests, tqdm, huggingface_hub)

REPO_DIR="$DEST_DIR/_wildrgbd_repo"
[ -d "$REPO_DIR/.git" ] || git clone https://github.com/wildrgbd/wildrgbd "$REPO_DIR"

# --cat all for everything, or a single category name to limit the download.
# download.py writes relative to its own cwd, so run it from inside the repo.
CAT="${CAT:-all}"
log "Running official download.py (--cat $CAT)"
( cd "$REPO_DIR" && uv_py download.py --cat "$CAT" --output_dir "$DEST_DIR" "${@:2}" ) \
    || ( cd "$REPO_DIR" && uv_py download.py --cat "$CAT" "${@:2}" )

log "Done. WILDRGBD_DIR=$DEST_DIR"
