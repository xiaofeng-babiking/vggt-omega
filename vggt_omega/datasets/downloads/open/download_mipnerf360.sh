#!/usr/bin/env bash
# Mip-NeRF 360 -- Barron et al., CVPR 2022. Open direct download (Google Research
# GCS bucket; no signup/license). Homepage: https://jonbarron.info/mipnerf360/
#
# Two archives on the same bucket, extracted side by side into MIPNERF360_DIR:
#   360_v2.zip           ~11.7 GB -- 7 core scenes: bicycle bonsai counter garden
#                                    kitchen room stump
#   360_extra_scenes.zip  ~4.2 GB -- the 2 extra scenes flowers + treehill
#                                    (initially withheld for licensing, later released)
# Each scene unpacks to a <scene>/ dir with full-res images/ + downsampled
# images_{2,4,8}/ and a COLMAP sparse/ model -- 9 scenes total, ~16 GB zipped.
#
# Archive selection (default both). Override to grab a subset:
#   MIPNERF360_ARCHIVES="360_v2.zip"           # 7 core scenes only
#   MIPNERF360_ARCHIVES="360_extra_scenes.zip" # flowers + treehill only
DATASET_NAME="mipnerf360"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "mipnerf360")"
log "Target: $DEST_DIR"

require_cmd unzip "Install it (e.g. 'apt-get install unzip')."

BASE="https://storage.googleapis.com/gresearch/refraw360"
ARCHIVES="${MIPNERF360_ARCHIVES:-360_v2.zip 360_extra_scenes.zip}"

# unzip wrapper: exit code 1 means "warnings, but extraction completed" -- treat
# it as success; only code >= 2 (a real error) is a failure. These >4 GB ZIP64
# archives can trip benign warnings on some unzip builds yet extract correctly.
unz() {
    local rc=0
    unzip -q -o "$@" || rc=$?
    [ "$rc" -le 1 ]
}

# shellcheck disable=SC2086  # intentional word-split of the archive list
for z in $ARCHIVES; do
    log "Fetching $z"
    fetch_retry "$BASE/$z" "$DEST_DIR/$z"
    log "Extracting $z"
    unz "$DEST_DIR/$z" -d "$DEST_DIR" \
        || warn "extract failed for $z (corrupt/partial, or unzip too old for ZIP64?)"
done

log "Done. MIPNERF360_DIR=$DEST_DIR"
