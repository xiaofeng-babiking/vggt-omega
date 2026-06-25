#!/usr/bin/env bash
# Microsoft 7-Scenes -- indoor RGB-D relocalization benchmark (Kinect v1). Open
# direct download from Microsoft's CDN (no signup/license).
# Homepage: https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
#
# Each scene ships as a single <scene>.zip on the CDN. It unpacks to a <scene>/
# dir holding TrainSplit.txt, TestSplit.txt, a <scene>.png thumbnail, and one
# *nested* seq-NN.zip per sequence. We extract the outer zip into SEVEN_SCENES_DIR
# and then unzip every inner seq-NN.zip in place, yielding the exact layout
# SevenScenesDataset globs (SEVEN_SCENES_DIR/<scene>/seq-NN/frame-*.{color,depth}.png
# + frame-*.pose.txt).
#
# Scene selection (first match wins):
#   1. explicit names as args after the dest dir:
#        ./download_seven_scenes.sh /data/7scenes heads chess
#   2. SEVEN_SCENES="heads chess ..."   (space-separated env override)
#   3. default: all seven scenes (chess fire heads office pumpkin redkitchen
#      stairs), ~18 GB zipped.
#
# Re-runs skip scenes already extracted (a <scene>/ dir with seq-* subdirs); set
# SEVEN_SCENES_FORCE=1 to re-fetch. The outer + inner zips are removed after a
# successful extract unless SEVEN_SCENES_KEEP_ZIP=1.
#
# NOTE on depth: the upstream release provides only the raw depth-sensor frame
# (frame-*.depth.png). It does NOT include frame-*.depth.proj.png (depth
# registered to the color camera). config/seven_scenes.yaml defaults to
# depth_variant: proj, so either generate the registered depth with your own
# registration step or set depth_variant: raw. This script warns about it at the
# end -- it never fabricates a proj file.
DATASET_NAME="seven_scenes"
source "$(dirname "$0")/common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "7scenes")"
[ "$#" -gt 0 ] && shift   # drop the dest arg; anything left is scene names
log "Target: $DEST_DIR"

require_cmd unzip "Install it (e.g. 'apt-get install unzip')."

BASE="https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"

ALL_SCENES=(chess fire heads office pumpkin redkitchen stairs)

# Resolve the scene list per the precedence documented in the header.
if [ "$#" -gt 0 ]; then
    SCENES=("$@")
elif [ -n "${SEVEN_SCENES:-}" ]; then
    # shellcheck disable=SC2206  # intentional word-split of the env override
    SCENES=(${SEVEN_SCENES})
else
    SCENES=("${ALL_SCENES[@]}")
    log "No scenes given; using all ${#SCENES[@]} scenes (~18 GB zipped)."
    log "  -> pass names or SEVEN_SCENES=... to limit (e.g. 'heads chess')."
fi

ok=0; skipped=0; failed=0
for scene in "${SCENES[@]}"; do
    scene_dir="$DEST_DIR/$scene"

    if [ -d "$scene_dir" ] && compgen -G "$scene_dir/seq-*" >/dev/null && [ "${SEVEN_SCENES_FORCE:-0}" != "1" ]; then
        log "already present: $scene/ (SEVEN_SCENES_FORCE=1 to re-fetch) -- skipping"
        skipped=$((skipped + 1)); continue
    fi

    zip="$DEST_DIR/$scene.zip"
    log "Fetching $scene.zip"
    if ! fetch "$BASE/$scene.zip" "$zip"; then
        warn "download failed for $scene -- skipping"
        rm -f "$zip"; failed=$((failed + 1)); continue
    fi

    log "Extracting $scene.zip"
    if ! unzip -q -o "$zip" -d "$DEST_DIR"; then
        warn "extract failed for $scene (corrupt/partial archive?) -- skipping"
        failed=$((failed + 1)); continue
    fi

    # Unpack the nested per-sequence zips in place (seq-NN.zip -> seq-NN/).
    inner_failed=0
    shopt -s nullglob
    for seqzip in "$scene_dir"/seq-*.zip; do
        if unzip -q -o "$seqzip" -d "$scene_dir"; then
            [ "${SEVEN_SCENES_KEEP_ZIP:-0}" = "1" ] || rm -f "$seqzip"
        else
            warn "inner extract failed: $(basename "$seqzip")"
            inner_failed=$((inner_failed + 1))
        fi
    done
    shopt -u nullglob

    [ "${SEVEN_SCENES_KEEP_ZIP:-0}" = "1" ] || rm -f "$zip"

    if [ "$inner_failed" -ne 0 ]; then
        warn "$scene: $inner_failed sequence archive(s) failed to extract"
        failed=$((failed + 1)); continue
    fi
    ok=$((ok + 1))
done

log "Done. ok=$ok skipped=$skipped failed=$failed"
log "SEVEN_SCENES_DIR=$DEST_DIR  (point config/seven_scenes.yaml's SEVEN_SCENES_DIR here)"
warn "Upstream ships only raw depth (frame-*.depth.png); config defaults to"
warn "depth_variant: proj (frame-*.depth.proj.png). Generate registered depth or"
warn "set depth_variant: raw before evaluating."
[ "$failed" -eq 0 ] || warn "$failed scene(s) failed -- see warnings above."
