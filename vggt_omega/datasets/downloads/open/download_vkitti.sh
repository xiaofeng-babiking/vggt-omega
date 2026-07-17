#!/usr/bin/env bash
# Virtual KITTI v2 (and v1) -- NAVER LABS Europe. Open direct download (CC BY-NC-SA 3.0).
# Homepage: https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/
DATASET_NAME="vkitti"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "vkitti")"
log "Target: $DEST_DIR"

V2="https://download.europe.naverlabs.com/virtual_kitti_2.0.3"
log "Fetching Virtual KITTI 2.0.3 archives"
for kind in rgb depth classSegmentation instanceSegmentation textgt \
            forwardFlow backwardFlow forwardSceneFlow backwardSceneFlow; do
    case "$kind" in
        textgt) ext="tar.gz" ;;      # upstream ships textgt as .tar.gz; everything else .tar
        *)      ext="tar" ;;
    esac
    fetch "$V2/vkitti_2.0.3_${kind}.${ext}" "$DEST_DIR/vkitti_2.0.3_${kind}.${ext}"
done

log "Extracting"
for f in "$DEST_DIR"/vkitti_2.0.3_*.tar*; do
    case "$f" in
        *.tar.gz) tar -xzf "$f" -C "$DEST_DIR" ;;
        *.tar)    tar -xf  "$f" -C "$DEST_DIR" ;;
    esac
done

log "Done. VKITTI_DIR=$DEST_DIR"
log "For v1.3.1 see: https://download.europe.naverlabs.com/virtual-kitti-1.3.1/"
