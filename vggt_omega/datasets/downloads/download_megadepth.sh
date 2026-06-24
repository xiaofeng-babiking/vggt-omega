#!/usr/bin/env bash
# MegaDepth v1 -- Cornell (Li & Snavely). Open direct download (CC BY 4.0).
# Homepage: https://www.cs.cornell.edu/projects/megadepth/
# Storage: v1 images+depth ~199 GB; optional SfM models ~667 GB (so ~866 GB if both).
DATASET_NAME="megadepth"
source "$(dirname "$0")/common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "megadepth")"
log "Target: $DEST_DIR"

BASE="https://www.cs.cornell.edu/projects/megadepth/dataset"

log "Fetching MegaDepth v1 (~199 GB)"
fetch "$BASE/Megadepth_v1/MegaDepth_v1.tar.gz" "$DEST_DIR/MegaDepth_v1.tar.gz"

log "Fetching train/val + test lists"
fetch "$BASE/data_lists/train_val_list.tar.gz" "$DEST_DIR/train_val_list.tar.gz"
fetch "$BASE/data_lists/test_lists.tar.gz"      "$DEST_DIR/test_lists.tar.gz"

# SfM models (~667 GB) are optional and large; uncomment if you need them.
# fetch "$BASE/MegaDepth_SfM/MegaDepth_SfM_v1.tar.xz" "$DEST_DIR/MegaDepth_SfM_v1.tar.xz"

log "Extracting archives"
for f in "$DEST_DIR"/*.tar.gz; do tar -xzf "$f" -C "$DEST_DIR"; done

log "Done. MEGADEPTH_DIR=$DEST_DIR"
