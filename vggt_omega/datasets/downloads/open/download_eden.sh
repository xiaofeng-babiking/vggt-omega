#!/usr/bin/env bash
# EDEN (Enclosed garDEN) -- synthetic. Open direct download from UvA ISIS server.
# Homepage: https://lhoangan.github.io/eden/
# Storage: ~1.0 TB full -- RGB 333 GB, Depth 241 GB, SurfaceNormal 130 GB,
#          Intrinsics ~440 GB, Segmentation 13 GB, metadata <0.3 GB.
DATASET_NAME="eden"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "eden")"
log "Target: $DEST_DIR"

BASE="https://isis-data.science.uva.nl/hale/EDEN"

# Metadata first (small), then the large modality archives.
for f in Stats.zip CameraInfo.zip LightInfo.zip; do
    log "Fetching $f"
    fetch "$BASE/$f" "$DEST_DIR/$f"
done

# Large modalities (RGB 333G, Depth 241G, SurfaceNormal 130G, Segmentation 13G).
# Comment out any you do not need.
for f in RGB.zip Depth.zip SurfaceNormal.zip Segmentation.zip; do
    log "Fetching $f"
    fetch "$BASE/$f" "$DEST_DIR/$f"
done

log "Extracting"
for f in "$DEST_DIR"/*.zip; do unzip -o "$f" -d "$DEST_DIR" || warn "unzip failed for $f"; done

log "Done. EDEN_DIR=$DEST_DIR"
