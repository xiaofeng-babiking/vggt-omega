#!/usr/bin/env bash
# UnrealStereo4K (SMD-Nets) -- public AWS S3. Per-scene zips ~67-76 GB each.
# Storage: ~600 GB full (9 scenes x ~67-76 GB at 4K); 960x540 HF variant is far smaller.
# Repo: https://github.com/fabiotosi92/SMD-Nets
DATASET_NAME="unrealstereo4k"
source "$(dirname "$0")/common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "unrealstereo4k")"
log "Target: $DEST_DIR"

BASE="https://s3.eu-central-1.amazonaws.com/avg-projects/smd_nets"
log "Fetching UnrealStereo4K scenes 00000..00008 (4K; ~600 GB total)"
for i in 00000 00001 00002 00003 00004 00005 00006 00007 00008; do
    fetch "$BASE/UnrealStereo4K_$i.zip" "$DEST_DIR/UnrealStereo4K_$i.zip"
done

log "Extracting"
for f in "$DEST_DIR"/UnrealStereo4K_*.zip; do unzip -o "$f" -d "$DEST_DIR" || warn "unzip failed for $f"; done

log "Done. UNREALSTEREO4K_DIR=$DEST_DIR"
log "Quarter-res (960x540) variant available at HF: fabiotosi92/UnrealStereo4K"
