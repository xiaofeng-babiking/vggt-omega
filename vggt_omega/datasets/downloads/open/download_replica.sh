#!/usr/bin/env bash
# Replica -- FAIR / Facebook Reality Labs. Open access.
# Storage: ~12 GB (18 high-fidelity indoor scenes) + ~50 MB Habitat configs.
# Repo: https://github.com/facebookresearch/Replica-Dataset
DATASET_NAME="replica"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "replica")"
log "Target: $DEST_DIR"

require_cmd wget
command -v unpigz >/dev/null 2>&1 || warn "unpigz (pigz) not found; install pigz for faster extraction."

BASE="https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0"
log "Fetching split tarball parts (partaa..partq)"
for p in a b c d e f g h i j k l m n o p q; do
    fetch "$BASE/replica_v1_0.tar.gz.parta$p" "$DEST_DIR/replica_v1_0.tar.gz.parta$p"
done

log "Reassembling and extracting"
if command -v unpigz >/dev/null 2>&1; then
    cat "$DEST_DIR"/replica_v1_0.tar.gz.parta* | unpigz | tar -x -C "$DEST_DIR"
else
    cat "$DEST_DIR"/replica_v1_0.tar.gz.parta* | tar -xz -C "$DEST_DIR"
fi

log "Fetching Habitat configs"
fetch "http://dl.fbaipublicfiles.com/habitat/Replica/additional_habitat_configs.zip" \
    "$DEST_DIR/additional_habitat_configs.zip"
( cd "$DEST_DIR" && unzip -o additional_habitat_configs.zip ) || warn "unzip failed; install unzip"

log "Done. REPLICA_DIR=$DEST_DIR"
