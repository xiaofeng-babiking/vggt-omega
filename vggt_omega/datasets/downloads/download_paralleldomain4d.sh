#!/usr/bin/env bash
# ParallelDomain-4D (PD4D) -- public TRI S3 buckets. ~2.3 TB (+ 3.2 TB extra modalities).
# Storage: basic ~2.3 TB; +OtherModalities ~3.2 TB (so ~5.5 TB if both).
# Homepage: https://gcd.cs.columbia.edu/#datasets
DATASET_NAME="paralleldomain4d"
source "$(dirname "$0")/common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "paralleldomain4d")"
log "Target: $DEST_DIR"

log "Fetching basic modalities (~2.3 TB)"
fetch "https://tri-ml-public.s3.amazonaws.com/datasets/ParallelDomain-4D.tar" \
    "$DEST_DIR/ParallelDomain-4D.tar"

# Additional modalities (~3.2 TB) -- uncomment to merge in.
# fetch "https://s3.us-east-1.amazonaws.com/tri-ml-public.s3.amazonaws.com/urp/datasets/tcow/ParallelDomain-4D-OtherModalities.tar" \
#     "$DEST_DIR/ParallelDomain-4D-OtherModalities.tar"

log "Extracting"
for f in "$DEST_DIR"/ParallelDomain-4D*.tar; do tar -xf "$f" -C "$DEST_DIR"; done

log "Done. PD4D_DIR=$DEST_DIR"
