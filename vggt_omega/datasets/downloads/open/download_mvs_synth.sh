#!/usr/bin/env bash
# MVS-Synth (DeepMVS) -- public HuggingFace dataset (research/educational use).
# HF: phuang17/MVS-Synth  Homepage: https://phuang17.github.io/DeepMVS/mvs-synth.html
# Storage: GTAV_1080 ~34 GB, GTAV_720 ~16 GB, GTAV_540 ~9 GB (pick one via RES).
DATASET_NAME="mvs_synth"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "mvs_synth")"
log "Target: $DEST_DIR"

# Resolution: 1080 (34GB), 720 (16GB), or 540 (9GB). Default 1080.
RES="${RES:-1080}"
# huggingface.co URLs are auto-rerouted to $HF_ENDPOINT (default hf-mirror.com) by fetch().
BASE="https://huggingface.co/datasets/phuang17/MVS-Synth/resolve/main"

log "Fetching GTAV_${RES}.tar.gz"
fetch_retry "$BASE/GTAV_${RES}.tar.gz" "$DEST_DIR/GTAV_${RES}.tar.gz"

log "Extracting"
tar -xzf "$DEST_DIR/GTAV_${RES}.tar.gz" -C "$DEST_DIR"

log "Done. MVS_SYNTH_DIR=$DEST_DIR"
