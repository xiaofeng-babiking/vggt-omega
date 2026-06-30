#!/usr/bin/env bash
# MegaSynth -- public HuggingFace dataset (700K synthetic scenes).
# HF: hwjiang/MegaSynth  Homepage: https://hwjiang1510.github.io/MegaSynth/
# Storage: ~7 TB full (700k synthetic scenes).
DATASET_NAME="megasynth"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "megasynth")"
log "Target: $DEST_DIR"

uv_sync   # isolated uv env: envs/megasynth (huggingface_hub[cli])
hf_download "hwjiang/MegaSynth" "$DEST_DIR" "${@:2}"

log "Done. MEGASYNTH_DIR=$DEST_DIR"
