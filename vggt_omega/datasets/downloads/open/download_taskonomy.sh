#!/usr/bin/env bash
# Taskonomy -- via the Omnidata download tool (uses aria2). ~11 TB full.
# Storage: fullplus ~11.16 TB; smaller subsets via SUBSET= (medium ~2.4 TB, tiny ~115 GB,
#          debug a few GB).
# Homepage: http://taskonomy.stanford.edu/  Tool: pip install omnidata-tools
DATASET_NAME="taskonomy"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "taskonomy")"
log "Target: $DEST_DIR"

require_cmd aria2c "Install with: sudo apt-get install -y aria2"
uv_sync   # isolated uv env: envs/taskonomy (omnidata-tools)

# subset: debug | tiny | medium | full | fullplus (default fullplus = everything)
SUBSET="${SUBSET:-fullplus}"
log "Downloading Taskonomy via omnitools (subset=$SUBSET); --agree accepts the terms"
uv_tool omnitools.download all \
    --components taskonomy \
    --subset "$SUBSET" \
    --dest "$DEST_DIR" \
    --connections_total 40 \
    --agree \
    "${@:2}"

log "Done. TASKONOMY_DIR=$DEST_DIR"
