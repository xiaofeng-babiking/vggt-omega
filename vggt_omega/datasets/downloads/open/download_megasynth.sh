#!/usr/bin/env bash
# MegaSynth -- public HuggingFace dataset (700K synthetic scenes), ~4 TB
# in ~100 x 40 GB split zips.
# HF: hwjiang/MegaSynth  Homepage: https://hwjiang1510.github.io/MegaSynth/
#
# We deliberately bypass the hf CLI: these zips are Xet-backed and the python
# hf client fails on them through hf-mirror ("Local entry not found"), while
# the plain resolve redirect chain (aria2c/wget) works, resumes, and is much
# faster segmented. File list + byte sizes come from the mirror's tree API;
# files whose on-disk size matches are skipped, so the script is idempotent
# and also repairs/finishes partial pulls from the old hf-CLI path.
#
# Tunables (env):
#   WORKERS=2      concurrent splits (each split is an 8-way aria2c)
DATASET_NAME="megasynth"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "megasynth")"
log "Target: $DEST_DIR"

require_cmd wget
require_cmd python3
require_cmd stat

# Direct network only (see download_dl3dv.sh: same hf-mirror + cas-bridge route).
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

WORKERS="${WORKERS:-2}"
# Per-download speed divides across WORKERS on the throttled cas-bridge route;
# the default 800K floor would abort healthy transfers (see common.sh).
export ARIA2_SPEED_FLOOR="${ARIA2_SPEED_FLOOR:-100K}"

REPO="hwjiang/MegaSynth"
BASE_URL="${HF_ENDPOINT%/}/datasets/$REPO/resolve/main"
log "Pulling $REPO via $BASE_URL (workers=$WORKERS)"

STATE_DIR="$DEST_DIR/.cache"
TREE_JSON="$STATE_DIR/tree.json"
LIST="$STATE_DIR/pending.txt"
FAIL_LOG="$STATE_DIR/failed.txt"
mkdir -p "$STATE_DIR"

# File list with sizes from the tree API (102 entries as of 2026-07; the API
# pages at 1000, so a single request covers this repo).
fetch "${HF_ENDPOINT%/}/api/datasets/$REPO/tree/main" "$TREE_JSON" \
    || die "cannot list $REPO via ${HF_ENDPOINT%/}/api/datasets/$REPO/tree/main"

: >"$LIST"
: >"$FAIL_LOG"
total=0 pending=0
while read -r size path; do
    total=$((total + 1))
    if [ -f "$DEST_DIR/$path" ] && [ ! -f "$DEST_DIR/$path.aria2" ] \
        && [ "$(stat -c%s "$DEST_DIR/$path")" = "$size" ]; then
        continue
    fi
    printf '%s %s\n' "$size" "$path" >>"$LIST"
    pending=$((pending + 1))
done < <(python3 -c "
import json, sys
for e in json.load(open('$TREE_JSON')):
    if e['type'] == 'file':
        print(e['size'], e['path'])
")
log "$total files, $((total - pending)) complete (size-verified), $pending to download"
if [ "$pending" -eq 0 ]; then
    log "Nothing to do. MEGASYNTH_DIR=$DEST_DIR"
    exit 0
fi

# Download one file and verify its byte size against the tree API's.
megasynth_file() {
    local size="$1" path="$2"
    local out="$MS_OUT/$path"
    if [ -f "$out" ] && [ ! -f "$out.aria2" ] && [ "$(stat -c%s "$out")" = "$size" ]; then
        return 0
    fi
    if ! fetch_aria2 "$MS_BASE/$path" "$out"; then
        printf '%s\n' "$path" >>"$MS_FAIL"
        return 1
    fi
    if [ "$(stat -c%s "$out")" != "$size" ]; then
        warn "size mismatch for $path ($(stat -c%s "$out") != $size) -- re-run to repair"
        printf '%s\n' "$path" >>"$MS_FAIL"
        return 1
    fi
}

export MS_OUT="$DEST_DIR" MS_BASE="$BASE_URL" MS_FAIL="$FAIL_LOG"
# 40 GB splits need a much larger retry budget than the common default 20:
# cas-bridge drops connections every few GB, and at throttled speeds a single
# split spans days -- 20 aria2c restarts get exhausted mid-file.
export DATASET_NAME HF_ENDPOINT ARIA2_SPEED_FLOOR FETCH_RETRIES="${FETCH_RETRIES:-200}"
export -f megasynth_file fetch_aria2 fetch_retry fetch log warn err die

xargs -n 2 -P "$WORKERS" bash -c 'megasynth_file "$@"' _ <"$LIST" || true

failed=$(wc -l <"$FAIL_LOG")
if [ "$failed" -gt 0 ]; then
    err "$failed/$pending files failed (see $FAIL_LOG) -- re-run to retry (partials resume)"
    exit 1
fi
log "Done: $pending files downloaded. MEGASYNTH_DIR=$DEST_DIR"
