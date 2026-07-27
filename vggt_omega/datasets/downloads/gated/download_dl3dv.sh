#!/usr/bin/env bash
# DL3DV-10K frames + poses. Gated HuggingFace (auto-approved after clicking
# "Agree and access" on https://huggingface.co/datasets/DL3DV/DL3DV-ALL-<RES>).
# Repo: https://github.com/DL3DV-10K/Dataset
# Storage: per resolution -- 480P ~730 GB, 960P ~2.8 TB, 2K ~11 TB, 4K ~44 TB.
#
# We deliberately bypass the upstream scripts/download.py (huggingface_hub):
# these zips are Xet-backed and the python hf client fails on them through
# hf-mirror, while plain ranged HTTP (wget/aria2c) works. So: read the scene
# list from the upstream repo's cache/DL3DV-valid.csv and pull each
# <batch>/<hash>.zip straight from $HF_ENDPOINT with fetch_aria2 (segmented,
# resumable, direct -- no proxy), then unzip. Idempotent: an extracted scene
# dir means done; partial zips resume via aria2c control files.
#
# Layout: $DEST/<RES>/<batch>/<hash>/{images_4,...}
# Tunables (env):
#   RES=960P       480P | 960P | 2K | 4K
#   SUBSET=all     "all" (batches 1K..11K, 10221 scenes; the CSV's extra
#                  'problem' batch has no downloadable zips) or a comma/space
#                  list of batches, e.g. "1K,2K"
#   WORKERS=8      concurrent scenes (each scene is an 8-way aria2c)
#   LIMIT=0        >0 caps scenes per batch (smoke tests)
DATASET_NAME="dl3dv"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "dl3dv")"
log "Target: $DEST_DIR"

require_cmd git
require_cmd wget
require_cmd unzip

# Direct network only: the lab proxy stalls/breaks TLS on bulk transfers, and
# hf-mirror + cas-bridge (the Xet CDN behind the resolve redirects) are both
# reachable direct from this box.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

RES="${RES:-960P}"
SUBSET="${SUBSET:-all}"
WORKERS="${WORKERS:-8}"
LIMIT="${LIMIT:-0}"

HF_REPO="DL3DV/DL3DV-ALL-$RES"
BASE_URL="${HF_ENDPOINT%/}/datasets/$HF_REPO/resolve/main"
log "Pulling $HF_REPO via $BASE_URL (subset=$SUBSET workers=$WORKERS)"

# Gated repo: need a HF token from an account that accepted the Terms of Use.
TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
[ -n "$TOKEN" ] || manual_gate "https://huggingface.co/datasets/$HF_REPO" <<'EOF'
Log in to HuggingFace, click "Agree and access" on the repo above
(auto-approved), then put a token on this box, either:
  huggingface-cli login          # writes ~/.cache/huggingface/token
or:
  export HF_TOKEN=hf_xxx
and re-run.
EOF

# Scene list: cache/DL3DV-valid.csv from the upstream GitHub repo (kept next
# to the data; raw.githubusercontent.com is not reliably reachable direct).
REPO_DIR="$DEST_DIR/_dl3dv_repo"
META="$REPO_DIR/cache/DL3DV-valid.csv"
if [ ! -f "$META" ]; then
    git clone --depth 1 https://github.com/DL3DV-10K/Dataset "$REPO_DIR" \
        || die "cannot fetch the scene list -- clone https://github.com/DL3DV-10K/Dataset to $REPO_DIR manually"
fi
[ -f "$META" ] || die "no cache/DL3DV-valid.csv in $REPO_DIR"

if [ "$SUBSET" = "all" ]; then
    BATCHES=(1K 2K 3K 4K 5K 6K 7K 8K 9K 10K 11K)
else
    IFS=', ' read -ra BATCHES <<<"$SUBSET"
fi

OUT_DIR="$DEST_DIR/$RES"
STATE_DIR="$DEST_DIR/.cache"
LIST="$STATE_DIR/pending_$RES.txt"
FAIL_LOG="$STATE_DIR/failed_$RES.txt"
mkdir -p "$OUT_DIR" "$STATE_DIR"
: >"$LIST"
: >"$FAIL_LOG"

total=0 pending=0
for b in "${BATCHES[@]}"; do
    while read -r h; do
        total=$((total + 1))
        if [ ! -d "$OUT_DIR/$b/$h" ]; then
            printf '%s %s\n' "$b" "$h" >>"$LIST"
            pending=$((pending + 1))
        fi
    done < <(awk -F, -v b="$b" -v lim="$LIMIT" \
                 'NR>1 && $3==b {print $1; if (lim>0 && ++n>=lim) exit}' "$META")
done
log "subset=$SUBSET resolution=$RES: $total scenes, $((total - pending)) done, $pending to download"
if [ "$pending" -eq 0 ]; then
    log "Nothing to do. DL3DV_DIR=$OUT_DIR"
    exit 0
fi

# Fail fast if the token/gate is bad: 1-byte ranged GET on the first pending zip.
read -r b h <"$LIST"
wget -q --tries=2 --timeout=30 --header="Authorization: Bearer $TOKEN" \
     --header="Range: bytes=0-0" -O /dev/null "$BASE_URL/$b/$h.zip" \
    || manual_gate "https://huggingface.co/datasets/$HF_REPO" <<'EOF'
The HF token was rejected for this repo. Log in to HuggingFace with the
token's account, open the repo page above, click "Agree and access"
(auto-approved), and re-run.
EOF

# Download + extract one scene. Extracts to a temp dir and mv's the scene dir
# into place, so an existing $OUT/<batch>/<hash> dir always means "complete".
dl3dv_scene() {
    local batch="$1" hash="$2"
    local sdir="$DL3DV_OUT/$batch/$hash" zip="$DL3DV_OUT/$batch/$hash.zip"
    [ -d "$sdir" ] && return 0
    mkdir -p "$DL3DV_OUT/$batch"
    local FETCH_HEADERS=("Authorization: Bearer $DL3DV_TOKEN")
    if ! fetch_aria2 "$DL3DV_BASE/$batch/$hash.zip" "$zip"; then
        printf '%s/%s\n' "$batch" "$hash" >>"$DL3DV_FAIL"
        return 1
    fi
    local tmp="$DL3DV_OUT/$batch/.extract_$hash"
    rm -rf "$tmp"
    if unzip -q "$zip" -d "$tmp" && [ -d "$tmp/$hash" ]; then
        mv "$tmp/$hash" "$sdir"
        rm -rf "$tmp" "$zip"
    else
        warn "unzip failed for $batch/$hash.zip -- deleting corrupt zip (re-run to re-download)"
        rm -rf "$tmp" "$zip" "$zip.aria2"
        printf '%s/%s\n' "$batch" "$hash" >>"$DL3DV_FAIL"
        return 1
    fi
}

export DL3DV_OUT="$OUT_DIR" DL3DV_BASE="$BASE_URL" DL3DV_TOKEN="$TOKEN" DL3DV_FAIL="$FAIL_LOG"
export DATASET_NAME HF_ENDPOINT FETCH_RETRIES="${FETCH_RETRIES:-20}"
export -f dl3dv_scene fetch_aria2 fetch_retry fetch log warn err die

xargs -n 2 -P "$WORKERS" bash -c 'dl3dv_scene "$@"' _ <"$LIST" || true

failed=$(wc -l <"$FAIL_LOG")
if [ "$failed" -gt 0 ]; then
    err "$failed/$pending scenes failed (see $FAIL_LOG) -- re-run to retry (partials resume)"
    exit 1
fi
log "Done: $pending scenes downloaded. DL3DV_DIR=$OUT_DIR"
