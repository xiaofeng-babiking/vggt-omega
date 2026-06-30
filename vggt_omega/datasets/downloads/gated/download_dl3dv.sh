#!/usr/bin/env bash
# DL3DV-10K. Gated HuggingFace (auto-approved on accepting Terms of Use).
# Repo: https://github.com/DL3DV-10K/Dataset
# Storage: per resolution -- 480P ~730 GB, 960P ~2.8 TB, 2K ~11 TB, 4K ~44 TB;
#          Sample ~5 GB (11 scenes), Benchmark ~25 GB (140 scenes).
DATASET_NAME="dl3dv"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "dl3dv")"
log "Target: $DEST_DIR"

require_cmd git
uv_sync   # isolated uv env: envs/dl3dv (huggingface_hub[cli], pandas, tqdm, requests)

cat >&2 <<EOF
NOTE: DL3DV repos are gated on HuggingFace. Before running, on each repo you use
click "Agree and access" (auto-approved), e.g.:
  - https://huggingface.co/datasets/DL3DV/DL3DV-10K-Sample      (11 scenes, free)
  - https://huggingface.co/datasets/DL3DV/DL3DV-10K-Benchmark   (140 scenes)
  - https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P        (~730 GB)  ... -960P/-2K/-4K
and authenticate once with:  uv run --project envs/dl3dv -- huggingface-cli login
(the token is stored in ~/.cache/huggingface and shared by all envs).
EOF

REPO_DIR="$DEST_DIR/_dl3dv_repo"
[ -d "$REPO_DIR/.git" ] || git clone https://github.com/DL3DV-10K/Dataset "$REPO_DIR"

# Default to the small Sample subset; override via args, e.g.:
#   download_dl3dv.sh <dest> --subset full --resolution 480P --file_type images+poses
SUBSET="${SUBSET:-sample}"
RES="${RES:-480P}"
log "Running scripts/download.py (subset=$SUBSET resolution=$RES; override with extra args)"
uv_py "$REPO_DIR/scripts/download.py" \
    --odir "$DEST_DIR" \
    --subset "$SUBSET" \
    --resolution "$RES" \
    --file_type images+poses \
    "${@:2}"

log "Done. DL3DV_DIR=$DEST_DIR"
