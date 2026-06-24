#!/usr/bin/env bash
# Aria Synthetic Environments (ASE) / Project Aria datasets -- Meta.
# Email-gated: you submit an email and receive a download-URLs JSON manifest.
# Storage: ASE full ~2.5 TB (~100k scenes, ~25 MB/scene); a 0-100 scene slice is ~2-3 GB.
DATASET_NAME="aria"
source "$(dirname "$0")/common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "aria")"
CDN_JSON="${CDN_JSON:-$DEST_DIR/aria_synthetic_environments_dataset_download_urls.json}"

if [ ! -f "$CDN_JSON" ]; then
    manual_gate "https://www.projectaria.com/datasets/ase/" <<EOF
Project Aria datasets are email-gated (license acceptance, instant links that expire):
1. Open the ASE dataset page and submit your email / accept the license.
2. It serves a CDN manifest JSON; save it as:
   $CDN_JSON   (or set CDN_JSON=<path>).
3. Re-run this script (its uv env provides projectaria-tools automatically).
4. (See also https://facebookresearch.github.io/projectaria_tools/ for ADT/AEA/AEO etc.)
EOF
fi

require_cmd git
uv_sync   # isolated uv env: envs/aria (projectaria-tools[all])

REPO_DIR="$DEST_DIR/_projectaria_tools"
[ -d "$REPO_DIR/.git" ] || git clone https://github.com/facebookresearch/projectaria_tools "$REPO_DIR"

log "Downloading ASE via the official downloader (edit --set/--scene-ids as needed)"
uv_py "$REPO_DIR/projects/AriaSyntheticEnvironment/aria_synthetic_environments_downloader.py" \
    --set train \
    --scene-ids "${SCENE_IDS:-0-100}" \
    --cdn-file "$CDN_JSON" \
    --output-dir "$DEST_DIR" \
    --unzip True

log "Done. ARIA_DIR=$DEST_DIR"
