#!/usr/bin/env bash
# EFM3D (Egocentric Foundation Model 3D benchmark) -- Meta. Email-gated (Project Aria).
# Homepage: https://www.projectaria.com/research/efm3D/
# Storage: ~200 GB (ASE train/eval subset + Aria Everyday Objects + eval meshes);
#          the seq136 sample clip is ~1-2 GB.
DATASET_NAME="efm3d"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "efm3d")"
CDN_JSON="${CDN_JSON:-$DEST_DIR/efm3d_download_urls.json}"

if [ ! -f "$CDN_JSON" ]; then
    manual_gate "https://www.projectaria.com/research/efm3D/#download-dataset" <<EOF
EFM3D uses the Project Aria email-gated download-URLs JSON pattern:
1. Open the EFM3D page, submit your email / accept the license.
2. Save the served download-urls JSON as:
   $CDN_JSON   (or set CDN_JSON=<path>).
3. Re-run this script. (Repo: https://github.com/facebookresearch/efm3d)
EOF
fi

require_cmd git
uv_sync   # isolated uv env: envs/efm3d (projectaria-tools[all])
REPO_DIR="$DEST_DIR/_efm3d_repo"
[ -d "$REPO_DIR/.git" ] || git clone https://github.com/facebookresearch/efm3d "$REPO_DIR"

cat >&2 <<EOF
The EFM3D repo drives downloads from its data/ and ckpt/ READMEs using the JSON
manifest above. Follow $REPO_DIR/data/README.md, e.g. place the manifest and run
the repo's data scripts inside this dataset's uv env, e.g.:
    uv run --project envs/efm3d -- sh $REPO_DIR/prepare_inference.sh
Manifest in use: $CDN_JSON
EOF

log "Repo cloned at $REPO_DIR; complete the download per its data/README.md. EFM3D_DIR=$DEST_DIR"
