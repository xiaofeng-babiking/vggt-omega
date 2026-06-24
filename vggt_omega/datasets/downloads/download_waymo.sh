#!/usr/bin/env bash
# Waymo Open Dataset -- gated (Google login + non-commercial license), GCS-hosted.
# Homepage: https://waymo.com/open/
DATASET_NAME="waymo"
source "$(dirname "$0")/common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "waymo")"

if ! command -v gsutil >/dev/null 2>&1; then
    manual_gate "https://waymo.com/open/" <<EOF
Waymo Open Dataset is gated (Google account + non-commercial license):
1. Sign in at https://waymo.com/open/ and accept the license:
   https://waymo.com/open/terms/
2. Install the gcloud SDK so 'gsutil' is available, and 'gcloud auth login'.
3. Re-run this script (or copy the gs:// bucket paths from your Download dashboard).
Parsing tools: pip install waymo-open-dataset-tf-2-*   Repo: https://github.com/waymo-research/waymo-open-dataset
EOF
fi

# Bucket path depends on the version/perception-vs-motion set shown on your dashboard.
BUCKET="${WAYMO_BUCKET:-gs://waymo_open_dataset_v_1_4_3/individual_files}"
log "Copying from $BUCKET (set WAYMO_BUCKET to the path from your dashboard)"
gsutil -m cp -r "$BUCKET" "$DEST_DIR"

log "Done. WAYMO_DIR=$DEST_DIR"
