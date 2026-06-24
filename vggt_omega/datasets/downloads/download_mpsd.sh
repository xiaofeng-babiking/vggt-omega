#!/usr/bin/env bash
# MPSD (Mapillary Planet-Scale Depth) -- Mapillary/Meta. Account + license via portal.
# Homepage: https://www.mapillary.com/dataset/depth
# Storage: multi-TB planet-scale depth (~several TB; distributed as zip volumes -- see portal).
DATASET_NAME="mpsd"
source "$(dirname "$0")/common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "mpsd")"

manual_gate "https://www.mapillary.com/dataset/depth" <<EOF
MPSD is distributed as zipped volumes via the Mapillary dataset portal (no anonymous wget):
1. Sign in / create a Mapillary account.
2. On the depth-dataset page, accept the Mapillary dataset license.
3. Download the zip volumes into $DEST_DIR, then unzip them.
(Confirmed via Meta MapAnything download docs:
 https://github.com/facebookresearch/map-anything)
EOF
