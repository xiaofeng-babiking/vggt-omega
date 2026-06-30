#!/usr/bin/env bash
# Mapillary Metropolis -- Mapillary/Meta. Account + dataset license via the portal.
# Homepage: https://www.mapillary.com/dataset/metropolis
# Storage: ~500 GB (multi-city street-level imagery with 2D/3D annotations; exact
#          size per the portal -- distributed as zip volumes).
DATASET_NAME="mapillary_metropolis"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "mapillary_metropolis")"

manual_gate "https://www.mapillary.com/dataset/metropolis" <<EOF
Mapillary Metropolis is distributed via the Mapillary dataset portal (no anonymous wget):
1. Sign in / create a Mapillary account.
2. On the Metropolis dataset page, accept the Mapillary dataset/research license.
3. Use the portal's download links/buttons to fetch the archives into $DEST_DIR.
If the page is unavailable, contact Mapillary research directly.
(Referenced by Meta MapAnything: https://github.com/facebookresearch/map-anything)
EOF
