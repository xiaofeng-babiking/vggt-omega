#!/usr/bin/env bash
# BEDLAM (CVPR 2023) -- synthetic humans, MPI. Registration + license required.
# Homepage: https://bedlam.is.tue.mpg.de/
# Storage: ~1.0 TB full (RGB 6fps + masks/depth/normals; ~0.5 TB images-only).
DATASET_NAME="bedlam"
source "$(dirname "$0")/common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "bedlam")"

manual_gate "https://bedlam.is.tue.mpg.de/download.php" <<EOF
BEDLAM is registration + license gated (links are session/cookie-bound):
1. Register a (free) MPI account on https://bedlam.is.tue.mpg.de/ and log in.
2. Accept the license: https://bedlam.is.tuebingen.mpg.de/license.html
3. From the Download page, copy the per-asset .tar/.zip links shown while logged in.
4. Download into: $DEST_DIR
   (the links require your session cookie; a typical recipe is to log in via
    'wget --post-data' then fetch each listed archive, or use a browser).
Render/training code: https://github.com/pixelite1201/BEDLAM   Contact: bedlam@tue.mpg.de
EOF
