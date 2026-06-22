#!/usr/bin/env bash
# SAIL-VOS (UIUC, from GTA-V). Click-through license; provides download_sailvos.sh.
# Homepage: http://sailvos.web.illinois.edu/_site/index.html
# Storage: SAIL-VOS ~100 GB (RGB + amodal masks); SAIL-VOS 3D adds depth/mesh (~larger).
DATASET_NAME="sail_vos"
source "$(dirname "$0")/common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "sail_vos")"
SCRIPT="${SAILVOS_SCRIPT:-$DEST_DIR/download_sailvos.sh}"

if [ ! -f "$SCRIPT" ]; then
    manual_gate "http://sailvos.web.illinois.edu/_site/index.html" <<EOF
SAIL-VOS is click-through license gated (non-commercial; you must own GTA-V):
1. Open the project page, go to Download, and click "I agree" to the license.
2. Download the provided 'download_sailvos.sh' (assests/download_sailvos.sh).
3. Save it as: $SCRIPT   (or set SAILVOS_SCRIPT=<path>).
4. Re-run this script.
(SAIL-VOS 3D is linked from the same site.)
EOF
fi

require_cmd bash
log "Running the provided download_sailvos.sh into $DEST_DIR"
( cd "$DEST_DIR" && bash "$SCRIPT" )

log "Done. SAILVOS_DIR=$DEST_DIR"
