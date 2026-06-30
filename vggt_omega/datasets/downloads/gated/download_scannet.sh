#!/usr/bin/env bash
# ScanNet series -- ScanNet (v1/v2) + ScanNet++ . Both require signed Terms of Use.
# ScanNet: http://www.scan-net.org/   ScanNet++: https://kaldir.vc.in.tum.de/scannetpp/
# Storage: ScanNet v2 ~1.3 TB full (~3.7 GB/scan x ~1500 scans; .sens RGB-D dominates);
#          ScanNet++ ~2-3 TB (iPhone+DSLR RGB + high-res laser scans).
DATASET_NAME="scannet"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "scannet")"
SCANNET_SCRIPT="${SCANNET_SCRIPT:-$DEST_DIR/download-scannet.py}"

if [ ! -f "$SCANNET_SCRIPT" ]; then
    manual_gate "http://www.scan-net.org/" <<EOF
The ScanNet series requires signed Terms of Use (per-dataset approval):

ScanNet (v1/v2):
1. Fill the ScanNet Terms of Use PDF with an INSTITUTIONAL email:
   http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf
2. Email it to scannet@googlegroups.com . On approval you receive 'download-scannet.py'.
3. Save it as: $SCANNET_SCRIPT   (or set SCANNET_SCRIPT=<path>), then re-run.

ScanNet++ (v1/v2):
1. Register + apply: https://kaldir.vc.in.tum.de/scannetpp/register
   (accept https://kaldir.vc.in.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf)
2. On approval, use the personalized token + download script from your dashboard.
   Toolkit: https://github.com/scannetpp/scannetpp
EOF
fi

uv_sync   # isolated uv env: envs/scannet (requests)
log "Running ScanNet downloader into $DEST_DIR (pass extra flags after the dest, e.g. --label_map)"
uv_py "$SCANNET_SCRIPT" -o "$DEST_DIR" "${@:2}"

log "Done. SCANNET_DIR=$DEST_DIR"
