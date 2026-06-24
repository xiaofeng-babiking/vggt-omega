#!/usr/bin/env bash
# Map-free Relocalization ("Mapfree") -- Niantic. Click-through license (academic).
# Homepage: https://research.nianticlabs.com/mapfree-reloc-benchmark/dataset
# Storage: ~163 GB total -- Training 110 GB, Test 15 GB, Validation 8 GB,
#          COLMAP models 30 GB, Sample 0.5 GB.
DATASET_NAME="mapfree"
source "$(dirname "$0")/common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "mapfree")"

manual_gate "https://research.nianticlabs.com/mapfree-reloc-benchmark/dataset" <<EOF
Map-free Relocalization is click-through license gated (non-commercial/academic):
1. Open the dataset page and tick the license-agreement checkbox; this activates
   the download buttons (links are served after agreeing).
2. Download the splits you need into $DEST_DIR :
   Sample (~0.5 GB), Training (~110 GB), Validation (~8 GB), Test (~15 GB),
   COLMAP models (~30 GB).  Test-set poses are withheld (use their submission service).
Code/eval: https://github.com/nianticlabs/map-free-reloc
EOF
