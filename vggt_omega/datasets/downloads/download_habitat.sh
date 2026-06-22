#!/usr/bin/env bash
# Habitat-Matterport 3D (HM3D) -- Meta/FAIR + Matterport. Token-gated (academic EULA).
# Homepage: https://aihabitat.org/datasets/hm3d/
# Storage: HM3D v0.2 ~140 GB (train habitat 27 GB + glb 32 GB, val 3.3 GB, semantics);
#          hm3d_full (all uids) ~200 GB.
DATASET_NAME="habitat"
source "$(dirname "$0")/common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "habitat")"

if [ -z "${MATTERPORT_TOKEN_ID:-}" ] || [ -z "${MATTERPORT_TOKEN_SECRET:-}" ]; then
    manual_gate "https://aihabitat.org/datasets/hm3d/" <<EOF
HM3D requires a Matterport API token (academic/non-commercial EULA):
1. Create a free Matterport account: https://buy.matterport.com/free-account-register
2. Account Settings -> Developer Tools -> request access to
   "Habitat - Matterport 3D Research Dataset" and complete the form.
3. Copy your API Token ID/Secret and export:
   export MATTERPORT_TOKEN_ID=...    MATTERPORT_TOKEN_SECRET=...
4. Install habitat-sim (conda only -- NOT on PyPI, so this dataset cannot use a
   uv env): conda install habitat-sim -c conda-forge -c aihabitat
   Then point HABITAT_PYTHON at that interpreter (default: python3 on PATH).
EULA: https://matterport.com/matterport-end-user-license-agreement-academic-use-model-data
EOF
fi

# habitat-sim is distributed via conda, not PyPI, so we use the interpreter that
# has it (set HABITAT_PYTHON to your conda env's python), not a uv-managed venv.
HABITAT_PYTHON="${HABITAT_PYTHON:-python3}"
require_cmd "$HABITAT_PYTHON"
"$HABITAT_PYTHON" -c "import habitat_sim" 2>/dev/null \
    || die "habitat-sim not importable via '$HABITAT_PYTHON'; install it (conda install habitat-sim -c conda-forge -c aihabitat) and set HABITAT_PYTHON."

log "Downloading HM3D via habitat_sim datasets_download"
"$HABITAT_PYTHON" -m habitat_sim.utils.datasets_download \
    --username "$MATTERPORT_TOKEN_ID" \
    --password "$MATTERPORT_TOKEN_SECRET" \
    --uids hm3d_full \
    --data-path "$DEST_DIR"

log "Done. HABITAT_DIR=$DEST_DIR"
