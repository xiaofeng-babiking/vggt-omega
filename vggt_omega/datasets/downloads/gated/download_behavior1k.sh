#!/usr/bin/env bash
# BEHAVIOR-1K (Stanford) -- OmniGibson assets via the repo setup script (EULA).
# Homepage: https://behavior.stanford.edu/
# Storage: OmniGibson 3D asset bundle ~25 GB (scenes+objects); +Isaac Sim install separate.
DATASET_NAME="behavior1k"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "behavior1k")"
log "Target: $DEST_DIR"

require_cmd git

REPO_DIR="$DEST_DIR/BEHAVIOR-1K"
[ -d "$REPO_DIR/.git" ] || git clone https://github.com/StanfordVL/BEHAVIOR-1K "$REPO_DIR"

cat >&2 <<EOF
BEHAVIOR-1K's OmniGibson 3D asset bundle is distributed under a license/EULA that
the setup step prompts you to accept before fetching the (encrypted) assets.
Running the official modular installer now:
  $REPO_DIR/setup.sh --omnigibson --bddl --dataset
Add/remove components (e.g. --teleop) as needed; follow the EULA prompt.
EOF

require_cmd bash
( cd "$REPO_DIR" && ./setup.sh --omnigibson --bddl --dataset "${@:2}" )

log "Done. BEHAVIOR1K_DIR=$REPO_DIR (assets under OmniGibson's data path)."
