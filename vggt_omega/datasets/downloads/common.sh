# Shared helpers for the per-dataset download scripts in this directory.
#
# Source this from each download_<dataset>.sh:
#     source "$(dirname "$0")/common.sh"
#
# Convention shared by every script here:
#   * The first positional arg (or the $DEST env var) is the target directory.
#   * Default target is $DATA_ROOT/<dataset>, with DATA_ROOT defaulting to
#     /jfs/Data_4DFF/train_data (matching the *_DIR paths in datasets/config/*.yaml).
#   * Datasets that need a license / signup / gated-HF approval do NOT guess a
#     URL: they print the exact manual steps and exit non-zero so nothing
#     silently downloads a broken file.
#
# Python isolation (uv):
#   Each dataset whose downloader needs Python tooling has its own
#   declarative environment under  downloads/envs/<dataset>/pyproject.toml .
#   We never touch the system interpreter or the project's main .venv: every
#   Python call goes through `uv run --project envs/<dataset>`, which uv backs
#   with a uv-managed CPython (UV_PYTHON below) in envs/<dataset>/.venv and a
#   per-dataset uv.lock. Use uv_sync once, then uv_py to run.

# shellcheck shell=bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/jfs/Data_4DFF/train_data}"

# Directory holding this file (downloads/), so env paths resolve regardless of cwd.
DOWNLOADS_DIR="${DOWNLOADS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ENVS_DIR="$DOWNLOADS_DIR/envs"
# uv-managed interpreter every dataset env is built on (override if you must).
UV_PYTHON="${UV_PYTHON:-3.10}"

# ---- logging ---------------------------------------------------------------
log()  { printf '\033[1;34m[%s]\033[0m %s\n' "${DATASET_NAME:-download}" "$*"; }
warn() { printf '\033[1;33m[%s] WARN:\033[0m %s\n' "${DATASET_NAME:-download}" "$*" >&2; }
err()  { printf '\033[1;31m[%s] ERROR:\033[0m %s\n' "${DATASET_NAME:-download}" "$*" >&2; }
die()  { err "$*"; exit 1; }

# Resolve the destination dir from $1 or $DEST, defaulting to DATA_ROOT/<sub>.
# Usage: DEST_DIR="$(resolve_dest "$1" "co3d")"
resolve_dest() {
    local arg="${1:-}" sub="${2:-${DATASET_NAME:-data}}"
    local dest="${arg:-${DEST:-$DATA_ROOT/$sub}}"
    mkdir -p "$dest"
    printf '%s\n' "$dest"
}

# Require a command on PATH or die with an install hint.
require_cmd() {
    local cmd="$1" hint="${2:-}"
    command -v "$cmd" >/dev/null 2>&1 || die "'$cmd' not found on PATH. ${hint}"
}

# Resumable download, wget-first with a curl fallback. Args: <url> <out_path>
# We try wget, then fall back to curl if wget *fails* (not just if it's absent):
# some CDNs (e.g. download.microsoft.com) don't serve the full intermediate
# chain, and wget won't fetch the missing issuer via AIA the way curl/OpenSSL
# does -- so wget reports "unable to verify the issuer's authority" while curl
# verifies fine. Certificate verification stays ON in both paths.
fetch() {
    local url="$1" out="$2"
    if command -v wget >/dev/null 2>&1; then
        wget --continue --tries=5 --timeout=60 -O "$out" "$url" && return 0
        command -v curl >/dev/null 2>&1 && warn "wget failed for $url -- falling back to curl"
    fi
    if command -v curl >/dev/null 2>&1; then
        curl -L --retry 5 --continue-at - -o "$out" "$url" && return 0
        return 1
    fi
    command -v wget >/dev/null 2>&1 || die "neither wget nor curl is available"
    return 1
}

# Print a clearly-formatted "manual access required" block and exit non-zero.
# Usage: manual_gate "Homepage URL" <<'EOF'
#   step 1
#   step 2
# EOF
manual_gate() {
    local url="$1"
    err "This dataset requires MANUAL access (license / signup / gated repo)."
    err "It cannot be downloaded by an unattended script."
    printf '\n  Access / homepage: %s\n\n  Steps:\n' "$url" >&2
    sed 's/^/    /' >&2
    printf '\n' >&2
    exit 2
}

# ---- uv-managed per-dataset Python environments ----------------------------
# Path to a dataset's env directory (holds pyproject.toml, uv.lock, .venv).
# Defaults to envs/$DATASET_NAME. Usage: ENV_DIR="$(uv_env_dir)".
uv_env_dir() {
    printf '%s\n' "$ENVS_DIR/${1:-${DATASET_NAME:?DATASET_NAME unset}}"
}

# Materialize the dataset venv from its pyproject.toml (idempotent; resumable).
# Builds/uses a uv-managed CPython == $UV_PYTHON. Call once before uv_py.
# Usage: uv_sync          # uses envs/$DATASET_NAME
#        uv_sync co3d     # explicit dataset
uv_sync() {
    require_cmd uv "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
    local env_dir; env_dir="$(uv_env_dir "${1:-}")"
    [ -f "$env_dir/pyproject.toml" ] || die "no pyproject.toml at $env_dir (expected a per-dataset env)"
    log "Syncing uv env: $env_dir (python $UV_PYTHON)"
    uv sync --project "$env_dir" --python "$UV_PYTHON"
}

# Run a Python module/file inside the dataset venv. Everything after the optional
# leading dataset name is passed to `uv run`. Usage:
#   uv_py path/to/script.py --flag        # uses envs/$DATASET_NAME
#   uv_py --env co3d -m pip --version     # explicit dataset env
uv_py() {
    require_cmd uv "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
    local env_name="${DATASET_NAME:-}"
    if [ "${1:-}" = "--env" ]; then env_name="$2"; shift 2; fi
    local env_dir; env_dir="$(uv_env_dir "$env_name")"
    uv run --project "$env_dir" --python "$UV_PYTHON" python "$@"
}

# Run an arbitrary console script / entry point installed in the dataset venv
# (e.g. omnitools.download, huggingface-cli). Usage: uv_tool <cmd> [args...]
uv_tool() {
    require_cmd uv "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
    local env_dir; env_dir="$(uv_env_dir)"
    uv run --project "$env_dir" --python "$UV_PYTHON" -- "$@"
}

# `uv pip` into the dataset venv (for editable installs of cloned repos that ship
# their own setup.py/pyproject, e.g. uCO3D). NOTE: unlike `uv sync`/`uv run`,
# `uv pip` does NOT locate the env from `--project`; it resolves it from
# VIRTUAL_ENV (or an active venv / `--python <interpreter-path>`). So we point
# VIRTUAL_ENV at the env's .venv (created by uv_sync) and pass the pip subcommand
# verbatim. Usage: uv_pip install -e <repo>
uv_pip() {
    require_cmd uv "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
    local env_dir; env_dir="$(uv_env_dir)"
    VIRTUAL_ENV="$env_dir/.venv" uv pip "$@"
}

# huggingface-cli download wrapper for ungated public datasets, run inside the
# dataset's uv env (which must depend on huggingface_hub[cli]).
# Args: <repo_id> <dest> [extra args...]
hf_download() {
    local repo="$1" dest="$2"; shift 2
    uv_tool huggingface-cli download "$repo" --repo-type dataset --local-dir "$dest" "$@"
}
