#!/usr/bin/env bash
# Hypersim -- Apple. Photorealistic synthetic indoor. Open (CC BY-SA 3.0). ~1.9 TB.
# Storage: ~1.9 TB downloads + similar decompressed (budget ~4 TB total; set
# HYPERSIM_DELETE_ZIP=1 to drop each zip right after its successful extract).
# Repo: https://github.com/apple/ml-hypersim
#
# Downloads the ~460 per-scene zips into downloads/ and extracts each into
# scenes/<scene>/. The URL list is read from the official repo's
# dataset_download_images.py; the official downloader itself is NOT used
# because it re-fetches every zip from scratch on every run. Re-runs here are
# incremental, per scene:
#   * scenes/<scene>/ already extracted          -> skip entirely
#   * downloads/<scene>.zip complete (listable)  -> skip download, just extract
#   * downloads/<scene>.zip partial/truncated    -> resume it (fetch = wget -c)
#   * neither present                            -> fresh download + extract
# Extraction goes through a temp dir + rename, so scenes/<scene>/ never exists
# half-written. Delete a scene dir and/or its zip to force a re-do.
DATASET_NAME="hypersim"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "hypersim")"
log "Target: $DEST_DIR"

require_cmd git
require_cmd unzip "Install it (e.g. 'apt-get install unzip')."

REPO_DIR="$DEST_DIR/_ml-hypersim"
[ -d "$REPO_DIR/.git" ] || git clone https://github.com/apple/ml-hypersim "$REPO_DIR"

ZIP_DIR="$DEST_DIR/downloads"
SCENES_DIR="$DEST_DIR/scenes"
mkdir -p "$ZIP_DIR" "$SCENES_DIR"

URL_LIST="$REPO_DIR/code/python/tools/dataset_download_images.py"
mapfile -t urls < <(grep -oE 'https://[^"]+\.zip' "$URL_LIST")
[ "${#urls[@]}" -gt 0 ] || die "no scene zip URLs found in $URL_LIST"

# unzip wrapper: rc 1 is "warnings but extraction completed"; >= 2 is real error.
unz() {
    local rc=0
    unzip -q "$@" || rc=$?
    [ "$rc" -le 1 ]
}

# A complete zip lists cleanly; a truncated download has no end-of-central-
# directory record and fails, which routes it to the resume path below.
zip_ok() { unzip -l "$1" >/dev/null 2>&1; }

total=${#urls[@]}; n=0; fetched=0; extracted=0; complete=0; failed=0
log "$total scene zips in the official list -- skipping whatever is already here"
for url in "${urls[@]}"; do
    n=$((n + 1))
    name="$(basename "$url")"   # ai_XXX_YYY.zip
    scene="${name%.zip}"
    zip_path="$ZIP_DIR/$name"

    if [ -d "$SCENES_DIR/$scene" ]; then
        complete=$((complete + 1)); continue
    fi

    if [ -f "$zip_path" ] && zip_ok "$zip_path"; then
        :   # zip already downloaded -- go straight to extract
    else
        if [ -f "$zip_path" ]; then
            log "[$n/$total] resuming partial $name"
        else
            log "[$n/$total] fetching $name"
        fi
        if ! fetch "$url" "$zip_path"; then
            warn "download failed: $name (partial kept; next run resumes it)"
            failed=$((failed + 1)); continue
        fi
        if ! zip_ok "$zip_path"; then
            warn "$name invalid after download -- deleting so next run re-fetches"
            rm -f "$zip_path"
            failed=$((failed + 1)); continue
        fi
        fetched=$((fetched + 1))
    fi

    log "[$n/$total] extracting $name"
    tmp_dir="$SCENES_DIR/.tmp.$scene"
    rm -rf "$tmp_dir"
    if unz "$zip_path" -d "$tmp_dir" && [ -d "$tmp_dir/$scene" ]; then
        mv "$tmp_dir/$scene" "$SCENES_DIR/$scene"
        rm -rf "$tmp_dir"
        extracted=$((extracted + 1))
        if [ "${HYPERSIM_DELETE_ZIP:-0}" = "1" ]; then rm -f "$zip_path"; fi
    else
        rm -rf "$tmp_dir"
        warn "extract failed: $name (corrupt zip? delete it to force a re-download)"
        failed=$((failed + 1)); continue
    fi
done

log "Done. already-complete=$complete fetched=$fetched extracted=$extracted failed=$failed (of $total scenes)"
log "HYPERSIM_DIR=$DEST_DIR/scenes"
[ "$failed" -eq 0 ] || die "$failed scene(s) failed -- re-run to retry them"
