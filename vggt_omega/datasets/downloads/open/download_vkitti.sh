#!/usr/bin/env bash
# Virtual KITTI v2 (and v1) -- NAVER LABS Europe. Open direct download (CC BY-NC-SA 3.0).
# Homepage: https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/
DATASET_NAME="vkitti"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "vkitti")"
log "Target: $DEST_DIR"

V2="https://download.europe.naverlabs.com/virtual_kitti_2.0.3"
log "Fetching Virtual KITTI 2.0.3 archives"
# The NAVER server throttles per connection (~150 KB/s observed) and going
# through the local proxy is slower still (flow archives are up to ~29 GB, so
# a single stream can take a day). 8-way segmented aria2c, connecting
# directly, measured ~18x faster. aria2c -c resumes wget partials; once
# aria2c has touched a file, only aria2c may finish it (it tracks sparse
# segments in a .aria2 control file), which holds because this branch is
# always taken when aria2c is installed. Plain fetch_retry is the fallback.
for kind in rgb depth classSegmentation instanceSegmentation textgt \
            forwardFlow backwardFlow forwardSceneFlow backwardSceneFlow; do
    case "$kind" in
        textgt) ext="tar.gz" ;;      # upstream ships textgt as .tar.gz; everything else .tar
        *)      ext="tar" ;;
    esac
    url="$V2/vkitti_2.0.3_${kind}.${ext}"
    out="$DEST_DIR/vkitti_2.0.3_${kind}.${ext}"
    if command -v aria2c >/dev/null 2>&1; then
        got=0
        for ((i = 1; i <= ${FETCH_RETRIES:-20}; i++)); do
            if aria2c -c -x 8 -s 8 --all-proxy="" --file-allocation=none \
                --console-log-level=warn --retry-wait=10 --max-tries=5 \
                -d "$DEST_DIR" -o "$(basename "$out")" "$url"; then
                got=1; break
            fi
            warn "aria2c attempt $i/${FETCH_RETRIES:-20} failed for $url -- retrying (resumes)"
            sleep 10
        done
        [ "$got" -eq 1 ] || die "download failed after ${FETCH_RETRIES:-20} aria2c attempts: $url"
    else
        # subshell: fetch_retry is a function, so env -u can't wrap it
        (unset http_proxy https_proxy; fetch_retry "$url" "$out")
    fi
done

log "Extracting"
for f in "$DEST_DIR"/vkitti_2.0.3_*.tar*; do
    case "$f" in
        *.tar.gz) tar -xzf "$f" -C "$DEST_DIR" ;;
        *.tar)    tar -xf  "$f" -C "$DEST_DIR" ;;
    esac
done

log "Done. VKITTI_DIR=$DEST_DIR"
log "For v1.3.1 see: https://download.europe.naverlabs.com/virtual-kitti-1.3.1/"
