#!/usr/bin/env bash
# WildRGB-D. Public HuggingFace repo (ungated). ~3.37 TB zipped.
# Repo: https://github.com/wildrgbd/wildrgbd  HF: hongchi/wildrgbd
#
# NOTE: the upstream download.py shells out to `wget https://huggingface.co/...`
# with a HARD-CODED host (ignores HF_ENDPOINT) and has no working --output_dir,
# so it neither honors the China mirror nor writes to the target dir. We instead
# pull the zip parts via huggingface-cli (which DOES honor HF_ENDPOINT, default
# hf-mirror.com) straight into DEST_DIR, then reassemble + unzip each category.
#
# Category selection: CAT=all (default) or a single category name, e.g. CAT=cup.
DATASET_NAME="wildrgbd"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "wildrgbd")"
log "Target: $DEST_DIR"
log "HF endpoint: ${HF_ENDPOINT}"

require_cmd unzip "Install with: sudo apt-get install -y unzip"
# `zip` is only needed to reassemble multi-part (.z01/.z02/...) categories; we
# check for it lazily, per-category, so single-zip categories work without it.
uv_sync   # isolated uv env: envs/wildrgbd (requests, tqdm, huggingface_hub)

# Per-category zip parts (mirrors upstream download.py's `categories` map).
declare -A CATEGORIES=(
    [bottle]="bottle.z01 bottle.z02 bottle.zip"
    [cup]="cup.z01 cup.z02 cup.z03 cup.zip"
    [tooth_brush]="tooth_brush.z01 tooth_brush.z02 tooth_brush.z03 tooth_brush.zip"
    [book]="book.z01 book.zip"
    [box]="box.z01 box.zip"
    [knife]="knife.z01 knife.z02 knife.zip"
    [remote_control]="remote_control.z01 remote_control.zip"
    [razor]="razor.z01 razor.zip"
    [keyboard]="keyboard.z01 keyboard.zip"
    [bowl]="bowl.z01 bowl.z02 bowl.zip"
    [scissor]="scissor.z01 scissor.z02 scissor.z03 scissor.zip"
    [kettle]="kettle.z01 kettle.z02 kettle.zip"
    [bucket]="bucket.zip"
    [pliers]="pliers.z01 pliers.z02 pliers.zip"
    [ball]="ball.z01 ball.zip"
    [mouse]="mouse.z01 mouse.zip"
    [handbag]="handbag.z01 handbag.z02 handbag.zip"
    [cellphone]="cellphone.zip"
    [microwave]="microwave.z01 microwave.zip"
    [hat]="hat.z01 hat.z02 hat.zip"
    [clock]="clock.z01 clock.zip"
    [shoe]="shoe.z01 shoe.z02 shoe.z03 shoe.zip"
    [flower_pot]="flower_pot.z01 flower_pot.z02 flower_pot.zip"
    [detergent]="detergent.z01 detergent.z02 detergent.zip"
    [backpack]="backpack.z01 backpack.z02 backpack.zip"
    [chair]="chair.z01 chair.z02 chair.zip"
    [TV]="TV.zip"
    [pineapple]="pineapple.z01 pineapple.zip"
    [potato]="potato.zip"
    [cucumber]="cucumber.zip"
    [apple]="apple.zip"
    [banana]="banana.zip"
    [pear]="pear.zip"
    [tomato]="tomato.zip"
    [peach]="peach.zip"
    [orange]="orange.z01 orange.zip"
    [carrot]="carrot.zip"
    [donut]="donut.zip"
    [cake]="cake.zip"
    [stuffed_toy]="stuffed_toy.z01 stuffed_toy.z02 stuffed_toy.zip"
    [train]="train.zip"
    [truck]="truck.zip"
    [boat]="boat.zip"
    [bus]="bus.zip"
    [plane]="plane.zip"
    [car]="car.zip"
)

REPO="hongchi/wildrgbd"   # HF model repo (resolve/main/<file>)
CAT="${CAT:-all}"

# Resolve the category list.
if [ "$CAT" = "all" ]; then
    CATS=($(printf '%s\n' "${!CATEGORIES[@]}" | sort))
else
    [ -n "${CATEGORIES[$CAT]:-}" ] || die "unknown category '$CAT'. Valid: $(printf '%s ' $(printf '%s\n' "${!CATEGORIES[@]}" | sort))"
    CATS=("$CAT")
fi

ok=0; skipped=0; failed=0
for cat in "${CATS[@]}"; do
    # Skip if already extracted (a <cat>/ dir exists) unless WILDRGBD_FORCE=1.
    if [ -d "$DEST_DIR/$cat" ] && [ "${WILDRGBD_FORCE:-0}" != "1" ]; then
        log "already present: $cat/ (WILDRGBD_FORCE=1 to re-fetch) -- skipping"
        skipped=$((skipped + 1)); continue
    fi

    parts=(${CATEGORIES[$cat]})
    if [ "${#parts[@]}" -gt 1 ] && ! command -v zip >/dev/null 2>&1; then
        warn "skipping $cat: multi-part archive needs 'zip' (sudo apt-get install -y zip)"
        failed=$((failed + 1)); continue
    fi
    log "Fetching $cat (${#parts[@]} part(s)) via HF mirror"
    fetch_failed=0
    for f in "${parts[@]}"; do
        # fetch() (wget --continue via HF_ENDPOINT) instead of `hf download`:
        # some files in this repo are Xet-backed and the hf CLI cannot fetch
        # those through hf-mirror ("Local entry not found"), while the plain
        # resolve/main redirect chain works and is resumable.
        #
        # Retry loop: multi-GB transfers on this route drop every few GB and
        # a single fetch() (wget --tries=5) often isn't enough for the 40 GB+
        # parts. Every retry resumes from the partial, so attempts converge.
        got=0
        for ((try = 1; try <= ${WILDRGBD_FETCH_RETRIES:-20}; try++)); do
            if fetch "https://huggingface.co/$REPO/resolve/main/$f" "$DEST_DIR/$f"; then
                got=1; break
            fi
            warn "fetch attempt $try/${WILDRGBD_FETCH_RETRIES:-20} failed for $f -- retrying (resumes partial)"
            sleep 10
        done
        if [ "$got" -ne 1 ]; then
            warn "download failed for $cat/$f"
            fetch_failed=1; break
        fi
    done
    if [ "$fetch_failed" -ne 0 ]; then
        warn "skipping $cat (incomplete download)"
        failed=$((failed + 1)); continue
    fi

    log "Extracting $cat"
    extract_ok=0
    (
        cd "$DEST_DIR" || exit 1
        if [ "${#parts[@]}" -gt 1 ]; then
            # Multi-part split zip: reassemble then unzip.
            zip -q -F "$cat.zip" --out "$cat-single.zip" \
                && unzip -q -o "$cat-single.zip" \
                && rm -f "$cat-single.zip" \
                && { [ "${WILDRGBD_KEEP_ZIP:-0}" = "1" ] || rm -f "${parts[@]}"; }
        else
            unzip -q -o "$cat.zip" \
                && { [ "${WILDRGBD_KEEP_ZIP:-0}" = "1" ] || rm -f "$cat.zip"; }
        fi
    ) && extract_ok=1

    if [ "$extract_ok" -ne 1 ]; then
        warn "extract failed for $cat (corrupt/partial archive?) -- skipping"
        failed=$((failed + 1)); continue
    fi
    ok=$((ok + 1))
done

log "Done. ok=$ok skipped=$skipped failed=$failed"
log "WILDRGBD_DIR=$DEST_DIR"
[ "$failed" -eq 0 ] || warn "$failed categor(y/ies) failed -- see warnings above."
