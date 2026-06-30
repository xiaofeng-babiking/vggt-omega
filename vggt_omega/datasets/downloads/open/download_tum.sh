#!/usr/bin/env bash
# TUM RGB-D -- Computer Vision Group, TU Munich. Open direct download (CC BY 4.0).
# Homepage: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
#
# Per-sequence .tgz archives live at  <BASE>/freiburg{1,2,3}/<seq>.tgz . Each
# unpacks to a <seq>/ dir holding rgb.txt, depth.txt, groundtruth.txt + rgb/ and
# depth/ -- exactly the layout TumDataset globs (TUM_DIR/<seq>/...), so we just
# extract every tarball straight into TUM_DIR.
#
# Sequence selection (first match wins):
#   1. explicit names as args after the dest dir:
#        ./download_tum.sh /data/tum rgbd_dataset_freiburg1_desk rgbd_dataset_freiburg2_xyz
#   2. TUM_SEQUENCES="seqA seqB ..."   (space-separated env override)
#   3. TUM_ALL=1                       (the full benchmark, ~50+ sequences, big)
#   4. default: the freiburg3 sitting/walking "dynamic objects" set this repo
#      evaluates on (8 seqs, ~5 GB) -- includes every sequence referenced by
#      datasets/config/tum.yaml and the TUM tests.
#
# Names must be the canonical on-disk dir names (rgbd_dataset_freiburgN_...); the
# freiburgN group is parsed from the name. Re-runs skip sequences already
# extracted (set TUM_FORCE=1 to re-fetch). Archives are removed after a
# successful extract unless TUM_KEEP_TGZ=1.
DATASET_NAME="tum"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "tum")"
[ "$#" -gt 0 ] && shift   # drop the dest arg; anything left is sequence names
log "Target: $DEST_DIR"

BASE="https://cvg.cit.tum.de/rgbd/dataset"

# The freiburg3 dynamic-objects set used for eval here (sitting + walking, all
# four camera motions). Superset of config/tum.yaml's referenced sequences.
DEFAULT_SEQUENCES=(
    rgbd_dataset_freiburg3_sitting_static
    rgbd_dataset_freiburg3_sitting_xyz
    rgbd_dataset_freiburg3_sitting_halfsphere
    rgbd_dataset_freiburg3_sitting_rpy
    rgbd_dataset_freiburg3_walking_static
    rgbd_dataset_freiburg3_walking_xyz
    rgbd_dataset_freiburg3_walking_halfsphere
    rgbd_dataset_freiburg3_walking_rpy
)

# Full TUM RGB-D benchmark (TUM_ALL=1). Best-effort: a failed name is warned and
# skipped, never silently written as a broken file.
ALL_SEQUENCES=(
    # Testing & Debugging
    rgbd_dataset_freiburg1_xyz rgbd_dataset_freiburg1_rpy
    rgbd_dataset_freiburg2_xyz rgbd_dataset_freiburg2_rpy
    # Handheld SLAM
    rgbd_dataset_freiburg1_360 rgbd_dataset_freiburg1_floor
    rgbd_dataset_freiburg1_desk rgbd_dataset_freiburg1_desk2 rgbd_dataset_freiburg1_room
    rgbd_dataset_freiburg2_360_hemisphere rgbd_dataset_freiburg2_360_kidnap
    rgbd_dataset_freiburg2_desk rgbd_dataset_freiburg2_large_no_loop
    rgbd_dataset_freiburg2_large_with_loop rgbd_dataset_freiburg3_long_office_household
    # Robot SLAM
    rgbd_dataset_freiburg2_pioneer_360 rgbd_dataset_freiburg2_pioneer_slam
    rgbd_dataset_freiburg2_pioneer_slam2 rgbd_dataset_freiburg2_pioneer_slam3
    # Structure vs. Texture
    rgbd_dataset_freiburg3_nostructure_notexture_far
    rgbd_dataset_freiburg3_nostructure_notexture_near_withloop
    rgbd_dataset_freiburg3_nostructure_texture_far
    rgbd_dataset_freiburg3_nostructure_texture_near_withloop
    rgbd_dataset_freiburg3_structure_notexture_far
    rgbd_dataset_freiburg3_structure_notexture_near
    rgbd_dataset_freiburg3_structure_texture_far
    rgbd_dataset_freiburg3_structure_texture_near
    # Dynamic Objects
    rgbd_dataset_freiburg2_desk_with_person
    rgbd_dataset_freiburg3_sitting_static rgbd_dataset_freiburg3_sitting_xyz
    rgbd_dataset_freiburg3_sitting_halfsphere rgbd_dataset_freiburg3_sitting_rpy
    rgbd_dataset_freiburg3_walking_static rgbd_dataset_freiburg3_walking_xyz
    rgbd_dataset_freiburg3_walking_halfsphere rgbd_dataset_freiburg3_walking_rpy
    # 3D Object Reconstruction
    rgbd_dataset_freiburg1_plant rgbd_dataset_freiburg1_teddy
    rgbd_dataset_freiburg2_coke rgbd_dataset_freiburg2_dishes
    rgbd_dataset_freiburg2_flowerbouquet rgbd_dataset_freiburg2_flowerbouquet_brownbackground
    rgbd_dataset_freiburg2_metallic_sphere rgbd_dataset_freiburg2_metallic_sphere2
    rgbd_dataset_freiburg3_cabinet rgbd_dataset_freiburg3_large_cabinet
    rgbd_dataset_freiburg3_teddy
)

# Resolve the sequence list per the precedence documented in the header.
if [ "$#" -gt 0 ]; then
    SEQUENCES=("$@")
elif [ -n "${TUM_SEQUENCES:-}" ]; then
    # shellcheck disable=SC2206  # intentional word-split of the env override
    SEQUENCES=(${TUM_SEQUENCES})
elif [ "${TUM_ALL:-0}" = "1" ]; then
    SEQUENCES=("${ALL_SEQUENCES[@]}")
else
    SEQUENCES=("${DEFAULT_SEQUENCES[@]}")
    log "No sequences given; using the default freiburg3 dynamic set (${#SEQUENCES[@]} seqs)."
    log "  -> TUM_ALL=1 for the whole benchmark, or pass names / TUM_SEQUENCES=..."
fi

ok=0; skipped=0; failed=0
for seq in "${SEQUENCES[@]}"; do
    fb="$(printf '%s' "$seq" | sed -nE 's/.*(freiburg[123]).*/\1/p')"
    if [ -z "$fb" ]; then
        warn "cannot derive freiburg{1,2,3} group from '$seq' -- skipping (use canonical rgbd_dataset_freiburgN_* names)"
        failed=$((failed + 1)); continue
    fi

    if [ -d "$DEST_DIR/$seq" ] && [ "${TUM_FORCE:-0}" != "1" ]; then
        log "already present: $seq/ (TUM_FORCE=1 to re-fetch) -- skipping"
        skipped=$((skipped + 1)); continue
    fi

    tgz="$DEST_DIR/$seq.tgz"
    log "Fetching $fb/$seq.tgz"
    if ! fetch "$BASE/$fb/$seq.tgz" "$tgz"; then
        warn "download failed for $seq -- skipping"
        rm -f "$tgz"; failed=$((failed + 1)); continue
    fi
    log "Extracting $seq"
    if ! tar -xzf "$tgz" -C "$DEST_DIR"; then
        warn "extract failed for $seq (corrupt/partial archive?) -- skipping"
        failed=$((failed + 1)); continue
    fi
    [ "${TUM_KEEP_TGZ:-0}" = "1" ] || rm -f "$tgz"
    ok=$((ok + 1))
done

log "Done. ok=$ok skipped=$skipped failed=$failed"
log "TUM_DIR=$DEST_DIR  (point datasets/config/tum.yaml's TUM_DIR here)"
[ "$failed" -eq 0 ] || warn "$failed sequence(s) failed -- see warnings above."
