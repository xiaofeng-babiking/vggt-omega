# Shared helpers for the Project Aria dataset scripts in this folder.
#
# Every download_<slug>.sh here sets DATASET_NAME / SLUG / MANIFEST_NAME /
# DEFAULT_GROUPS, sources this file and calls `aria_dataset_main "$@"`.
#
# Access: the Aria Dataset Explorer (https://explorer.projectaria.com/) exposes
# an unauthenticated JSON API. GET /data/<slug>/download_links returns the very
# same `<Dataset>_download_urls.json` the explorer's email dialog would hand
# you (signed fbcdn URLs + sha1 + size per file, ~30-day expiry) -- so these
# datasets need no email/license click-through to fetch; the license terms
# linked in each script header still apply to the data.
#
# Transport: NO PROXY, by decision (2026-08-26). ARIA_TRANSPORT=direct (default)
# probes a no-proxy ranged GET on the CDN before every pass; if the CDN is not
# reachable directly the pass waits ARIA_WAIT_DIRECT seconds and re-probes --
# it never touches $https_proxy. (From this network the fbcdn CDN currently
# stalls at the TLS handshake without the proxy, so lanes sit in that wait
# loop until the route opens; the explorer API itself answers directly and
# the manifest is fetched with --no-proxy too.) ARIA_TRANSPORT=auto|proxy exist
# for other networks / explicit opt-in only.
#
# Knobs (env):
#   DATA_GROUPS   comma list or 'all' (default: the script's DEFAULT_GROUPS)
#   EXCLUDE_GROUPS comma list to drop from the selection
#   SEQUENCES     comma list of sequence uids, '@file', or 'all' (default all)
#   WORKERS=4 ARIA2_CONNS=8 ARIA2_SPEED_FLOOR=100K ARIA_MAX_RATE=  (aria2c tuning)
#   ARIA_TRANSPORT=direct  (auto|proxy opt-in)   ARIA_WAIT_DIRECT=300  wait between direct probes
#   ARIA_PASSES=20 ARIA_PASS_SLEEP=60    in-script retry loop (resumes each pass)
#   ARIA_REFRESH_MANIFEST=1              force re-fetching the manifest
#   STATUS_ONLY=1 | DRY_RUN=1 | VERIFY=1 | MAX_FILES=n | SMALL_FIRST=1
#
# Layout: $DEST/<sequence_uid>/<files> ; manifest at $DEST/$MANIFEST_NAME, which
# the official `aria_dataset_downloader --cdn_file` (envs/aria) also accepts if
# you later want its unzip-into-place behaviour.

# shellcheck shell=bash
ARIA_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../common.sh
source "$ARIA_SCRIPTS_DIR/../../common.sh"

ARIA_DRIVER="$ARIA_SCRIPTS_DIR/aria_explorer_dl.py"
EXPLORER_BASE="${EXPLORER_BASE:-https://explorer.projectaria.com}"
ARIA_PASSES="${ARIA_PASSES:-20}"
ARIA_PASS_SLEEP="${ARIA_PASS_SLEEP:-60}"
ARIA_MIN_TTL="${ARIA_MIN_TTL:-3600}"
ARIA_TRANSPORT="${ARIA_TRANSPORT:-direct}"
ARIA_WAIT_DIRECT="${ARIA_WAIT_DIRECT:-300}"

# Seconds until the earliest signed link in a manifest expires (0 if unreadable).
aria_manifest_ttl() {
    local manifest="$1" exp
    exp="$(python3 "$ARIA_DRIVER" --manifest "$manifest" --expiry 2>/dev/null | cut -d' ' -f1 || true)"
    [ -n "${exp:-}" ] || exp=0
    echo $(( exp - $(date +%s) ))
}

# Fetch $EXPLORER_BASE/data/$SLUG/download_links into $1 unless it is present
# and its links still have > ARIA_MIN_TTL seconds of life.
# The API answers directly; with the default direct transport wget is told
# --no-proxy so nothing here goes through $https_proxy.
aria_fetch_manifest() {
    local manifest="$1" url="$EXPLORER_BASE/data/$SLUG/download_links" tmp ttl
    if [ -f "$manifest" ] && [ "${ARIA_REFRESH_MANIFEST:-0}" != 1 ]; then
        ttl="$(aria_manifest_ttl "$manifest")"
        if [ "$ttl" -gt "$ARIA_MIN_TTL" ]; then
            log "manifest $manifest valid for $(( ttl / 86400 )) more days"
            return 0
        fi
        log "manifest $manifest expired/expiring -> re-fetching"
    fi
    require_cmd wget
    tmp="$manifest.tmp.$$"
    log "fetching manifest: $url"
    local wargs=(-q --tries=5 --timeout=60)
    [ "$ARIA_TRANSPORT" = direct ] && wargs+=(--no-proxy)
    wget "${wargs[@]}" -O "$tmp" "$url" || { rm -f "$tmp"; die "manifest fetch failed: $url"; }
    python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); assert ("sequences" in m or "releases" in m), "unexpected manifest shape"' "$tmp" \
        || { rm -f "$tmp"; die "manifest at $url is not a download_links JSON"; }
    mv -f "$tmp" "$manifest"
    ttl="$(aria_manifest_ttl "$manifest")"
    log "manifest saved: $manifest (links valid $(( ttl / 86400 )) days)"
    ARIA_REFRESH_MANIFEST=0
}

# Run the driver once with the env knobs. Returns its exit code (0/1/3).
aria_run_driver() {
    local dest="$1" manifest="$2"; shift 2
    local args=(--manifest "$manifest" --dest "$dest"
                --groups "${DATA_GROUPS:-${DEFAULT_GROUPS:-all}}"
                --sequences "${SEQUENCES:-all}"
                --transport "$ARIA_TRANSPORT"
                --min-ttl "$ARIA_MIN_TTL")
    [ -n "${EXCLUDE_GROUPS:-}" ] && args+=(--exclude-groups "$EXCLUDE_GROUPS")
    [ "${STATUS_ONLY:-0}" = 1 ] && args+=(--status)
    [ "${DRY_RUN:-0}" = 1 ] && args+=(--dry-run)
    [ "${VERIFY:-0}" = 1 ] && args+=(--verify)
    [ "${SMALL_FIRST:-0}" = 1 ] && args+=(--small-first)
    [ -n "${MAX_FILES:-}" ] && args+=(--max-files "$MAX_FILES")
    python3 "$ARIA_DRIVER" "${args[@]}" "$@"
}

# Standard body for every explorer-backed dataset script.
# $1 (optional) = destination dir; default $DATA_ROOT/aria/$SLUG.
aria_dataset_main() {
    : "${SLUG:?SLUG unset}" "${MANIFEST_NAME:?MANIFEST_NAME unset}"
    require_cmd python3
    require_cmd aria2c "apt install aria2 (segmented, resumable, sha1-verified transfers)"
    local dest manifest pass rc
    dest="$(resolve_dest "${1:-}" "aria/${DEST_SUBDIR:-$SLUG}")"
    manifest="$dest/$MANIFEST_NAME"
    log "dataset=$SLUG dest=$dest groups=${DATA_GROUPS:-${DEFAULT_GROUPS:-all}} sequences=${SEQUENCES:-all} transport=$ARIA_TRANSPORT"
    aria_fetch_manifest "$manifest"
    if [ "${STATUS_ONLY:-0}" = 1 ] || [ "${DRY_RUN:-0}" = 1 ]; then
        aria_run_driver "$dest" "$manifest"
        return $?
    fi
    for ((pass = 1; pass <= ARIA_PASSES; pass++)); do
        log "pass $pass/$ARIA_PASSES"
        set +e; aria_run_driver "$dest" "$manifest"; rc=$?; set -e
        case "$rc" in
            0) log "Done. ${SLUG^^}_DIR=$dest"; return 0 ;;
            3) ARIA_REFRESH_MANIFEST=1 aria_fetch_manifest "$manifest" ;;
            1) warn "pass $pass left files incomplete; sleeping ${ARIA_PASS_SLEEP}s then resuming"; sleep "$ARIA_PASS_SLEEP" ;;
            4) warn "CDN not reachable without a proxy; waiting ${ARIA_WAIT_DIRECT}s and re-probing (never using the proxy)"; sleep "$ARIA_WAIT_DIRECT" ;;
            *) die "driver failed with rc=$rc" ;;
        esac
    done
    err "still incomplete after $ARIA_PASSES passes -- re-run to resume (every pass is idempotent)"
    return 1
}
