#!/usr/bin/env bash
# Launcher for the explorer-backed Project Aria datasets (this folder).
# Every lane runs detached (setsid+nohup, own process group) with a pidfile and
# a log under $DOWNLOAD_ROOT/aria/.launch/, and re-runs its (idempotent,
# resumable) dataset scripts for up to MAX_PASSES passes with PASS_SLEEP
# between passes, so a spent retry budget becomes "try again later".
# Lanes are size-ordered so small datasets finish first; each lane is sequential:
#   small   : aeo aria_scenes gen2pilot dtc_objects dtc aea hot3d_quest hot3d_aria  (~1.6 TB)
#   adt     : adt                                                                    (~2.2 TB)
#   ritw    : ritw                                                                   (~6.6 TB)
#   nymeria : nymeria_plus                                                           (~38 TB)
# Not in any lane: nymeria (v0.0, superseded by nymeria_plus), ase / efm3d / hot3d
# assets (email-gated, see their scripts).
# Transport: NO PROXY (ARIA_TRANSPORT=direct, see aria_common.sh). Each pass
# probes direct CDN access; while it is unavailable a lane just waits and
# re-probes, so lanes can stay up until the route opens. MAX_PASSES=0 (default)
# therefore means unlimited; `stop` ends a lane.
#
# Usage: launch_all.sh start  [lane ...]    # default: all lanes
#        launch_all.sh stop   [lane ...]
#        launch_all.sh status [--brief]     # per-dataset completion (files / bytes)
#        launch_all.sh logs   [lane ...]    # tail each lane log
# Tunables (env): MAX_PASSES=0(unlimited) PASS_SLEEP=600 WORKERS=2 ARIA2_CONNS=8
#                 ARIA2_SPEED_FLOOR=100K ARIA_TRANSPORT=direct ARIA_WAIT_DIRECT=300 ARIA_MAX_RATE=
set -uo pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
HERE="$(dirname "$SELF")"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${DATA_ROOT:-/ipfs/babiking/datasets/3DR}}"
STATE="$DOWNLOAD_ROOT/aria/.launch"
ALL_LANES=(small adt ritw nymeria)
declare -A LANE_SETS=(
    [small]="aeo aria_scenes gen2pilot dtc_objects dtc aea hot3d_quest hot3d_aria"
    [adt]="adt"
    [ritw]="ritw"
    [nymeria]="nymeria_plus"
)
MAX_PASSES="${MAX_PASSES:-0}"   # 0 = unlimited
PASS_SLEEP="${PASS_SLEEP:-600}"
export WORKERS="${WORKERS:-2}" ARIA2_CONNS="${ARIA2_CONNS:-8}" ARIA2_SPEED_FLOOR="${ARIA2_SPEED_FLOOR:-100K}"
export ARIA_TRANSPORT="${ARIA_TRANSPORT:-direct}" ARIA_WAIT_DIRECT="${ARIA_WAIT_DIRECT:-300}"
export DOWNLOAD_ROOT

ts() { date -u +%FT%TZ; }
lane_known() { [[ -v "LANE_SETS[$1]" ]]; }
dest_of() {  # dataset script name -> its data dir
    local sub
    sub="$(sed -n 's/^DEST_SUBDIR="\(.*\)"/\1/p' "$HERE/download_$1.sh")"
    [ -n "$sub" ] || sub="$(sed -n 's/^SLUG="\(.*\)"/\1/p' "$HERE/download_$1.sh")"
    echo "$DOWNLOAD_ROOT/aria/$sub"
}

# --- lane body (runs inside the detached bash) ------------------------------
run_passes() {
    local label="$1"; shift
    local i
    for ((i = 1; MAX_PASSES == 0 || i <= MAX_PASSES; i++)); do
        printf '\n===== [%s] pass %d/%s  %s =====\n' "$label" "$i" "${MAX_PASSES/#0/inf}" "$(ts)"
        if "$@"; then
            printf '===== [%s] finished OK on pass %d  %s =====\n' "$label" "$i" "$(ts)"
            return 0
        fi
        printf '===== [%s] pass %d exited non-zero; sleeping %ss =====\n' "$label" "$i" "$PASS_SLEEP"
        sleep "$PASS_SLEEP"
    done
    printf '===== [%s] gave up after %d passes -- `%s start` later to resume =====\n' "$label" "$MAX_PASSES" "$SELF"
    return 1
}
run_lane() {
    local lane="$1" ds rc=0
    printf '===== lane %s: %s  (%s) =====\n' "$lane" "${LANE_SETS[$lane]}" "$(ts)"
    for ds in ${LANE_SETS[$lane]}; do
        run_passes "$ds" "$HERE/download_$ds.sh" || rc=1
    done
    printf '===== lane %s done rc=%d  (%s) =====\n' "$lane" "$rc" "$(ts)"
    return $rc
}

# --- commands ---------------------------------------------------------------
lane_pid() { cat "$STATE/$1.pid" 2>/dev/null; }
lane_alive() { local p; p="$(lane_pid "$1")"; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

cmd_start() {
    local lanes=("$@") lane
    [ ${#lanes[@]} -gt 0 ] || lanes=("${ALL_LANES[@]}")
    mkdir -p "$STATE"
    for lane in "${lanes[@]}"; do
        lane_known "$lane" || { echo "unknown lane: $lane (have: ${ALL_LANES[*]})" >&2; continue; }
        if lane_alive "$lane"; then echo "lane $lane already running (pid $(lane_pid "$lane"))"; continue; fi
        setsid nohup bash "$SELF" _lane "$lane" >>"$STATE/$lane.log" 2>&1 < /dev/null &
        echo $! >"$STATE/$lane.pid"
        echo "started lane $lane (pid $!) -> $STATE/$lane.log"
    done
}
cmd_stop() {
    local lanes=("$@") lane p
    [ ${#lanes[@]} -gt 0 ] || lanes=("${ALL_LANES[@]}")
    for lane in "${lanes[@]}"; do
        p="$(lane_pid "$lane")"
        if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
            kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null
            echo "stopped lane $lane (pgid $p)"
        else
            echo "lane $lane not running"
        fi
        rm -f "$STATE/$lane.pid"
    done
}
cmd_status() {
    local brief=0 lane ds dest manifest line
    [ "${1:-}" = "--brief" ] && brief=1
    for lane in "${ALL_LANES[@]}"; do
        if lane_alive "$lane"; then echo "lane $lane: RUNNING (pid $(lane_pid "$lane"))"; else echo "lane $lane: not running"; fi
        for ds in ${LANE_SETS[$lane]}; do
            dest="$(dest_of "$ds")"
            manifest="$(ls "$dest"/*_download_urls.json 2>/dev/null | head -1)"
            if [ -z "$manifest" ]; then printf '  %-14s not started\n' "$ds"; continue; fi
            if [ "$brief" = 1 ]; then
                line="$(STATUS_ONLY=1 "$HERE/download_$ds.sh" 2>/dev/null | grep -E '^\[aria-dl\]   TOTAL' | sed 's/^\[aria-dl\]   TOTAL *//')"
                printf '  %-14s %s\n' "$ds" "${line:-?}"
            else
                printf '  --- %s (%s)\n' "$ds" "$dest"
                STATUS_ONLY=1 "$HERE/download_$ds.sh" 2>/dev/null | grep -E '^\[aria-dl\]   ' | sed 's/^\[aria-dl\]/   /'
            fi
        done
    done
}
cmd_logs() {
    local lanes=("$@") lane
    [ ${#lanes[@]} -gt 0 ] || lanes=("${ALL_LANES[@]}")
    for lane in "${lanes[@]}"; do
        echo "===== $STATE/$lane.log ====="; tail -n "${TAIL:-15}" "$STATE/$lane.log" 2>/dev/null || echo "(no log)"
    done
}

case "${1:-}" in
    start)  shift; cmd_start "$@" ;;
    stop)   shift; cmd_stop "$@" ;;
    status) shift; cmd_status "$@" ;;
    logs)   shift; cmd_logs "$@" ;;
    _lane)  run_lane "$2" ;;
    *) sed -n '2,25p' "$SELF" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
