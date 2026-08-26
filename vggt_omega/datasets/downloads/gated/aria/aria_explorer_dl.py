#!/usr/bin/env python3
"""Download driver for Project Aria explorer manifests (stdlib only).

Input is the JSON that https://explorer.projectaria.com/data/<slug>/download_links
(or https://dtc.projectaria.com/data/objects/download_links) serves -- the very
same ``<Dataset>_download_urls.json`` the explorer's email dialog would hand you
and that the official ``aria_dataset_downloader --cdn_file`` consumes:

    {"sequences": {<uid>: {<group>: {filename, sha1sum, file_size_bytes, download_url}}},
     "sequence_config": {...}}                          # sequence datasets
    {"releases": {<rel>: {"objects": {<uid>: {<group>: {...}}}}}}   # DTC object library

The driver selects data groups / sequences, works out which files are still
missing or partial on disk, and hands ONLY those to a single aria2c run
(segmented, resumable, sha-1 verified by aria2c itself via ``checksum=``).
Files that finished in this pass are re-hashed and quarantined on mismatch, so
a "complete" file on disk is always a verified one.

Layout: <dest>/<uid>/<filename>            (sequence datasets)
        <dest>/<release>/<uid>/<filename>  (DTC objects)
State:  <dest>/.aria_dl/{aria2.input,aria2c.log}

Transport: --transport direct (default) NEVER uses a proxy: it probes a
no-proxy ranged GET on the CDN first and exits 4 if that fails, so the caller
can wait and re-probe. 'auto' falls back to $https_proxy; 'proxy' forces it.

Exit codes: 0 every selected file is complete
            1 some files still missing/partial after this pass (re-run: resumes)
            3 manifest links expire within --min-ttl (re-fetch the manifest)
            4 direct CDN access unavailable and --transport direct (wait, re-probe)
            2 bad arguments / unreadable manifest
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

TAG = "[aria-dl]"
OE_RE = re.compile(r"[?&]oe=([0-9A-Fa-f]+)")


def log(msg):
    print(f"{TAG} {msg}", flush=True)


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(nbytes) < 1000 or unit == "PB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} B"
        nbytes /= 1000.0
    return f"{nbytes:.1f} PB"


# ---------------------------------------------------------------- manifest
def _rec(uid, group, subdir, f):
    return {
        "uid": uid,
        "group": group,
        "subdir": subdir,
        "filename": f["filename"],
        "size": int(f.get("file_size_bytes") or 0),
        "sha1": (f.get("sha1sum") or "").lower(),
        "url": f["download_url"],
    }


def load_manifest(path):
    """Return (items, config). Handles both the sequence and the DTC object shape."""
    with open(path) as fh:
        m = json.load(fh)
    items = []
    if "sequences" in m:
        for uid, groups in m["sequences"].items():
            for g, f in groups.items():
                if isinstance(f, dict) and "download_url" in f:
                    items.append(_rec(uid, g, uid, f))
    elif "releases" in m:
        for rel, objs in m["releases"].items():
            objs = objs.get("objects", objs)
            for uid, groups in objs.items():
                for g, f in groups.items():
                    if isinstance(f, dict) and "download_url" in f:
                        items.append(_rec(uid, g, os.path.join(rel, uid), f))
    else:
        sys.exit(f"{TAG} manifest {path} has neither 'sequences' nor 'releases'")
    if not items:
        sys.exit(f"{TAG} manifest {path} lists no downloadable files")
    return items, m.get("sequence_config")


def earliest_expiry(items):
    """Earliest signed-URL expiry (unix seconds) across items, or None."""
    exp = None
    for it in items:
        mo = OE_RE.search(it["url"])
        if mo:
            t = int(mo.group(1), 16)
            exp = t if exp is None else min(exp, t)
    return exp


# ---------------------------------------------------------------- selection
def parse_list(spec):
    """'all' -> None; 'a,b' -> {a,b}; '@file' -> lines of file."""
    if spec is None or spec.strip().lower() in ("", "all"):
        return None
    if spec.startswith("@"):
        with open(spec[1:]) as fh:
            return {ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")}
    return {s.strip() for s in spec.split(",") if s.strip()}


def select(items, groups, exclude, seqs):
    all_groups = {it["group"] for it in items}
    if groups is not None:
        unknown = groups - all_groups
        if unknown:
            log(f"WARNING: unknown groups ignored: {sorted(unknown)}; available: {sorted(all_groups)}")
    if seqs is not None:
        all_uids = {it["uid"] for it in items}
        unknown = seqs - all_uids
        if unknown:
            log(f"WARNING: unknown sequences ignored: {sorted(unknown)[:10]}{' ...' if len(unknown) > 10 else ''}")
    out = []
    for it in items:
        if groups is not None and it["group"] not in groups:
            continue
        if it["group"] in exclude:
            continue
        if seqs is not None and it["uid"] not in seqs:
            continue
        out.append(it)
    return out


# ---------------------------------------------------------------- disk state
def out_path(dest, it):
    return os.path.join(dest, it["subdir"], it["filename"])


def state_of(dest, it):
    """'complete' | 'partial' | 'missing' based on size + aria2 control file."""
    p = out_path(dest, it)
    if os.path.exists(p + ".aria2"):
        return "partial"
    try:
        sz = os.path.getsize(p)
    except OSError:
        return "missing"
    if it["size"] and sz == it["size"]:
        return "complete"
    return "partial"


def sha1_of(path, bufsize=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def quarantine(path, why):
    bad = f"{path}.corrupt-{int(time.time())}"
    os.replace(path, bad)
    log(f"QUARANTINED ({why}): {path} -> {os.path.basename(bad)}")


def summarize(dest, items, header):
    by_group = {}
    tot = {"n": 0, "done": 0, "bytes": 0, "done_bytes": 0}
    for it in items:
        st = state_of(dest, it)
        g = by_group.setdefault(it["group"], {"n": 0, "done": 0, "bytes": 0, "done_bytes": 0})
        for d in (g, tot):
            d["n"] += 1
            d["bytes"] += it["size"]
            if st == "complete":
                d["done"] += 1
                d["done_bytes"] += it["size"]
    log(header)
    for g, d in sorted(by_group.items(), key=lambda kv: -kv[1]["bytes"]):
        log(f"  {g:<36} {d['done']:>6}/{d['n']:<6} files  {human(d['done_bytes']):>10} / {human(d['bytes']):<10}")
    pct = 100.0 * tot["done_bytes"] / tot["bytes"] if tot["bytes"] else 100.0
    log(f"  {'TOTAL':<36} {tot['done']:>6}/{tot['n']:<6} files  {human(tot['done_bytes']):>10} / {human(tot['bytes']):<10} ({pct:.1f}%)")
    return tot


# ---------------------------------------------------------------- transport
def probe_direct(url, timeout):
    """True if a ranged GET of `url` succeeds with NO proxy within `timeout` s."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "aria-dl/1"})
    try:
        with opener.open(req, timeout=timeout) as r:
            r.read(1)
            return r.status in (200, 206)
    except Exception as e:  # noqa: BLE001 - any failure means "not reachable direct"
        log(f"direct probe failed: {type(e).__name__}: {str(e)[:80]}")
        return False


def choose_transport(mode, proxy, probe_url, timeout):
    """Return (label, aria2c --all-proxy value)."""
    if mode == "direct":
        log(f"probing direct (no-proxy) access to the CDN ({timeout}s cap) ...")
        if probe_direct(probe_url, timeout):
            log("direct access works -> downloading WITHOUT proxy")
            return "direct", ""
        log("direct CDN access unavailable and transport=direct -> NOT using any proxy (exit 4)")
        sys.exit(4)
    if mode == "proxy":
        if not proxy:
            sys.exit(f"{TAG} --transport proxy but no proxy given (set https_proxy or --proxy)")
        return "proxy", proxy
    # auto
    log(f"probing direct (no-proxy) access to the CDN ({timeout}s cap) ...")
    if probe_direct(probe_url, timeout):
        log("direct access works -> downloading WITHOUT proxy")
        return "direct", ""
    if not proxy:
        sys.exit(f"{TAG} CDN not reachable direct and no proxy configured (https_proxy unset)")
    log(f"WARNING: CDN not reachable without proxy from this host -> falling back to proxy {proxy}")
    return "proxy", proxy


# ---------------------------------------------------------------- aria2c
def write_input(path, dest, todo):
    with open(path, "w") as fh:
        for it in todo:
            fh.write(it["url"] + "\n")
            fh.write(f"  dir={os.path.join(dest, it['subdir'])}\n")
            fh.write(f"  out={it['filename']}\n")
            if it["sha1"]:
                fh.write(f"  checksum=sha-1={it['sha1']}\n")


def run_aria2c(args, input_file, log_file, all_proxy):
    cmd = [
        args.aria2c,
        "-i", input_file,
        "-c",
        "-j", str(args.workers),
        "-x", str(args.conns),
        "-s", str(args.conns),
        "--min-split-size=8M",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--check-integrity=true",
        "--console-log-level=warn",
        f"--summary-interval={args.summary_interval}",
        "--retry-wait=10",
        f"--max-tries={args.max_tries}",
        "--timeout=60",
        "--connect-timeout=30",
        f"--lowest-speed-limit={args.speed_floor}",
        f"--all-proxy={all_proxy}",
        f"--log={log_file}",
        "--log-level=notice",
        "--user-agent=aria-dl/1",
    ]
    if args.max_overall_rate:
        cmd.append(f"--max-overall-download-limit={args.max_overall_rate}")
    log("aria2c " + " ".join(c if not c.startswith("--all-proxy=http") else "--all-proxy=<proxy>" for c in cmd[1:]))
    t0 = time.time()
    rc = subprocess.call(cmd)
    log(f"aria2c exited rc={rc} after {time.time() - t0:.0f}s")
    return rc


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="*_download_urls.json from the explorer API")
    ap.add_argument("--dest", help="output root (required unless --expiry)")
    ap.add_argument("--groups", default="all", help="comma list of data groups, or 'all'")
    ap.add_argument("--exclude-groups", default="", help="comma list of groups to skip")
    ap.add_argument("--sequences", default="all", help="comma list of sequence uids, '@file', or 'all'")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 4)), help="concurrent files (aria2c -j)")
    ap.add_argument("--conns", type=int, default=int(os.environ.get("ARIA2_CONNS", 8)), help="connections per file (aria2c -x/-s)")
    ap.add_argument("--speed-floor", default=os.environ.get("ARIA2_SPEED_FLOOR", "100K"), help="aria2c --lowest-speed-limit")
    ap.add_argument("--max-tries", type=int, default=5, help="aria2c --max-tries per file within one pass")
    ap.add_argument("--max-overall-rate", default=os.environ.get("ARIA_MAX_RATE", ""), help="aria2c --max-overall-download-limit (e.g. 20M)")
    ap.add_argument("--summary-interval", type=int, default=120)
    ap.add_argument("--transport", default=os.environ.get("ARIA_TRANSPORT", "direct"), choices=["direct", "auto", "proxy"],
                    help="direct (default, never a proxy) | auto (direct, else $https_proxy) | proxy")
    ap.add_argument("--proxy", default=os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or os.environ.get("all_proxy") or "")
    ap.add_argument("--probe-timeout", type=int, default=15)
    ap.add_argument("--min-ttl", type=int, default=3600, help="exit 3 if links expire within this many seconds")
    ap.add_argument("--max-files", type=int, default=0, help="cap files handed to aria2c this pass (testing)")
    ap.add_argument("--small-first", action="store_true", help="order the pass by file size instead of manifest order")
    ap.add_argument("--verify", action="store_true", help="re-hash every file already complete on disk (slow)")
    ap.add_argument("--status", action="store_true", help="print completion summary and exit 0")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, do not download")
    ap.add_argument("--expiry", action="store_true", help="print earliest link expiry as '<epoch> <utc>' and exit")
    ap.add_argument("--aria2c", default="aria2c")
    args = ap.parse_args()

    items, _cfg = load_manifest(args.manifest)
    exp = earliest_expiry(items)
    if args.expiry:
        print(f"{exp or 0} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(exp)) if exp else 'unknown'}")
        return 0
    if not args.dest:
        ap.error("--dest is required")
    dest = os.path.abspath(args.dest)

    groups = parse_list(args.groups)
    exclude = parse_list(args.exclude_groups) or set()
    seqs = parse_list(args.sequences)
    sel = select(items, groups, exclude, seqs)
    if not sel:
        sys.exit(f"{TAG} selection is empty (groups={args.groups} sequences={args.sequences})")
    nseq = len({it['uid'] for it in sel})
    log(f"manifest {os.path.basename(args.manifest)}: {len(items)} files; selected {len(sel)} files in {nseq} sequences, "
        f"groups={'all' if groups is None else ','.join(sorted(groups))}"
        + (f" minus {','.join(sorted(exclude))}" if exclude else ""))
    if exp:
        log(f"links expire {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(exp))} ({(exp - time.time()) / 86400:.1f} days)")

    if args.verify:
        n_bad = 0
        for it in sel:
            if state_of(dest, it) == "complete" and it["sha1"]:
                p = out_path(dest, it)
                if sha1_of(p) != it["sha1"]:
                    quarantine(p, "sha1 mismatch on --verify")
                    n_bad += 1
        log(f"--verify: re-hashed complete files, {n_bad} quarantined")

    tot = summarize(dest, sel, f"status in {dest}:")
    if args.status:
        return 0
    todo = [it for it in sel if state_of(dest, it) != "complete"]
    if not todo:
        log("everything selected is complete")
        return 0
    if exp and exp - time.time() < args.min_ttl:
        log(f"manifest links expire in {max(0, exp - time.time()) / 60:.0f} min -> refresh the manifest (exit 3)")
        return 3
    if args.small_first:
        todo.sort(key=lambda it: it["size"])
    if args.max_files and len(todo) > args.max_files:
        log(f"--max-files: handing {args.max_files} of {len(todo)} pending files to aria2c this pass")
        todo = todo[: args.max_files]
    remaining = sum(it["size"] for it in todo)
    log(f"this pass: {len(todo)} files, {human(remaining)} (partials resume)")
    if args.dry_run:
        for it in todo[:20]:
            log(f"  {state_of(dest, it):<8} {human(it['size']):>10}  {it['subdir']}/{it['filename']}")
        if len(todo) > 20:
            log(f"  ... {len(todo) - 20} more")
        return 1

    # Decide the transport first: with --transport direct this exits 4 when the
    # CDN is unreachable, before any directory or state file is created.
    _label, all_proxy = choose_transport(args.transport, args.proxy, todo[0]["url"], args.probe_timeout)

    state_dir = os.path.join(dest, ".aria_dl")
    os.makedirs(state_dir, exist_ok=True)
    for it in todo:
        os.makedirs(os.path.join(dest, it["subdir"]), exist_ok=True)
    input_file = os.path.join(state_dir, "aria2.input")
    write_input(input_file, dest, todo)
    t_start = time.time()
    run_aria2c(args, input_file, os.path.join(state_dir, "aria2c.log"), all_proxy)

    # Re-verify whatever finished in this pass (aria2c already checked the sha-1
    # via checksum=, this guards against a file left full-size after an error).
    n_new = n_bad = 0
    for it in todo:
        p = out_path(dest, it)
        if state_of(dest, it) != "complete":
            continue
        try:
            fresh = os.path.getmtime(p) >= t_start - 5
        except OSError:
            continue
        if fresh and it["sha1"]:
            n_new += 1
            if sha1_of(p) != it["sha1"]:
                quarantine(p, "sha1 mismatch after download")
                n_bad += 1
    log(f"post-pass verification: {n_new} newly complete files hashed, {n_bad} quarantined")
    tot = summarize(dest, sel, f"status after pass in {dest}:")
    return 0 if tot["done"] == tot["n"] else 1


if __name__ == "__main__":
    sys.exit(main())
