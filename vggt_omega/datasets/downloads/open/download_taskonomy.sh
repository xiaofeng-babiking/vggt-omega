#!/usr/bin/env bash
# Taskonomy -- via the Omnidata download tool (uses aria2). ~11 TB full.
# Storage: fullplus ~11.16 TB; smaller subsets via SUBSET= (medium ~2.4 TB, tiny ~115 GB,
#          debug a few GB).
# Homepage: http://taskonomy.stanford.edu/  Tool: pip install omnidata-tools
DATASET_NAME="taskonomy"
source "$(dirname "$0")/../common.sh"

DEST_DIR="$(resolve_dest "${1:-}" "taskonomy")"
log "Target: $DEST_DIR"

require_cmd aria2c "Install with: sudo apt-get install -y aria2"
uv_sync   # isolated uv env: envs/taskonomy (omnidata-tools)

# subset: debug | tiny | medium | full | fullplus (default fullplus = everything)
SUBSET="${SUBSET:-fullplus}"

# Proxy routing: the license clickthrough posts to docs.google.com (needs the
# proxy on this network), while the data host datasets.epfl.ch is faster and
# more reliable DIRECT -- so exempt it (and aria2's localhost RPC).
export no_proxy="datasets.epfl.ch,localhost,127.0.0.1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

# datasets.epfl.ch drops connections mid-transfer, and omnidata fetches its
# multi-MB link index with a bare requests.get (no retry) -- one drop kills
# the whole run before any download starts. Patch the installed package
# (idempotent, marker-guarded) so the index fetch retries.
uv_py - <<'PY'
import pathlib, omnidata_tools.dataset.starter_dataset as sd
p = pathlib.Path(sd.__file__)
src = p.read_text()
if 'PATCHED(vggt-omega)' in src:
    print('omnidata link-index retry patch: already applied')
else:
    old = """  @functools.cached_property
  def links(self): return [k for k in requests.get(self.link_file).text.splitlines()
      if k.endswith(self.expected_suffix) and not any([d in k for d in ('depth_zbuffer2', 'mask_valid2')])]"""
    new = """  @functools.cached_property
  def links(self):
    # PATCHED(vggt-omega): datasets.epfl.ch drops connections mid-body and a
    # bare requests.get dies on IncompleteRead -- retry the index fetch.
    import time
    last = None
    for _ in range(20):
      try:
        text = requests.get(self.link_file, timeout=120).text
        return [k for k in text.splitlines()
            if k.endswith(self.expected_suffix) and not any([d in k for d in ('depth_zbuffer2', 'mask_valid2')])]
      except Exception as e:
        last = e
        time.sleep(5)
    raise last"""
    assert old in src, 'omnidata source changed upstream; update the patch in download_taskonomy.sh'
    p.write_text(src.replace(old, new))
    print('omnidata link-index retry patch: applied to', p)
PY

log "Downloading Taskonomy via omnitools (subset=$SUBSET); --agree accepts the terms"
# omnitools refuses --agree without a name and (syntactically valid) email.
uv_tool omnitools.download all \
    --components taskonomy \
    --subset "$SUBSET" \
    --dest "$DEST_DIR" \
    --connections_total 40 \
    --agree \
    --name "${TASKONOMY_NAME:-$USER}" \
    --email "${TASKONOMY_EMAIL:?omnitools needs an email: TASKONOMY_EMAIL=you@example.com $0}" \
    "${@:2}"

log "Done. TASKONOMY_DIR=$DEST_DIR"
