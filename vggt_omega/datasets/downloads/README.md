# Dataset download scripts

One `download_<dataset>.sh` per training dataset used by VGGT-Omega, split into
two folders by how each one is accessed:

- **`open/`** — runs directly; no license, key, account, or manual step needed.
- **`gated/`** — needs a license, key, account, or a one-time manual step first.
  These print the exact manual steps and exit non-zero (they never guess a URL).

## Usage

```bash
# Default target dir is $DOWNLOAD_ROOT/<dataset>  (DOWNLOAD_ROOT defaults to /ipfs/babiking/datasets/3DR)
./open/download_co3d.sh                            # -> /ipfs/babiking/datasets/3DR/co3d
./open/download_co3d.sh /my/path/co3d              # explicit target
DOWNLOAD_ROOT=/data ./open/download_megadepth.sh   # override the root for all scripts (legacy DATA_ROOT also works)
./gated/download_scannet.sh                    # gated: prints the manual steps, exits non-zero
```

Each script either downloads automatically (the `open/` ones) or, for datasets
behind a license / signup / gated repo (the `gated/` ones), prints the exact
manual steps and exits with a non-zero status (it never guesses a broken URL).
Point the resulting path back at the matching `datasets/config/<dataset>.yaml`
(`*_DIR` keys).

`common.sh` lives at the top level and is sourced by every script as
`../common.sh`. It holds the shared helpers (`fetch`, `resolve_dest`,
`manual_gate`, `hf_download`, plus the uv helpers below).

## Layout

```
downloads/
  common.sh                 # shared helpers (sourced as ../common.sh)
  open/                     # runs directly, no license/key
    download_<dataset>.sh
  gated/                    # license / key / account / manual step
    download_<dataset>.sh
  envs/                     # one uv env per Python-needing dataset (stays top-level)
    co3d/        pyproject.toml  uv.lock
    tartanair/   pyproject.toml  uv.lock
    ...
```

## Python environments (uv)

Every dataset whose downloader needs Python tooling has its own **isolated,
declarative** environment under `envs/<dataset>/pyproject.toml`. The scripts
never touch the system interpreter or the project's main `.venv`: each Python
call goes through `uv`, which backs the env with a uv-managed CPython
(default 3.10, override with `UV_PYTHON`) in `envs/<dataset>/.venv` and a
committed `envs/<dataset>/uv.lock`. `envs/` stays at the top level and is
resolved from `common.sh`, so it works regardless of which folder a script is in.

Helpers (in `common.sh`):

- `uv_sync` — materialize `envs/$DATASET_NAME/.venv` from its `pyproject.toml` (idempotent).
- `uv_py …` — run `python …` inside that env (`uv run --project`).
- `uv_tool <cmd> …` — run a console entry point (e.g. `omnitools.download`, `huggingface-cli`).
- `uv_pip install -e <repo>` — editable-install a cloned upstream repo into the env (uCO3D).

You normally just run the dataset script; it calls `uv_sync` itself. To prep an
env ahead of time: `uv sync --project envs/co3d --python 3.10`. The `.venv`
dirs and runtime-cloned upstream repos (`_*_repo/`) are git-ignored; the
`pyproject.toml` + `uv.lock` are committed.

Datasets with **no Python dependency** (pure `wget`/`curl`: megadepth, replica,
eden, paralleldomain4d, unrealstereo4k, vkitti, tum, seven_scenes, mvs_synth, mipnerf360, plus the
manual-access-only mapfree/mpsd/mapillary/bedlam/behavior1k/sail_vos and the
`gsutil`-based waymo) have **no** `envs/` entry. `habitat` is special:
`habitat-sim` is conda-only (not on PyPI), so that script uses `$HABITAT_PYTHON`
(your conda interpreter), not a uv env.

## Access matrix

### `open/` — runs directly (no license/key)

| Script | Source | Access |
|---|---|---|
| `open/download_co3d.sh` | FAIR repo downloader (CDN) | Open |
| `open/download_uco3d.sh` | FAIR repo / HF `facebook/uco3d` | Open |
| `open/download_megadepth.sh` | Cornell direct wget | Open |
| `open/download_hypersim.sh` | `apple/ml-hypersim` script | Open |
| `open/download_replica.sh` | GitHub release tarballs | Open |
| `open/download_megasynth.sh` | HF `hwjiang/MegaSynth` | Open |
| `open/download_mvs_synth.sh` | HF `phuang17/MVS-Synth` | Open |
| `open/download_eden.sh` | UvA ISIS direct wget | Open |
| `open/download_paralleldomain4d.sh` | TRI public S3 | Open |
| `open/download_unrealstereo4k.sh` | autonomousvision S3 / HF | Open |
| `open/download_vkitti.sh` | NAVER LABS direct wget | Open (CC BY-NC-SA) |
| `open/download_tum.sh` | TUM CVG per-sequence `.tgz` (direct wget) | Open |
| `open/download_seven_scenes.sh` | Microsoft CDN per-scene `.zip` (direct wget) | Open |
| `open/download_mipnerf360.sh` | Google Research GCS `.zip` (direct wget) | Open |
| `open/download_tartanair.sh` | `tartanair` pkg (uv env) | Open |
| `open/download_tartanground.sh` | `tartanair` pkg (uv env) | Open |
| `open/download_wildrgbd.sh` | repo `download.py` (HF, uv env) | Open |
| `open/download_taskonomy.sh` | `omnidata-tools` (uv env, `--agree`) | Open (click-through) |

### `gated/` — license / key / account / manual step

| Script | Source | Access |
|---|---|---|
| `gated/download_dl3dv.sh` | gated HF `DL3DV/*` | Gated HF + token |
| `gated/download_midair.sh` | checkbox config + helper script | Manual config step |
| `gated/download_dynamic_replica.sh` | project `links.json` + script | License click |
| `gated/download_sail_vos.sh` | project `download_sailvos.sh` | License click |
| `gated/download_mapfree.sh` | Niantic page buttons | License click |
| `gated/download_bedlam.sh` | MPI BEDLAM portal (login + license) | Account + license |
| `gated/download_behavior1k.sh` | StanfordVL `setup.sh` (OmniGibson) | EULA (interactive) |
| `gated/download_aria.sh` | Project Aria email JSON + tool | Email signup |
| `gated/download_efm3d.sh` | Project Aria email JSON + repo | Email signup |
| `gated/download_hot3d.sh` | Project Aria email JSON + tool | License + email |
| `gated/download_habitat.sh` | Matterport API token (HM3D) | Form + EULA + token |
| `gated/download_mapillary_metropolis.sh` | Mapillary portal | Account + license |
| `gated/download_mpsd.sh` | Mapillary portal | Account + license |
| `gated/download_scannet.sh` | ScanNet / ScanNet++ TOS scripts | Signed form |
| `gated/download_waymo.sh` | GCS via `gsutil` | Gated login |

Sizes are large (several datasets are multi-TB); review each script's inline
comments and uncomment/limit subsets before running the full download.
