# Dataset download scripts

One `download_<dataset>.sh` per training dataset used by VGGT-Omega.

## Usage

```bash
# Default target dir is $DATA_ROOT/<dataset>  (DATA_ROOT defaults to /jfs/Data_4DFF/train_data)
./download_co3d.sh                       # -> /jfs/Data_4DFF/train_data/co3d
./download_co3d.sh /my/path/co3d         # explicit target
DATA_ROOT=/data ./download_megadepth.sh  # override the root for all scripts
```

Each script either downloads automatically (open datasets) or, for datasets
behind a license / signup / gated repo, prints the exact manual steps and exits
with a non-zero status (it never guesses a broken URL). Point the resulting
path back at the matching `datasets/config/<dataset>.yaml` (`*_DIR` keys).

`common.sh` holds the shared helpers (`fetch`, `resolve_dest`, `manual_gate`,
`hf_download`, plus the uv helpers below) and is sourced by every script.

## Python environments (uv)

Every dataset whose downloader needs Python tooling has its own **isolated,
declarative** environment under `envs/<dataset>/pyproject.toml`. The scripts
never touch the system interpreter or the project's main `.venv`: each Python
call goes through `uv`, which backs the env with a uv-managed CPython
(default 3.10, override with `UV_PYTHON`) in `envs/<dataset>/.venv` and a
committed `envs/<dataset>/uv.lock`.

```
downloads/
  common.sh                 # uv_sync / uv_py / uv_tool / uv_pip helpers
  download_<dataset>.sh
  envs/
    co3d/        pyproject.toml  uv.lock   # one env per Python-needing dataset
    tartanair/   pyproject.toml  uv.lock
    ...
```

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
eden, paralleldomain4d, unrealstereo4k, vkitti, mvs_synth, plus the
manual-gate-only mapfree/mpsd/mapillary/bedlam/behavior1k/sail_vos and the
`gsutil`-based waymo) have **no** `envs/` entry. `habitat` is special:
`habitat-sim` is conda-only (not on PyPI), so that script uses `$HABITAT_PYTHON`
(your conda interpreter), not a uv env.

## Access matrix


| Script | Source | Access |
|---|---|---|
| `download_co3d.sh` | FAIR repo downloader (CDN) | Open |
| `download_uco3d.sh` | FAIR repo / HF `facebook/uco3d` | Open |
| `download_megadepth.sh` | Cornell direct wget | Open |
| `download_hypersim.sh` | `apple/ml-hypersim` script | Open |
| `download_replica.sh` | GitHub release tarballs | Open |
| `download_megasynth.sh` | HF `hwjiang/MegaSynth` | Open |
| `download_mvs_synth.sh` | HF `phuang17/MVS-Synth` | Open |
| `download_eden.sh` | UvA ISIS direct wget | Open |
| `download_paralleldomain4d.sh` | TRI public S3 | Open |
| `download_unrealstereo4k.sh` | autonomousvision S3 / HF | Open |
| `download_vkitti.sh` | NAVER LABS direct wget | Open (CC BY-NC-SA) |
| `download_tartanair.sh` | `tartanair` pkg (uv env) | Open |
| `download_tartanground.sh` | `tartanair` pkg (uv env) | Open |
| `download_wildrgbd.sh` | repo `download.py` (HF, uv env) | Open |
| `download_taskonomy.sh` | `omnidata-tools` (uv env, `--agree`) | Open (click-through) |
| `download_midair.sh` | checkbox config + script | Open (config step) |
| `download_dl3dv.sh` | gated HF `DL3DV/*` | Gated HF (auto) |
| `download_dynamic_replica.sh` | project `links.json` + script | License click |
| `download_sail_vos.sh` | project `download_sailvos.sh` | License click |
| `download_mapfree.sh` | Niantic page buttons | License click |
| `download_aria.sh` | Project Aria email JSON + tool | Email signup |
| `download_efm3d.sh` | Project Aria email JSON + repo | Email signup |
| `download_hot3d.sh` | Project Aria email JSON + tool | License + email |
| `download_habitat.sh` | Matterport API token (HM3D) | Form + EULA |
| `download_mapillary_metropolis.sh` | Mapillary portal | Account + license |
| `download_mpsd.sh` | Mapillary portal | Account + license |
| `download_scannet.sh` | ScanNet / ScanNet++ TOS scripts | Signed form |
| `download_waymo.sh` | GCS via `gsutil` | Gated login |

Sizes are large (several datasets are multi-TB); review each script's inline
comments and uncomment/limit subsets before running the full download.
