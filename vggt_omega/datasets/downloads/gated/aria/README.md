# Project Aria datasets — crawl of explorer.projectaria.com (2026-08-26)

Everything the **Aria Dataset Explorer** (https://explorer.projectaria.com/) serves as of the crawl date, plus the linked **DTC Object Explorer** (https://dtc.projectaria.com/). Numbers come straight from the explorer's own JSON API (see *How the explorer serves data*), not from marketing pages. Machine-readable copy: `explorer_datasets.json`; raw roster: `versions.json`.

**Grand total: 11 sequence datasets, 4,239 sequences, 785 h of recordings, ≈ 130.11 TB if you take every data group — plus 2,399 DTC object models (≈ 152 GB).**

## Roster

| slug | dataset | gen | sequences | hours | full size | biggest groups | license | docs |
|---|---|---|---:|---:|---:|---|---|---|
| [`adt`](https://explorer.projectaria.com/adt) | Aria Digital Twin | Gen 1 | 236 | 8.1 | 2.28 TB | depth 1.3 TB, main_vrs 433 GB, synthetic 208 GB | [Aria dataset license](https://www.projectaria.com/datasets/adt/license) | [docs](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset) |
| [`nymeria`](https://explorer.projectaria.com/nymeria) | Nymeria | Gen 1 | 1,100 | 340.1 | 51.34 TB | recording_head_data_data_vrs 17.1 TB, recording_observer_data_data_vrs 16.3 TB, semidense_observations 11.6 TB | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode) | [docs](https://www.projectaria.com/datasets/nymeria/) |
| [`nymeria_plus`](https://explorer.projectaria.com/nymeria_plus) | NymeriaPlus | Gen 1 | 1,100 | 340.1 | 67.73 TB | recording_head_data_data_vrs 18.4 TB, recording_observer_data_data_vrs 17.5 TB, slam_semidense_observations 11.6 TB | — (none listed in `version_config`; upstream repo says CC BY-NC 4.0) | [docs](https://github.com/facebookresearch/nymeria_dataset) |
| [`hot3d-aria`](https://explorer.projectaria.com/hot3d-aria) | HOT3D-Aria | Gen 1 | 198 | 6.5 | 526.3 GB | main_vrs 329 GB, mps_artifacts 89 GB, mps_slam_points 84 GB | [Aria dataset license](https://www.projectaria.com/datasets/hot3d/license/) | [docs](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/hot3d) |
| [`hot3d-quest`](https://explorer.projectaria.com/hot3d-quest) | HOT3D-Quest | Gen 1 | 226 | 0.0 | 393.8 GB | main_vrs 386 GB, video_main_rgb 7 GB, hand_data 1 GB | [Aria dataset license](https://www.projectaria.com/datasets/hot3d/license/) | [docs](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/hot3d) |
| [`dtc`](https://explorer.projectaria.com/dtc) | DTC-Aria | Gen 1 | 200 | 1.6 | 307.0 GB | main_vrs 273 GB, video_main_rgb 24 GB, mps_artifacts 5 GB | [Aria dataset license](https://www.projectaria.com/datasets/dtc/license) | [docs](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/digital_twin_catalog) |
| [`aeo`](https://explorer.projectaria.com/aeo) | Aria Everyday Objects | Gen 1 | 25 | 0.8 | 18.7 GB | main_vrs 13 GB, video_main_rgb 3 GB, mps 2 GB | [Aria dataset license](https://www.projectaria.com/datasets/aeo/license) | [docs](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_objects) |
| [`aea`](https://explorer.projectaria.com/aea) | Aria Everyday Activities | Gen 1 | 143 | 7.6 | 396.9 GB | main_vrs 309 GB, mps_artifacts 34 GB, mps_slam_points 31 GB | [Aria dataset license](https://www.projectaria.com/datasets/aea/license) | [docs](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_activities_dataset) |
| [`ritw`](https://explorer.projectaria.com/ritw) | Reading In The Wild | Gen 1 | 987 | 78.7 | 6.83 TB | recording_vrs 4.4 TB, mps 2.2 TB, video_main_rgb 192 GB | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode) | [docs](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/ritw) |
| [`aria-scenes`](https://explorer.projectaria.com/aria-scenes) | Aria Scenes | Gen 1 | 12 | 0.5 | 133.2 GB | main_vrs 108 GB, video_main_rgb 18 GB, mps_artifacts 3 GB | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode) | [docs](https://www.projectaria.com/photoreal-reconstruction/) |
| [`gen2pilot`](https://explorer.projectaria.com/gen2pilot) | Aria Gen 2 Pilot Dataset | Gen 2 | 12 | 1.1 | 158.9 GB | mps_artifacts 45 GB, mps_slam_points 44 GB, main_vrs 35 GB | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode) | [docs](https://facebookresearch.github.io/projectaria_tools/gen2/research-tools/dataset/pilot/content) |
| [`objects`](https://dtc.projectaria.com/) | DTC Object Explorer (Digital Twin Catalog object library) | Gen 1 | 2,399 objects | — | 152 GB | 3d-asset_glb 118 GB, 3d-asset-manifold_glb 33 GB, 3d-asset-manifold-10k_glb 1 GB | [Aria dataset license](https://www.projectaria.com/datasets/dtc/license) | [docs](https://www.projectaria.com/datasets/dtc/) |

HOT3D-Quest has no `duration_s` field in its metadata, hence 0 h. *Full size* sums every data group of every sequence; the `main_vrs`/`recording_*` raw recordings and the semi-dense observation dumps dominate — a VRS-only or MPS-only pull is a small fraction (see per-dataset tables).

## How the explorer serves data (what the crawl used)

The explorer is a React SPA over a plain JSON API on the same origin. **No login, cookie, or email is required for any of these calls** — the `x-api-key` header the SPA sends is read from `localStorage` and is empty for public datasets. Verified 2026-08-26 with `wget` (note: `curl`/OpenSSL clients fail through this machine's proxy — use `wget`).

```bash
BASE=https://explorer.projectaria.com
wget -qO- $BASE/versions                          # roster: name, url_name (slug), gen, desc
wget -qO- $BASE/version_config/<slug>              # license/consent URL, docs URL, filterable metadata fields
wget -qO- $BASE/data/<slug>                        # per-sequence metadata (duration, scene, device, computed.* stats …)
wget -qO- $BASE/data/<slug>/previews               # per-sequence preview mp4 + Rerun .rrd (signed URLs)
wget -qO- $BASE/data/<slug>/download_links         # per-sequence {group: {filename, sha1sum, file_size_bytes, download_url}}
                                                   #   + sequence_config (which files each group contains)
# DTC objects: same endpoints on https://dtc.projectaria.com with slug 'objects' (nested releases/<release>/objects/<uid>/<group>)
```

Facts about the `download_links` manifests:

- `download_url`s are signed `scontent.xx.fbcdn.net` links; the `oe=` query parameter is the hex UNIX expiry. At crawl time every dataset's links expire **2026-09-24** (≈30 days out) except `aria-scenes`, whose served manifest already carried links expiring 2026-08-26 (server-side cache; re-fetch to refresh).
- Every file has `sha1sum` + `file_size_bytes`, so resumable downloads can be verified offline.
- The manifest is exactly the `<Dataset>_download_urls.json` that the official `aria_dataset_downloader` (`projectaria_tools`) and the per-dataset downloaders (`hot3d`, `nymeria_dataset`, `dtc_object_downloader`) consume via `--cdn_file`; the SPA's email dialog just gates the *browser* download of that same JSON.
- One signed URL was spot-checked with `wget --spider` (HTTP 200, `Remote file exists`) with no session — the CDN does not check referer/cookies.

## Per-dataset detail

### `adt` — Aria Digital Twin (Gen 1)

A real-world dataset with dynamic and photorealistic digital counterparts.

- Explorer: https://explorer.projectaria.com/adt  ·  Learn more: https://www.projectaria.com/datasets/adt/  ·  Docs: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset
- License / consent: https://www.projectaria.com/datasets/adt/license
- Release prefix in filenames: `ADT`  ·  236 sequences, 2,832 files, 8.1 h, **2.28 TB** total
- Main recording: `video.vrs` · MPS dir: `mps`

| data group | files | size | contents (from `sequence_config`) |
|---|---:|---:|---|
| `video_main_rgb` | 236 | 23.91 GB | preview RGB mp4 (browser preview, not sensor data) |
| `mps_slam_trajectories` | 236 | 3.02 GB | Machine Perception Services output (slam trajectories) |
| `mps_slam_calibration` | 236 | 0.05 GB | Machine Perception Services output (slam calibration) |
| `mps_slam_points` | 236 | 102.39 GB | Machine Perception Services output (slam points) |
| `mps_slam_summary` | 236 | 0.00 GB | Machine Perception Services output (slam summary) |
| `mps_eye_gaze` | 236 | 0.03 GB | Machine Perception Services output (eye gaze) |
| `mps_artifacts` | 236 | 105.48 GB | Machine Perception Services output (artifacts) |
| `main_vrs` | 236 | 432.73 GB | raw sensor recording `video.vrs` |
| `main_groundtruth` | 236 | 10.65 GB | `2d_bounding_box.csv`, `2d_bounding_box_with_skeleton.csv`, `3d_bounding_box.csv`, `Skeleton_T.json`, `Skeleton_C.json`, `aria_trajectory.csv` … (+5 more) |
| `segmentation` | 236 | 94.13 GB | `segmentations.vrs`, `segmentations_with_skeleton.vrs` |
| `depth` | 236 | 1.30 TB | `depth_images.vrs`, `depth_images_with_skeleton.vrs` |
| `synthetic` | 236 | 208.32 GB | `synthetic_video.vrs` |

<details><summary>filterable metadata fields</summary>

`sequence_uid`, `scene`, `activity`, `device_serial`, `duration_s`, `is_multi_person`, `num_skeletons`, `gt_creation_time`, `visible_object_names`, `visible_object_categories`, `object_names_interacted_with`, `object_categories_interacted_with`, `computed.light_intensity_lux_median`, `computed.trajectory_length_m`, `computed.covered_area_m2`, `computed.covered_volume_m3`, `computed.speed_mps_mean`

</details>

### `nymeria` — Nymeria (Gen 1)

A large-scale multimodal egocentric dataset for full-body motion understanding.

- Explorer: https://explorer.projectaria.com/nymeria  ·  Learn more: https://www.projectaria.com/datasets/nymeria/  ·  Docs: https://www.projectaria.com/datasets/nymeria/
- License / consent: https://creativecommons.org/licenses/by-nc/4.0/legalcode  ·  Download instructions: https://github.com/facebookresearch/nymeria_dataset?tab=readme-ov-file#getting-started
- Release prefix in filenames: `v0.0`  ·  1,100 sequences, 15,109 files, 340.1 h, **51.34 TB** total
- Main recording: `recording_head/data/data.vrs` · MPS dir: `recording_head/mps`

| data group | files | size | contents (from `sequence_config`) |
|---|---:|---:|---|
| `video_main_rgb` | 1,100 | 1.06 TB | preview RGB mp4 (browser preview, not sensor data) |
| `body_motion` | 1,100 | 1.09 TB | `body/xdata.npz`, `body/xdata_blueman.glb` |
| `recording_head` | 1,100 | 758.15 GB | `recording_head/data/motion.vrs`, `recording_head/data/et.vrs`, `recording_head/mps/slam/closed_loop_trajectory.csv`, `recording_head/mps/slam/semidense_points.csv.gz`, `recording_head/mps/slam/summary.json`, `recording_head/mps/slam/online_calibration.jsonl` … (+1 more) |
| `recording_lwrist` | 1,100 | 451.82 GB | `recording_lwrist/data/motion.vrs`, `recording_lwrist/mps/slam/closed_loop_trajectory.csv`, `recording_lwrist/mps/slam/semidense_points.csv.gz`, `recording_lwrist/mps/slam/summary.json`, `recording_lwrist/mps/slam/online_calibration.jsonl` |
| `recording_rwrist` | 1,100 | 467.47 GB | `recording_rwrist/data/motion.vrs`, `recording_rwrist/mps/slam/closed_loop_trajectory.csv`, `recording_rwrist/mps/slam/semidense_points.csv.gz`, `recording_rwrist/mps/slam/summary.json`, `recording_rwrist/mps/slam/online_calibration.jsonl` |
| `recording_observer` | 1,100 | 616.09 GB | `recording_observer/data/motion.vrs`, `recording_observer/data/et.vrs`, `recording_observer/mps/slam/closed_loop_trajectory.csv`, `recording_observer/mps/slam/semidense_points.csv.gz`, `recording_observer/mps/slam/summary.json`, `recording_observer/mps/slam/online_calibration.jsonl` … (+1 more) |
| `semidense_observations` | 1,100 | 11.62 TB | `recording_head/mps/slam/semidense_observations.csv.gz`, `recording_lwrist/mps/slam/semidense_observations.csv.gz`, `recording_rwrist/mps/slam/semidense_observations.csv.gz`, `recording_observer/mps/slam/semidense_observations.csv.gz` |
| `LICENSE` | 1,100 | 0.02 GB | per-sequence LICENSE |
| `metadata_json` | 1,100 | 0.00 GB | per-sequence metadata.json |
| `narration_atomic_action_csv` | 845 | 0.05 GB |  |
| `narration_activity_summarization_csv` | 864 | 0.01 GB |  |
| `recording_head_data_data_vrs` | 1,100 | 17.06 TB | raw `data.vrs` of that device |
| `recording_observer_data_data_vrs` | 1,100 | 16.26 TB | raw `data.vrs` of that device |
| `body_xdata_mvnx` | 1,064 | 1.96 TB |  |
| `narration_motion_narration_csv` | 236 | 0.02 GB |  |

<details><summary>filterable metadata fields</summary>

`sequence_uid`, `motion_narration`, `atomic_action`, `activity_summarization`, `computed.light_intensity_lux_median`, `computed.trajectory_length_m`, `computed.covered_area_m2`, `computed.covered_volume_m3`, `date`, `participant_id`, `act_id`, `location`, `script`, `action_duration_sec`, `has_two_participants`, `pt2`, `participant_gender`, `participant_height_cm`, `participant_weight_kg`, `participant_bmi`, `participant_age_group`, `participant_ethnicity`, `participant_xsens_suit_size`, `body_motion`, `head_data`, `head_slam`, `head_trajectory_m`, `head_duration_sec`, `head_personalized_gaze`, `left_wrist_data`, `left_wrist_slam`, `left_wrist_trajectory_m`, `left_wrist_duration_sec`, `right_wrist_data`, `right_wrist_slam`, `right_wrist_trajectory_m`, `right_wrist_duration_sec`, `observer_data`, `observer_slam`, `observer_personalized_gaze`, `observer_trajectory_m`, `observer_duration_sec`, `has_timesync`

</details>

### `nymeria_plus` — NymeriaPlus (Gen 1)

Upgraded Nymeria dataset with optimized human motion in open formats, new annotations of object bounding boxes/meshes and additional data modalities.

- Explorer: https://explorer.projectaria.com/nymeria_plus  ·  Learn more: https://github.com/facebookresearch/nymeria_dataset  ·  Docs: https://github.com/facebookresearch/nymeria_dataset
- License / consent: (none listed in version_config)  ·  Download instructions: https://github.com/facebookresearch/nymeria_dataset?tab=readme-ov-file#getting-started
- Release prefix in filenames: `v1.0`  ·  1,100 sequences, 18,683 files, 340.1 h, **67.73 TB** total
- Main recording: `recording_head/data/data.vrs` · MPS dir: `recording_head/mps`

| data group | files | size | contents (from `sequence_config`) |
|---|---:|---:|---|
| `video_main_rgb` | 1,100 | 1.08 TB | preview RGB mp4 (browser preview, not sensor data) |
| `body_raw` | 1,100 | 1.29 TB | `body/xdata.healthcheck`, `body/xdata.mvnx`, `body/xdata.npz` |
| `body_processed` | 1,100 | 350.46 GB | `body/xdata_mhr.glb`, `body/xdata_smpl_neutral.npz` |
| `slam` | 1,100 | 1.70 TB | `recording_head/mps/slam/closed_loop_trajectory.csv`, `recording_head/mps/slam/open_loop_trajectory.csv`, `recording_head/mps/slam/semidense_points.csv.gz`, `recording_head/mps/slam/summary.json`, `recording_head/mps/slam/online_calibration.jsonl`, `recording_lwrist/mps/slam/closed_loop_trajectory.csv` … (+14 more) |
| `slam_semidense_observations` | 1,100 | 11.62 TB | `recording_head/mps/slam/semidense_observations.csv.gz`, `recording_lwrist/mps/slam/semidense_observations.csv.gz`, `recording_rwrist/mps/slam/semidense_observations.csv.gz`, `recording_observer/mps/slam/semidense_observations.csv.gz` |
| `timesync_and_imu` | 1,100 | 350.96 GB | `recording_head/data/motion.vrs`, `recording_lwrist/data/motion.vrs`, `recording_rwrist/data/motion.vrs`, `recording_observer/data/motion.vrs` |
| `audio` | 1,100 | 2.47 TB | `recording_head/data/audio.vrs`, `recording_observer/data/audio.vrs` |
| `eye_tracking` | 1,100 | 435.25 GB | `recording_head/data/et.vrs`, `recording_head/mps/eye_gaze`, `recording_observer/data/et.vrs`, `recording_observer/mps/eye_gaze` |
| `object_bounding_box` | 1,100 | 124.94 GB | `objects/boxy` |
| `object_mesh` | 1,100 | 37.58 GB | `objects/shaper` |
| `narration` | 1,100 | 0.01 GB | `narration` |
| `LICENSE` | 1,100 | 0.04 GB | per-sequence LICENSE |
| `metadata_json` | 1,100 | 0.00 GB | per-sequence metadata.json |
| `recording_head_data_data_vrs` | 1,100 | 18.39 TB | raw `data.vrs` of that device |
| `recording_lwrist_data_data_vrs` | 1,094 | 6.26 TB | raw `data.vrs` of that device |
| `recording_rwrist_data_data_vrs` | 1,089 | 6.14 TB | raw `data.vrs` of that device |
| `recording_observer_data_data_vrs` | 1,100 | 17.50 TB | raw `data.vrs` of that device |

<details><summary>filterable metadata fields</summary>

`sequence_uid`, `computed.light_intensity_lux_median`, `computed.trajectory_length_m`, `computed.covered_area_m2`, `computed.covered_volume_m3`, `date`, `location`, `script`, `action_duration_sec`, `has_two_participants`, `head_data`, `head_slam`, `head_trajectory_m`, `head_duration_sec`, `head_general_gaze`, `head_personalized_gaze`, `left_wrist_data`, `left_wrist_motion`, `left_wrist_slam`, `left_wrist_trajectory_m`, `left_wrist_duration_sec`, `right_wrist_data`, `right_wrist_motion`, `right_wrist_slam`, `right_wrist_trajectory_m`, `right_wrist_duration_sec`, `observer_data`, `observer_slam`, `observer_general_gaze`, `observer_personalized_gaze`, `observer_trajectory_m`, `observer_duration_sec`, `timesync`, `motion_narration`, `atomic_action`, `activity_summarization`, `participant_gender`, `participant_height_cm`, `participant_weight_kg`, `participant_bmi`, `participant_age_group`, `participant_ethnicity`, `participant_xsens_suit_size`, `body_motion_xsens`, `body_motion_mhr`, `body_motion_smpl`, `objects_bounding_box`, `objects_shaper_mesh`

</details>

### `hot3d-aria` — HOT3D-Aria (Gen 1)

A new benchmark dataset for researching vision-based hand-object interaction.

- Explorer: https://explorer.projectaria.com/hot3d-aria  ·  Learn more: https://www.projectaria.com/datasets/hot3D/  ·  Docs: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/hot3d
- License / consent: https://www.projectaria.com/datasets/hot3d/license/  ·  Download instructions: https://github.com/facebookresearch/hot3d?tab=readme-ov-file#step-1-install-the-downloader
- Release prefix in filenames: `v4.0.0`  ·  198 sequences, 1,980 files, 6.5 h, **526.3 GB** total
- Main recording: `recording.vrs` · MPS dir: `mps`

| data group | files | size | contents (from `sequence_config`) |
|---|---:|---:|---|
| `video_main_rgb` | 198 | 17.78 GB | preview RGB mp4 (browser preview, not sensor data) |
| `mps_slam_trajectories` | 198 | 5.06 GB | Machine Perception Services output (slam trajectories) |
| `mps_slam_calibration` | 198 | 0.04 GB | Machine Perception Services output (slam calibration) |
| `mps_slam_points` | 198 | 83.65 GB | Machine Perception Services output (slam points) |
| `mps_slam_summary` | 198 | 0.00 GB | Machine Perception Services output (slam summary) |
| `mps_eye_gaze` | 198 | 0.05 GB | Machine Perception Services output (eye gaze) |
| `mps_artifacts` | 198 | 88.79 GB | Machine Perception Services output (artifacts) |
| `main_vrs` | 198 | 329.36 GB | raw sensor recording `recording.vrs` |
| `ground_truth` | 198 | 0.67 GB | `box2d_hands.csv`, `box2d_objects.csv`, `dynamic_objects.csv`, `headset_trajectory.csv`, `metadata.json`, `camera_models.json` … (+9 more) |
| `hand_data` | 198 | 0.96 GB | `umetrack_hand_pose_trajectory.jsonl`, `umetrack_hand_user_profile.json`, `mano_hand_pose_trajectory.jsonl`, `license.txt` |

<details><summary>filterable metadata fields</summary>

`sequence_uid`, `device_serial`, `duration_s`, `computed.light_intensity_lux_median`, `computed.trajectory_length_m`, `computed.covered_area_m2`, `computed.covered_volume_m3`, `computed.speed_mps_mean`, `have_hand_object_pose_gt`, `participant_id`, `object_bop_uids`, `object_names`, `object_uids`

</details>

### `hot3d-quest` — HOT3D-Quest (Gen 1)

A new benchmark dataset for researching vision-based hand-object interaction.

- Explorer: https://explorer.projectaria.com/hot3d-quest  ·  Learn more: https://www.projectaria.com/datasets/hot3D/  ·  Docs: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/hot3d
- License / consent: https://www.projectaria.com/datasets/hot3d/license/  ·  Download instructions: https://github.com/facebookresearch/hot3d?tab=readme-ov-file#step-1-install-the-downloader
- Release prefix in filenames: `v4.0.0`  ·  226 sequences, 904 files, 0.0 h, **393.8 GB** total
- Main recording: `recording.vrs` · MPS dir: `None`

| data group | files | size | contents (from `sequence_config`) |
|---|---:|---:|---|
| `video_main_rgb` | 226 | 7.24 GB | preview RGB mp4 (browser preview, not sensor data) |
| `main_vrs` | 226 | 385.66 GB | raw sensor recording `recording.vrs` |
| `ground_truth` | 226 | 0.40 GB | `box2d_hands.csv`, `box2d_objects.csv`, `dynamic_objects.csv`, `headset_trajectory.csv`, `metadata.json`, `camera_models.json` … (+8 more) |
| `hand_data` | 226 | 0.52 GB | `umetrack_hand_pose_trajectory.jsonl`, `umetrack_hand_user_profile.json`, `mano_hand_pose_trajectory.jsonl`, `license.txt` |

<details><summary>filterable metadata fields</summary>

`sequence_uid`, `have_hand_object_pose_gt`, `participant_id`, `object_bop_uids`, `object_names`, `object_uids`

</details>

### `dtc` — DTC-Aria (Gen 1)

The world's highest quality dataset for object reconstruction research.

- Explorer: https://explorer.projectaria.com/dtc  ·  Learn more: https://www.projectaria.com/datasets/dtc/  ·  Docs: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/digital_twin_catalog
- License / consent: https://www.projectaria.com/datasets/dtc/license
- Release prefix in filenames: `v1.0.1`  ·  200 sequences, 1,600 files, 1.6 h, **307.0 GB** total
- Main recording: `video.vrs` · MPS dir: `mps`

| data group | files | size | contents (from `sequence_config`) |
|---|---:|---:|---|
| `video_main_rgb` | 200 | 23.53 GB | preview RGB mp4 (browser preview, not sensor data) |
| `mps_slam_trajectories` | 200 | 0.65 GB | Machine Perception Services output (slam trajectories) |
| `mps_slam_calibration` | 200 | 0.01 GB | Machine Perception Services output (slam calibration) |
| `mps_slam_points` | 200 | 4.42 GB | Machine Perception Services output (slam points) |
| `mps_slam_summary` | 200 | 0.00 GB | Machine Perception Services output (slam summary) |
| `mps_artifacts` | 200 | 5.08 GB | Machine Perception Services output (artifacts) |
| `main_vrs` | 200 | 273.32 GB | raw sensor recording `video.vrs` |
| `main_groundtruth` | 200 | 0.00 GB | `object_pose.json`, `metadata.json`, `CC_BY-SA.txt` |

<details><summary>filterable metadata fields</summary>

`sequence_uid`, `device_serial`, `duration_s`, `mode`, `model_name`, `computed.light_intensity_lux_median`, `computed.trajectory_length_m`, `computed.covered_area_m2`, `computed.covered_volume_m3`, `computed.speed_mps_mean`

</details>

### `aeo` — Aria Everyday Objects (Gen 1)

A small-scale, real-world Project Aria dataset with high quality 3D OBB annotations.

- Explorer: https://explorer.projectaria.com/aeo  ·  Learn more: https://www.projectaria.com/datasets/aeo/  ·  Docs: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_objects
- License / consent: https://www.projectaria.com/datasets/aeo/license
- Release prefix in filenames: `1.0.2`  ·  25 sequences, 100 files, 0.8 h, **18.7 GB** total
- Main recording: `main.vrs` · MPS dir: `mps`

| data group | files | size | contents (from `sequence_config`) |
|---|---:|---:|---|
| `video_main_rgb` | 25 | 2.86 GB | preview RGB mp4 (browser preview, not sensor data) |
| `mps` | 25 | 2.44 GB | `mps/slam/closed_loop_trajectory.csv`, `mps/slam/online_calibration.jsonl`, `mps/slam/semidense_points.csv.gz`, `mps/slam/semidense_observations.csv.gz` |
| `main_groundtruth` | 25 | 0.03 GB | `2d_bounding_box.csv`, `3d_bounding_box.csv`, `instances.json`, `scene_objects.csv` |
| `main_vrs` | 25 | 13.33 GB | raw sensor recording `main.vrs` |

<details><summary>filterable metadata fields</summary>

`sequence_uid`, `device_serial`, `duration_s`, `scene`, `visible_object_names`, `computed.light_intensity_lux_median`, `computed.trajectory_length_m`, `computed.covered_area_m2`, `computed.covered_volume_m3`, `computed.speed_mps_mean`

</details>

### `aea` — Aria Everyday Activities (Gen 1)

A re-release of Aria’s first Pilot Dataset, updated with new tooling and location data, to accelerate the state of machine perception and AI.

- Explorer: https://explorer.projectaria.com/aea  ·  Learn more: https://www.projectaria.com/datasets/aea/  ·  Docs: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_activities_dataset
- License / consent: https://www.projectaria.com/datasets/aea/license
- Release prefix in filenames: `1.0.0`  ·  143 sequences, 1,287 files, 7.6 h, **396.9 GB** total
- Main recording: `recording.vrs` · MPS dir: `mps`

| data group | files | size | contents (from `sequence_config`) |
|---|---:|---:|---|
| `video_main_rgb` | 143 | 19.84 GB | preview RGB mp4 (browser preview, not sensor data) |
| `mps_slam_trajectories` | 143 | 2.97 GB | Machine Perception Services output (slam trajectories) |
| `mps_slam_calibration` | 143 | 0.18 GB | Machine Perception Services output (slam calibration) |
| `mps_slam_points` | 143 | 30.92 GB | Machine Perception Services output (slam points) |
| `mps_slam_summary` | 143 | 0.00 GB | Machine Perception Services output (slam summary) |
| `mps_eye_gaze` | 143 | 0.01 GB | Machine Perception Services output (eye gaze) |
| `mps_artifacts` | 143 | 34.08 GB | Machine Perception Services output (artifacts) |
| `main_vrs` | 143 | 308.93 GB | raw sensor recording `recording.vrs` |
| `annotations` | 143 | 0.00 GB | `speech.csv`, `metadata.json` |

<details><summary>filterable metadata fields</summary>

`sequence_uid`, `device_serial`, `duration_s`, `computed.light_intensity_lux_median`, `computed.trajectory_length_m`, `computed.covered_area_m2`, `computed.covered_volume_m3`, `computed.speed_mps_mean`

</details>

### `ritw` — Reading In The Wild (Gen 1)

Large-scale multimodal dataset with diverse reading and non-reading activities, featuring high-frequency eye-tracking data.

- Explorer: https://explorer.projectaria.com/ritw  ·  Learn more: https://www.projectaria.com/datasets/reading-in-the-wild/  ·  Docs: https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/ritw
- License / consent: https://creativecommons.org/licenses/by-nc/4.0/legalcode
- Release prefix in filenames: `v1.0`  ·  987 sequences, 3,948 files, 78.7 h, **6.83 TB** total
- Main recording: `recording.vrs` · MPS dir: `mps`

| data group | files | size | contents (from `sequence_config`) |
|---|---:|---:|---|
| `video_main_rgb` | 987 | 192.10 GB | preview RGB mp4 (browser preview, not sensor data) |
| `recording_vrs` | 987 | 4.44 TB | `recording.vrs` |
| `metadata_json` | 987 | 0.00 GB | `metadata.json` |
| `mps` | 987 | 2.19 TB | `mps` |

<details><summary>filterable metadata fields</summary>

`sequence_uid`, `duration_s`, `upload_date`, `task`, `campaign_task_name`, `mode`, `medium`, `material`, `reading`, `calibration_success`, `computed.light_intensity_lux_median`, `computed.trajectory_length_m`, `computed.covered_area_m2`, `computed.covered_volume_m3`, `computed.speed_mps_mean`

</details>

### `aria-scenes` — Aria Scenes (Gen 1)

Aria Scenes is a benchmark dataset for future research on photorealistic reconstruction. The dataset includes 12 .vrs files created in diverse indoor and outdoor environments.

- Explorer: https://explorer.projectaria.com/aria-scenes  ·  Learn more: https://www.projectaria.com/photoreal-reconstruction/  ·  Docs: https://www.projectaria.com/photoreal-reconstruction/
- License / consent: https://creativecommons.org/licenses/by-nc/4.0/legalcode
- Release prefix in filenames: `v0.2`  ·  12 sequences, 84 files, 0.5 h, **133.2 GB** total
- Main recording: `recording.vrs` · MPS dir: `mps`

| data group | files | size | contents (from `sequence_config`) |
|---|---:|---:|---|
| `video_main_rgb` | 12 | 17.98 GB | preview RGB mp4 (browser preview, not sensor data) |
| `mps_slam_trajectories` | 12 | 0.31 GB | Machine Perception Services output (slam trajectories) |
| `mps_slam_calibration` | 12 | 0.00 GB | Machine Perception Services output (slam calibration) |
| `mps_slam_points` | 12 | 3.08 GB | Machine Perception Services output (slam points) |
| `mps_slam_summary` | 12 | 0.00 GB | Machine Perception Services output (slam summary) |
| `mps_artifacts` | 12 | 3.39 GB | Machine Perception Services output (artifacts) |
| `main_vrs` | 12 | 108.41 GB | raw sensor recording `recording.vrs` |

<details><summary>filterable metadata fields</summary>

`duration_s`, `sensor.slam_camera_fps`

</details>

### `gen2pilot` — Aria Gen 2 Pilot Dataset (Gen 2)

A dataset for understanding the fundamental capabilities of the Aria Gen 2 device for your research applications.

- Explorer: https://explorer.projectaria.com/gen2pilot  ·  Learn more: https://www.projectaria.com/datasets/gen2pilot/  ·  Docs: https://facebookresearch.github.io/projectaria_tools/gen2/research-tools/dataset/pilot/content
- License / consent: https://creativecommons.org/licenses/by-nc/4.0/legalcode
- Release prefix in filenames: `v1.0`  ·  12 sequences, 156 files, 1.1 h, **158.9 GB** total
- Main recording: `video.vrs` · MPS dir: `mps`

| data group | files | size | contents (from `sequence_config`) |
|---|---:|---:|---|
| `video_main_rgb` | 12 | 9.79 GB | preview RGB mp4 (browser preview, not sensor data) |
| `mps_slam_trajectories` | 12 | 0.40 GB | Machine Perception Services output (slam trajectories) |
| `mps_slam_calibration` | 12 | 0.01 GB | Machine Perception Services output (slam calibration) |
| `mps_slam_points` | 12 | 44.21 GB | Machine Perception Services output (slam points) |
| `mps_slam_summary` | 12 | 0.00 GB | Machine Perception Services output (slam summary) |
| `mps_hand_tracking` | 12 | 0.04 GB | Machine Perception Services output (hand tracking) |
| `mps_artifacts` | 12 | 44.65 GB | Machine Perception Services output (artifacts) |
| `main_vrs` | 12 | 34.87 GB | raw sensor recording `video.vrs` |
| `depth` | 12 | 24.91 GB | `depth/pinhole_camera_parameters.json`, `depth/depth`, `depth/rectified_images`, `depth/summary.json` |
| `scene` | 12 | 0.02 GB | `scene/2d_bounding_box.csv`, `scene/3d_bounding_box.csv`, `scene/instances.json`, `scene/scene_objects.csv`, `scene/summary.json` |
| `heart_rate` | 12 | 0.00 GB | `heart_rate/heart_rate_results.csv`, `heart_rate/summary.json` |
| `diarization` | 12 | 0.00 GB | `diarization/diarization_results.csv`, `diarization/summary.json` |
| `hand_object_interaction` | 12 | 0.02 GB | `hand_object_interaction/hand_object_interaction_results.json`, `hand_object_interaction/summary.json` |

<details><summary>filterable metadata fields</summary>

`sequence_uid`, `Scene`, `Is_multi_person`, `device_time_alignment_host`, `device_time_alignment_client`, `device_serial`, `duration_s`, `computed.light_intensity_lux_median`, `computed.trajectory_length_m`, `computed.covered_area_m2`, `computed.covered_volume_m3`, `computed.speed_mps_mean`

</details>

### `objects` — DTC Object Explorer (Digital Twin Catalog object library)

Separate site at https://dtc.projectaria.com/ (same SPA in object-viewer mode). Docs: https://www.projectaria.com/datasets/dtc/ · License: https://www.projectaria.com/datasets/dtc/license · links expire 2026-09-24

| release | objects | files | size |
|---|---:|---:|---:|
| DTC | 1,999 | 9,973 | 146.5 GB |
| ADT | 400 | 1,200 | 5.5 GB |

| per-object file group | files | size | example |
|---|---:|---:|---|
| `3d-asset_glb` | 2,399 | 118.0 GB | `DTC_1_1_Bowl_B0C7MFSFF7_WhiteWithThreeOvals_TU_3d-asset.glb` |
| `3d-asset-manifold_glb` | 1,988 | 33.2 GB | `DTC_1_1_Bowl_B0C7MFSFF7_WhiteWithThreeOvals_TU_3d-asset-manifold.glb` |
| `3d-asset-manifold-10k_glb` | 1,988 | 0.7 GB | `DTC_1_1_Bowl_B0C7MFSFF7_WhiteWithThreeOvals_TU_3d-asset-manifold-10k.glb` |
| `license` | 2,399 | 0.1 GB | `DTC_1_1_Bowl_B0C7MFSFF7_WhiteWithThreeOvals_TU_CC_BY-SA.txt` |
| `metadata` | 2,399 | 0.0 GB | `DTC_1_1_Bowl_B0C7MFSFF7_WhiteWithThreeOvals_TU_metadata.json` |

Object licenses: CC_BY-SA ×2399. Top categories: (uncategorised — the 400 ADT-release objects carry no `category`) (400), building blocks (128), shoes (121), bowl (106), dino (103), birdhouse (102), vase (101), teapot (99), fakefruit (96), figurine (77), fakefoodcan (69), basketball (55).
Metadata fields: `object_uid`, `release`, `category`, `license`, `avg_metallic`, `avg_roughness`, `texture`, `num_vertices`, `num_triangles`.

## Project Aria datasets that are NOT in the explorer

These still use the older per-dataset email form on projectaria.com (which e-mails/serves a `*_download_urls.json`) or a different host entirely:

| dataset | where | notes |
|---|---|---|
| Aria Synthetic Environments (ASE) | https://www.projectaria.com/datasets/ase/ | 100K procedurally generated scenes, ~2.5 TB. Covered by `../download_aria.sh` (email-gated manifest + `aria_synthetic_environments_downloader.py`). |
| EFM3D benchmark | https://www.projectaria.com/research/efm3D/ | ASE/AEO subsets + eval meshes; covered by `../download_efm3d.sh`. |
| Ego-Exo4D | https://ego-exo4d-data.org/ | Aria-captured but hosted by the Ego4D consortium (license form → `egoexo` CLI, AWS creds). |
| HOT3D-Clips (BOP subset) | https://huggingface.co/datasets/bop-benchmark/hot3d | Ungated ~80 GB clip subset of HOT3D; full VRS sets are the explorer's `hot3d-aria` / `hot3d-quest`. |
| Aria Pilot Dataset (2022) | superseded | Re-released as `aea` (Aria Everyday Activities) in the explorer. |
| Aria Gen 2 docs / other Gen 2 releases | https://facebookresearch.github.io/projectaria_tools/gen2/ | Only `gen2pilot` is in the explorer so far. |

## Scripts in this folder

Every explorer dataset has a thin `download_<slug>.sh` (sets `SLUG`, `MANIFEST_NAME`,
`DEFAULT_GROUPS`, sources `aria_common.sh`); the work is done by `aria_explorer_dl.py`
(stdlib Python) which fetches the manifest from the API above (no email), plans only the
files still missing/partial, and hands them to **one aria2c run** (`-c` resume, 8-way
segmented, `checksum=sha-1=` from the manifest, `--check-integrity`). Files finished in a
pass are re-hashed and quarantined as `*.corrupt-<ts>` on mismatch, so a full-size file on
disk is always a verified one. Passes are idempotent; expired links (exit 3) trigger a
manifest re-fetch automatically.

| script | data dir (`$DOWNLOAD_ROOT/aria/…`) | default groups (≈ size) |
|---|---|---|
| `download_adt.sh` | `adt` | main_vrs, main_groundtruth, depth, segmentation, synthetic, mps_slam_* (2.2 TB) |
| `download_nymeria_plus.sh` | `nymeria_plus` | head + observer `data.vrs`, slam, object_bounding_box, object_mesh, metadata, LICENSE (37.7 TB) |
| `download_nymeria.sh` | `nymeria` | v0.0 head + observer `data.vrs` + their mps zips (34.7 TB) — superseded by Plus, **not in a lane** |
| `download_hot3d_aria.sh` | `hot3d-aria` | main_vrs, ground_truth, hand_data, mps_slam_* (420 GB) |
| `download_hot3d_quest.sh` | `hot3d-quest` | main_vrs, ground_truth, hand_data (387 GB) |
| `download_dtc.sh` | `dtc` | main_vrs, main_groundtruth, mps_slam_* (278 GB) |
| `download_dtc_objects.sh` | `dtc_objects` | 3d-asset_glb, metadata, license — from dtc.projectaria.com (118 GB) |
| `download_aeo.sh` | `aeo` | main_vrs, mps, main_groundtruth (16 GB) |
| `download_aea.sh` | `aea` | main_vrs, annotations, mps_slam_* (343 GB) |
| `download_ritw.sh` | `ritw` | recording_vrs, metadata_json, mps (6.6 TB) |
| `download_aria_scenes.sh` | `aria-scenes` | main_vrs, mps_slam_* (112 GB) |
| `download_gen2pilot.sh` | `gen2pilot` | main_vrs, depth, scene, mps_slam_*, mps_hand_tracking + tiny annotations (105 GB) |
| `download_hot3d.sh` | `hot3d-aria`, `hot3d-quest`, `hot3d` | umbrella: the two above, plus the email-gated object models if `Hot3DAssets_download_urls.json` is present |
| `download_ase.sh` | `ase` | ASE — **not in the explorer**, email-gated manifest + official downloader (was `gated/download_aria.sh`) |
| `download_efm3d.sh` | `efm3d` | EFM3D — **not in the explorer**, email-gated manifest + repo scripts |

Defaults are the groups a 3D-reconstruction pipeline consumes (raw VRS, MPS SLAM
trajectories/calibration/points, depth/segmentation/GT/meshes). Deliberately skipped by
default: `video_main_rgb` (browser preview mp4s), `mps_artifacts` (byte-for-byte the
`mps_*` zips re-bundled), eye gaze / audio / body motion / IMU-only / narration, the wrist
recordings, and `semidense_observations` (11.6 TB per Nymeria release). `DATA_GROUPS=all`
takes everything; `EXCLUDE_GROUPS=…` trims; `SEQUENCES=uid1,uid2` or `SEQUENCES=@file`
limits sequences.

```bash
./download_adt.sh                              # -> $DOWNLOAD_ROOT/aria/adt  (default groups)
DATA_GROUPS=all ./download_aeo.sh /my/aeo      # everything, explicit dir
SEQUENCES=sunroom DATA_GROUPS=main_vrs ./download_aria_scenes.sh
STATUS_ONLY=1 ./download_ritw.sh               # completion summary, no download
DRY_RUN=1 MAX_FILES=5 SMALL_FIRST=1 ./download_dtc.sh
VERIFY=1 ./download_aea.sh                     # re-hash everything on disk, quarantine + re-fetch bad files
```

Transport: **no proxy, by decision.** `ARIA_TRANSPORT=direct` (default) probes a no-proxy
ranged GET on the CDN before every pass and, if that fails, waits `ARIA_WAIT_DIRECT` (300 s)
and re-probes — it never falls back to `$https_proxy` (the manifest is fetched `--no-proxy` as
well). Measured 2026-08-26 from this host: the explorer API answers directly, but
`scontent.xx.fbcdn.net` never completes a TLS handshake without the proxy (IPv4; no IPv6
here; TLS 1.2-only, HTTP/1.1 and plain HTTP all stall too), so lanes currently sit in the
wait/re-probe loop until the route opens. `ARIA_TRANSPORT=auto` (direct, else proxy) and
`ARIA_TRANSPORT=proxy` exist only as explicit opt-ins.
Tuning: `WORKERS` (files at once, default 4; the launcher uses 2 per lane), `ARIA2_CONNS`
(8), `ARIA2_SPEED_FLOOR` (100K, reaps stalled connections; the next pass resumes),
`ARIA_MAX_RATE` (e.g. `20M`), `ARIA_PASSES`/`ARIA_PASS_SLEEP` (20 / 60 s).

### Launcher

`./launch_all.sh start|stop|status|logs [lane…]` runs four detached, size-ordered lanes
(`small` → aeo, aria_scenes, gen2pilot, dtc_objects, dtc, aea, hot3d_quest, hot3d_aria;
`adt`; `ritw`; `nymeria` → nymeria_plus), each re-running its scripts for `MAX_PASSES`
passes (default `0` = unlimited, so a lane keeps waiting for direct access) with `PASS_SLEEP=600` between. Pidfiles and logs live in
`$DOWNLOAD_ROOT/aria/.launch/`. `status --brief` prints one done/total line per dataset.

## Reproduce this crawl

```bash
S=/some/scratch; wget -qO $S/versions.json https://explorer.projectaria.com/versions
for s in $(python3 -c "import json;print(' '.join(v['url_name'] for v in json.load(open('$S/versions.json'))['versions']))"); do
  for ep in version_config/$s data/$s data/$s/previews data/$s/download_links; do
    wget -qO "$S/$(echo $ep | tr / _).json" "https://explorer.projectaria.com/$ep"; done; done
wget -qO $S/dtc_objects_download_links.json https://dtc.projectaria.com/data/objects/download_links
```

The signed-URL manifests are ~85 MB in total and expire, so they are kept out of git; only this summary and `versions.json` live here.
