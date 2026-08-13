"""mip-NeRF-360 sequence: a ColmapSequence with image-pyramid resolution selection.

mip-NeRF-360 scenes are COLMAP sparse reconstructions shipped with an image
pyramid (``images``, ``images_2``, ``images_4``, ``images_8``). This vendor adds
one thing on top of :class:`ColmapSequence`: ``scale_factor`` picks the on-disk
pyramid level (the parent then auto-rescales intrinsics to that resolution), so
callers choose the working resolution by a single integer instead of naming an
``images_*`` directory.

**No depth.** The dataset is photogrammetry -- there is no depth sensor and no
native depth map, so this vendor advertises no ``Modality.DEPTH`` and mono-depth
eval self-skips for it. (It once synthesized one by rasterizing the COLMAP sparse
cloud into each frame; that was a *derived* proxy, not ground truth -- it covered
only the handful of SfM points per frame, leaving most pixels zero, and scoring
predicted depth against it measured agreement with SfM triangulation rather than
with the scene. Camera pose, which the dataset does supply, remains scored.)
"""
from __future__ import annotations

from typing import Optional, Set, Union

from vggt_omega.datasets.sequences.base_sequence import Modality
from vggt_omega.datasets.sequences.colmap import ColmapSequence


class MipNerf360Sequence(ColmapSequence):
    """A mip-NeRF-360 scene: :class:`ColmapSequence` + pyramid resolution selection."""

    # On-disk image-pyramid levels shipped with the mip-NeRF-360 release.
    _SCALE_FACTORS = (1, 2, 4, 8)

    def __init__(
        self,
        data_root: str,
        seq_id: str,
        cache_dir: Optional[str] = None,
        sparse_dir: str = "sparse/0",
        scale_factor: int = 1,
    ):
        """Open a mip-NeRF-360 scene at a chosen resolution.

        Args:
            data_root: directory containing the scenes.
            seq_id: scene sub-directory (e.g. ``"garden"``).
            cache_dir: cache directory (unused; kept for the base signature).
            sparse_dir: COLMAP model sub-dir (``{cameras,images,points3D}.bin``).
            scale_factor: image-pyramid level in ``{1, 2, 4, 8}``; selects
                ``images`` (1) or ``images_{scale_factor}``. Intrinsics are
                auto-rescaled to that resolution by :class:`ColmapSequence`.
        """
        if scale_factor not in self._SCALE_FACTORS:
            raise ValueError(
                f"scale_factor must be one of {self._SCALE_FACTORS}, got {scale_factor!r}"
            )
        self._scale_factor = int(scale_factor)
        images_dir = "images" if scale_factor == 1 else f"images_{scale_factor}"
        super().__init__(
            data_root, seq_id, cache_dir, sparse_dir=sparse_dir, images_dir=images_dir
        )

    # -- modalities ---------------------------------------------------------- #
    def get_modalities(self, sensor_id: Union[int, str]) -> Set[Modality]:
        """What a mip-NeRF-360 scene actually ships, stated explicitly.

        Spelled out rather than inherited so this vendor's ground truth is readable
        in one place and cannot drift with the parent: a COLMAP reconstruction gives
        the images, the per-frame camera pose and its intrinsics, and the sparse
        point cloud. Notably absent is ``DEPTH`` -- the dataset is photogrammetry,
        with no depth sensor and no native depth map -- so mono-depth eval self-skips.
        """
        return {
            Modality.RGB,
            Modality.POSE,
            Modality.INTRINSIC,
            Modality.EXTRINSIC,
            Modality.POINTCLOUD,
        }
