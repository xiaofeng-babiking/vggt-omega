# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC

from hydra.utils import instantiate
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import bisect
from .dataset_util import *
from .track_util import *
from .augmentation import get_image_augmentation
from .modality import carry_extra_modalities


class ComposedDataset(Dataset, ABC):
    """
    Composes multiple base datasets and applies common configurations.

    This dataset provides a flexible way to combine multiple base datasets while
    applying shared augmentations, track generation, and other processing steps.
    It handles image normalization, tensor conversion, and other preparations
    needed for training computer vision models with sequences of images.
    """
    def __init__(self, dataset_configs: dict, common_config: dict, **kwargs):
        """
        Initializes the ComposedDataset.

        Args:
            dataset_configs (dict): List of Hydra configurations for base datasets.
            common_config (dict): Shared configurations (augs, tracks, mode, etc.).
            **kwargs: Additional arguments (unused).
        """
        base_dataset_list = []

        # Instantiate each base dataset with common configuration
        for baseset_dict in dataset_configs:
            baseset = instantiate(baseset_dict, common_conf=common_config)
            base_dataset_list.append(baseset)

        # Use custom concatenation class that supports tuple indexing
        self.base_dataset = TupleConcatDataset(base_dataset_list, common_config)

        # --- Augmentation Settings ---
        # Controls whether to apply identical color jittering across all frames in a sequence
        self.cojitter = common_config.augs.cojitter
        # Probability of using shared jitter vs. frame-specific jitter
        self.cojitter_ratio = common_config.augs.cojitter_ratio
        # Initialize image augmentations (color jitter, grayscale, gaussian blur)
        self.image_aug = get_image_augmentation(
            color_jitter=common_config.augs.color_jitter,
            gray_scale=common_config.augs.gray_scale,
            gau_blur=common_config.augs.gau_blur,
        )

        # --- Optional Fixed Settings (useful for debugging) ---
        # Force each sequence to have exactly this many images (if > 0)
        self.fixed_num_images = common_config.fix_img_num
        # Force a specific aspect ratio for all images
        self.fixed_aspect_ratio = common_config.fix_aspect_ratio

        # --- Track Settings ---
        # Whether to include point tracks in the output
        self.load_track = common_config.load_track
        # Number of point tracks to include per sequence
        self.track_num = common_config.track_num
        # Fraction of on-the-fly tracks that should be negative pairs
        self.track_neg_ratio = common_config.get("track_neg_ratio", 0.5)

        # --- Mode Settings ---
        # Whether the dataset is being used for training (affects augmentations)
        self.training = common_config.training
        self.common_config = common_config

        self.total_samples = len(self.base_dataset)

        # Flat index over the vendors' real sequences (drives get_sample / the
        # deterministic inference path). Cheap: sequence_list is already built.
        self.sequence_index = self._build_sequence_index()

    def __len__(self):
        """Returns the total number of sequences in the dataset."""
        return self.total_samples


    def __getitem__(self, idx_tuple):
        """
        Retrieves a data sample (sequence) from the dataset.

        Loads raw data, converts to PyTorch tensors, applies augmentations,
        and prepares tracks if enabled.

        Args:
            idx_tuple (tuple): a tuple of (seq_idx, num_images, aspect_ratio)

        Returns:
            dict: A dictionary containing the sequence data (images, poses, tracks, etc.).
        """
        # If fixed settings are provided, override the tuple values
        if self.fixed_num_images > 0:
            seq_idx = idx_tuple[0] if isinstance(idx_tuple, tuple) else idx_tuple
            idx_tuple = (seq_idx, self.fixed_num_images, self.fixed_aspect_ratio)

        # Retrieve the raw data batch from the appropriate base dataset
        batch = self.base_dataset[idx_tuple]
        return self._tensorize(batch)

    def _tensorize(self, batch):
        """Convert a raw numpy batch (from a vendor's ``get_data``) into the
        training tensor sample dict: images normalized to ``(V,3,H,W)`` in
        ``[0,1]``, every other modality stacked + tensorized, color augmentation
        applied only when ``self.training``.

        Shared by ``__getitem__`` (random-sampler path) and ``get_sample``
        (explicit-ids eval/inference path), so the two paths can never diverge.
        """
        seq_name = batch["seq_name"]

        # --- Data Conversion and Preparation ---
        # Convert numpy arrays to tensors
        images = torch.from_numpy(np.stack(batch["images"]).astype(np.float32)).contiguous()
        # Normalize images from [0, 255] to [0, 1]
        images = images.permute(0,3,1,2).to(torch.get_default_dtype()).div(255)

        # Convert other data to tensors with appropriate types
        depths = torch.from_numpy(np.stack(batch["depths"]).astype(np.float32))
        extrinsics = torch.from_numpy(np.stack(batch["extrinsics"]).astype(np.float32))
        intrinsics = torch.from_numpy(np.stack(batch["intrinsics"]).astype(np.float32))
        cam_points = torch.from_numpy(np.stack(batch["cam_points"]).astype(np.float32))
        world_points = torch.from_numpy(np.stack(batch["world_points"]).astype(np.float32))
        point_masks = torch.from_numpy(np.stack(batch["point_masks"])) # Mask indicating valid depths / world points / cam points per frame
        ids = torch.from_numpy(batch["ids"])    # Frame indices sampled from the original sequence


        # --- Apply Color Augmentation (training mode only) ---
        if self.training and self.image_aug is not None:
            if self.cojitter and random.random() > self.cojitter_ratio:
                # Apply the same color jittering transformation to all frames
                images = self.image_aug(images)
            else:
                # Apply different color jittering to each frame individually
                for aug_img_idx in range(len(images)):
                    images[aug_img_idx] = self.image_aug(images[aug_img_idx])


        # --- Prepare Final Sample Dictionary ---
        sample = {
            "seq_name": seq_name,
            "ids": ids,
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
        }

        # --- Track Processing (if enabled) ---
        if self.load_track:
            if batch.get("tracks") is not None:
                # Use pre-computed tracks from the dataset
                tracks = torch.from_numpy(np.stack(batch["tracks"]).astype(np.float32))
                track_vis_mask = torch.from_numpy(np.stack(batch["track_masks"]).astype(bool))

                # Sample a subset of tracks randomly
                valid_indices = torch.where(track_vis_mask[0])[0]
                if len(valid_indices) >= self.track_num:
                    # If we have enough tracks, sample without replacement
                    sampled_indices = valid_indices[torch.randperm(len(valid_indices))][:self.track_num]
                else:
                    # If not enough tracks, sample with replacement (allow duplicates)
                    sampled_indices = valid_indices[torch.randint(0, len(valid_indices),
                                                    (self.track_num,),
                                                    dtype=torch.int64,
                                                    device=valid_indices.device)]

                # Extract the sampled tracks and their masks
                tracks = tracks[:, sampled_indices, :]
                track_vis_mask = track_vis_mask[:, sampled_indices]
                track_positive_mask = torch.ones(track_vis_mask.shape[1]).bool()

            else:
                # Generate tracks on-the-fly using depth information
                # This creates synthetic tracks based on the 3D information available
                tracks, track_vis_mask, track_positive_mask = build_tracks_by_depth(
                    extrinsics, intrinsics, world_points, depths, point_masks, images,
                    target_track_num=self.track_num, neg_ratio=self.track_neg_ratio,
                    seq_name=seq_name
                )

            # Add track information to the sample dictionary
            sample["tracks"] = tracks
            sample["track_vis_mask"] = track_vis_mask
            sample["track_positive_mask"] = track_positive_mask

        sample = carry_extra_modalities(batch, sample)
        return sample

    # --- Explicit-ids eval/inference path -----------------------------------
    # The training DataLoader (DynamicBatchSampler) injects random image counts,
    # aspect ratios, sequences and frame ids -- incompatible with deterministic,
    # ordered, sharded evaluation. These helpers reuse the SAME tensorization
    # (_tensorize) for explicit sequences/ids, so inference stays training-identical.

    def _build_sequence_index(self):
        """Flat map: global sequence index -> ``(vendor, local_idx, seq_name)``,
        built over the vendors' REAL ``sequence_list``. (``ConcatDataset``'s
        ``cumulative_sizes`` use the virtual ``len_train``, so they cannot
        enumerate real sequences.)"""
        index = []
        for vendor in self.base_dataset.datasets:
            seq_list = getattr(vendor, "sequence_list", None)
            if seq_list is None:
                raise AttributeError(
                    f"{type(vendor).__name__} exposes no `sequence_list`; "
                    "ComposedDataset sequence enumeration requires one per vendor."
                )
            for local_idx, name in enumerate(seq_list):
                index.append((vendor, local_idx, name))
        return index

    def num_sequences(self):
        """Total number of real sequences across all composed vendors."""
        return len(self.sequence_index)

    def sequence_name(self, seq_index):
        """Raw sequence name (the vendor's ``sequence_list`` entry) for a global index."""
        return self.sequence_index[seq_index][2]

    def sequence_num_frames(self, seq_index):
        """Number of available frames in the sequence at a global sequence index."""
        vendor, local_idx, _ = self.sequence_index[seq_index]
        return vendor.sequence_num_frames(local_idx)

    def get_sample(self, seq_index, ids, aspect_ratio=1.0):
        """Tensorized sample for EXPLICIT, ordered frame ``ids`` of one sequence --
        the training-identical inference path. Reuses ``_tensorize`` verbatim.
        Passing ``seq_name`` (not ``seq_index``) to the vendor sidesteps its
        ``inside_random`` remap, so the result is deterministic regardless of config.
        """
        vendor, _, name = self.sequence_index[seq_index]
        batch = vendor.get_data(
            seq_name=name, ids=np.asarray(ids), aspect_ratio=aspect_ratio
        )
        return self._tensorize(batch)

    def native_image_size(self, seq_index=0):
        """Native ``(H, W)`` of the source frames for a global sequence index --
        lets the eval/inference resolution be read from the data, not hardcoded."""
        vendor, local_idx, _ = self.sequence_index[seq_index]
        return vendor.native_image_size(local_idx)

    def set_img_size(self, img_size):
        """Override the target long-side resolution on every vendor (e.g. to
        evaluate at the data's native resolution). Affects get_data / get_sample."""
        for vendor in self.base_dataset.datasets:
            vendor.img_size = int(img_size)

    @property
    def img_size(self):
        """Current target long-side resolution (uniform across vendors)."""
        return self.base_dataset.datasets[0].img_size


class TupleConcatDataset(ConcatDataset):
    """
    A custom ConcatDataset that supports indexing with a tuple.

    Standard PyTorch ConcatDataset only accepts an integer index. This class extends
    that functionality to allow passing a tuple like (sample_idx, num_images, aspect_ratio),
    where the first element is used to determine which sample to fetch, and the full
    tuple is passed down to the selected dataset's __getitem__ method.

    It also supports an option to randomly sample across all datasets, ignoring the
    provided index. This is useful during training when shuffling the entire dataset
    might cause memory issues due to duplicating dictionaries. If doing this, you can
    set pytorch's dataloader shuffle to False.
    """
    def __init__(self, datasets, common_config):
        """
        Initialize the TupleConcatDataset.

        Args:
            datasets (iterable): An iterable of PyTorch Dataset objects to concatenate.
            common_config (dict): Common configuration dict, used to check for random sampling.
        """
        super().__init__(datasets)
        # If True, ignores the input index and samples randomly across all datasets
        # This provides an alternative to dataloader shuffling for large datasets
        self.inside_random = common_config.inside_random

    def __getitem__(self, idx):
        """
        Retrieves an item using either an integer index or a tuple index.

        Args:
            idx (int or tuple): The index. If tuple, the first element is the sequence
                               index across the concatenated datasets, and the rest are
                               passed down. If int, it's treated as the sequence index.

        Returns:
            The item returned by the underlying dataset's __getitem__ method.

        Raises:
            ValueError: If the index is out of range or the tuple doesn't have exactly 3 elements.
        """
        idx_tuple = None
        if isinstance(idx, tuple):
            idx_tuple = idx
            idx = idx_tuple[0]  # Extract the sequence index

        # Override index with random value if inside_random is enabled
        if self.inside_random:
            total_len = self.cumulative_sizes[-1]
            idx = random.randint(0, total_len - 1)

        # Handle negative indices
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        # Find which dataset the index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # Create the tuple to pass to the underlying dataset
        if len(idx_tuple) == 3:
            idx_tuple = (sample_idx,) + idx_tuple[1:]
        else:
            raise ValueError("Tuple index must have exactly three elements")

        # Pass the modified tuple to the appropriate dataset
        return self.datasets[dataset_idx][idx_tuple]
