"""
Dataset handler for 1D time series with variable sequence length.

Adapted from datasets_randomized.py for scalar time series instead of 2D surfaces.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, BatchSampler, Sampler
from typing import Iterator
from collections import defaultdict
import random


class GroupedRandomSampler1D(Sampler):
    """
    Groups dataset samples by sequence length and shuffles within groups.

    Ensures batches contain sequences of the same length for efficient processing.
    """

    def __init__(self, dataset, min_seq_len=4):
        self.dataset = dataset
        self.groups = defaultdict(list)

        for idx in range(len(dataset)):
            seq_len = dataset[idx]["target"].shape[0]
            if seq_len < min_seq_len:
                continue
            self.groups[seq_len].append(idx)

    def __iter__(self) -> Iterator:
        group_idx = list(self.groups.values())
        random.shuffle(group_idx)
        for group in group_idx:
            random.shuffle(group)
            yield from group

    def __len__(self):
        return len(self.dataset)


class CustomBatchSampler1D(BatchSampler):
    """
    Custom batch sampler that ensures all sequences in a batch have the same length.

    This is required because LSTM expects fixed-length sequences within each batch.
    Variable lengths are handled across batches, not within a single batch.
    """

    def __init__(self, dataset, batch_size: int, min_seq_len=4) -> None:
        super().__init__(GroupedRandomSampler1D(dataset, min_seq_len), batch_size, drop_last=False)
        self.dataset = dataset

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            if not batch:
                batch.append(idx)
            else:
                seq_len = self.dataset[batch[0]]["target"].shape[0]
                if self.dataset[idx]["target"].shape[0] == seq_len:
                    batch.append(idx)
                else:
                    yield batch
                    batch = [idx]
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


class TimeSeriesDataSetRand(Dataset):
    """
    Dataset for 1D time series with variable sequence length.

    Supports:
    - Target-only (e.g., Amazon returns only)
    - Target + conditioning features (e.g., Amazon + SP500/MSFT)

    Each sample is a random-length subsequence drawn from the full time series.
    This provides data augmentation and tests model robustness to different context windows.

    Args:
        dataset: Either:
            - Single numpy array (N,) for target-only
            - Tuple (target, cond_feats) where:
                - target: (N,) numpy array (e.g., Amazon returns)
                - cond_feats: (N, K) numpy array (e.g., SP500, MSFT returns)
        min_seq_len: Minimum sequence length (default: 4)
        max_seq_len: Maximum sequence length (default: 10)
        dtype: torch dtype (default: torch.float64)

    Returns:
        Dictionary with keys:
            - "target": (T, 1) - Target time series
            - "cond_feats": (T, K) - Conditioning features (if provided)

        where T ∈ [min_seq_len, max_seq_len]
    """

    def __init__(self, dataset, min_seq_len=4, max_seq_len=10, dtype=torch.float64):
        if isinstance(dataset, tuple):
            # Target + conditioning features
            target_data = dataset[0]
            cond_data = dataset[1]

            # Convert to torch tensors
            self.target = torch.from_numpy(target_data)
            self.cond_feats = torch.from_numpy(cond_data)

            # Cast to desired dtype
            if dtype == torch.float32:
                self.target = self.target.float()
                self.cond_feats = self.cond_feats.float()

            # Verify matching lengths
            assert len(self.target) == len(self.cond_feats), \
                "target and cond_feats should have the same length"

            # Ensure target is (N, 1) and cond_feats is (N, K)
            if len(self.target.shape) == 1:
                self.target = self.target.unsqueeze(-1)

            if len(self.cond_feats.shape) == 1:
                self.cond_feats = self.cond_feats.unsqueeze(-1)

        else:
            # Target only (no conditioning)
            self.target = torch.from_numpy(dataset)
            self.cond_feats = None

            # Cast to desired dtype
            if dtype == torch.float32:
                self.target = self.target.float()

            # Ensure target is (N, 1)
            if len(self.target.shape) == 1:
                self.target = self.target.unsqueeze(-1)

        # Create list of all possible sequence lengths
        self.seq_lens = list(range(min_seq_len, max_seq_len + 1))

    def __len__(self):
        """
        Total number of possible samples.

        For each starting position in the dataset, we can sample multiple
        sequence lengths, so total = N_positions × N_sequence_lengths.
        """
        return len(self.target) * len(self.seq_lens)

    def __getitem__(self, idx):
        """
        Get a single sample (subsequence of the time series).

        The idx is decomposed into:
        - seq_len_idx: which sequence length to use
        - seq_start_idx: where to start the subsequence

        Returns:
            Dictionary with "target" and optionally "cond_feats"
        """
        seq_len_idx, seq_start_idx = divmod(idx, len(self.target))
        seq_len = self.seq_lens[seq_len_idx]

        # Extract target subsequence
        target_seq = self.target[seq_start_idx:seq_start_idx + seq_len]

        if self.cond_feats is not None:
            # Extract conditioning features subsequence
            cond_seq = self.cond_feats[seq_start_idx:seq_start_idx + seq_len]
            return {"target": target_seq, "cond_feats": cond_seq}

        return {"target": target_seq}
