"""MCVD Multi-Task Masking for Block-AR Diffusion."""
from enum import IntEnum
from typing import Tuple

import torch


class MCVDTask(IntEnum):
    FORWARD = 0       # past visible, future masked
    BACKWARD = 1      # past masked, future visible
    INTERPOLATION = 2 # both visible
    UNCONDITIONAL = 3 # both masked


def sample_mcvd_masks(
    batch_size: int,
    p_mask: float = 0.5,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample independent past/future masks for MCVD multi-task training.

    Args:
        batch_size: B
        p_mask: probability of masking each source (0.5 -> ~25% each task)
        device: torch device

    Returns:
        mask_past: (B,) bool - True means mask past (use null embedding)
        mask_future: (B,) bool - True means mask future
    """
    mask_past = torch.rand(batch_size, device=device) < p_mask
    mask_future = torch.rand(batch_size, device=device) < p_mask
    return mask_past, mask_future


def get_task_types(
    mask_past: torch.Tensor,
    mask_future: torch.Tensor,
) -> torch.Tensor:
    """Determine MCVD task type per batch item from masks.

    Returns:
        tasks: (B,) int tensor of MCVDTask values
    """
    tasks = torch.zeros(mask_past.shape[0], dtype=torch.long, device=mask_past.device)
    # FORWARD: past visible (not masked), future masked
    tasks[(~mask_past) & mask_future] = MCVDTask.FORWARD
    # BACKWARD: past masked, future visible
    tasks[mask_past & (~mask_future)] = MCVDTask.BACKWARD
    # INTERPOLATION: both visible
    tasks[(~mask_past) & (~mask_future)] = MCVDTask.INTERPOLATION
    # UNCONDITIONAL: both masked
    tasks[mask_past & mask_future] = MCVDTask.UNCONDITIONAL
    return tasks
