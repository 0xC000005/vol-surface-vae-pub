import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.evaluate_588a_unified_flow_full11_bridge import (
    alignment_diagnostics,
)


def test_alignment_diagnostics_accepts_matching_iv_panel() -> None:
    history = np.arange(2 * 3 * 25, dtype=np.float32).reshape(2, 3, 25) / 100.0
    future = np.arange(2 * 4 * 25, dtype=np.float32).reshape(2, 4, 25) / 100.0
    val_block = SimpleNamespace(
        history_state=np.concatenate([history, np.ones((2, 3, 2), dtype=np.float32)], axis=-1),
        future_state=np.concatenate([future, np.ones((2, 4, 2), dtype=np.float32)], axis=-1),
    )
    batch = SimpleNamespace(
        history_01=torch.from_numpy(history.reshape(2, 3, 5, 5)),
        future_01=torch.from_numpy(future.reshape(2, 4, 5, 5)),
    )

    diag = alignment_diagnostics(val_block, batch, n_windows=2)

    assert diag["history_max_abs_error"] == 0.0
    assert diag["future_max_abs_error"] == 0.0
