import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.audit_576a_unified_increment_panel import (  # noqa: E402
    UnifiedIncrementBlock,
    UnifiedVariableSpec,
    decode_state,
)
from experiments.backfill.block_ar.train_609a_unified_ar_transition_flow import select_scope  # noqa: E402


def _toy_block() -> UnifiedIncrementBlock:
    specs = [
        UnifiedVariableSpec(name="iv:00", source_column="iv:00", source_index=0, transform="log_level"),
        UnifiedVariableSpec(name="factor:signed", source_column="factor:signed", source_index=1, transform="diff_level"),
    ]
    history = np.array(
        [
            [[0.20, -1.0], [0.25, -0.5]],
            [[0.30, 0.5], [0.35, 1.0]],
        ],
        dtype=np.float32,
    )
    future = np.array(
        [
            [[0.22, -0.8], [0.27, -0.4]],
            [[0.32, 0.7], [0.38, 1.2]],
        ],
        dtype=np.float32,
    )
    return UnifiedIncrementBlock(
        history_state=history,
        future_state=future,
        future_increment=np.zeros_like(future),
        reconstructed_future_state=future.copy(),
        indices=np.array([0, 1], dtype=np.int64),
        specs=specs,
    )


def test_select_scope_encoded_uses_log_level_and_decodes_back() -> None:
    block = _toy_block()

    history, future, specs = select_scope(block, "joint38", iv_count=1, value_coordinate="encoded")

    assert np.allclose(history[..., 0], np.log(block.history_state[..., 0]))
    assert np.allclose(future[..., 0], np.log(block.future_state[..., 0]))
    assert np.allclose(history[..., 1], block.history_state[..., 1])
    assert np.allclose(decode_state(history, specs), block.history_state)
    assert np.allclose(decode_state(future, specs), block.future_state)


def test_select_scope_encoded_iv_only_keeps_decodable_iv_channel() -> None:
    block = _toy_block()

    history, future, specs = select_scope(block, "iv_only", iv_count=1, value_coordinate="encoded")

    assert history.shape == (2, 2, 1)
    assert future.shape == (2, 2, 1)
    assert [spec.name for spec in specs] == ["iv:00"]
    assert np.allclose(decode_state(history, specs)[..., 0], block.history_state[..., 0])


def test_select_scope_rejects_unknown_value_coordinate() -> None:
    with pytest.raises(ValueError, match="value_coordinate"):
        select_scope(_toy_block(), "joint38", iv_count=1, value_coordinate="bad")

