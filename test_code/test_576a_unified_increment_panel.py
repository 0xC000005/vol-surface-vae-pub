import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.audit_576a_unified_increment_panel import (
    build_unified_increment_block,
    build_unified_variable_specs,
    clean_nonpositive_log_level_factors,
    summarize_unified_increment_block,
)


def _toy_panel() -> tuple[np.ndarray, list[str]]:
    iv = np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float32)
    spx = np.array([100.0, 110.0, 121.0, 133.1, 146.41], dtype=np.float32)
    rate = np.array([5.0, 5.5, 5.25, 5.75, 6.0], dtype=np.float32)
    spx_logret = np.zeros_like(spx)
    spx_logret[1:] = np.log(spx[1:] / spx[:-1])
    rate_diff = np.zeros_like(rate)
    rate_diff[1:] = rate[1:] - rate[:-1]
    panel = np.stack([iv, spx, rate, spx_logret, rate_diff], axis=1)
    columns = [
        "iv:00",
        "factor:spx",
        "factor:rate",
        "factor:spx_logret",
        "factor:rate_diff",
    ]
    return panel.astype(np.float32), columns


def test_build_unified_variable_specs_drops_duplicate_return_targets() -> None:
    _panel, columns = _toy_panel()

    specs = build_unified_variable_specs(columns, iv_count=1)

    assert [spec.name for spec in specs] == ["iv:00", "factor:spx", "factor:rate"]
    assert [spec.transform for spec in specs] == ["log_level", "log_level", "diff_level"]
    assert specs[1].reference_increment_column == "factor:spx_logret"
    assert specs[2].reference_increment_column == "factor:rate_diff"


def test_unified_increment_block_reconstructs_future_state_exactly() -> None:
    panel, columns = _toy_panel()

    block = build_unified_increment_block(
        panel,
        columns,
        indices=[0],
        history_len=2,
        future_len=3,
        iv_count=1,
    )

    assert block.history_state.shape == (1, 2, 3)
    assert block.future_state.shape == (1, 3, 3)
    assert block.future_increment.shape == (1, 3, 3)
    assert np.allclose(block.reconstructed_future_state, block.future_state, rtol=1e-6, atol=1e-6)
    assert np.allclose(block.future_increment[0, :, 1], panel[2:, 3], rtol=1e-6, atol=1e-6)
    assert np.allclose(block.future_increment[0, :, 2], panel[2:, 4], rtol=1e-6, atol=1e-6)


def test_summarize_unified_increment_block_reports_clean_target_contract() -> None:
    panel, columns = _toy_panel()
    block = build_unified_increment_block(
        panel,
        columns,
        indices=[0],
        history_len=2,
        future_len=3,
        iv_count=1,
    )

    summary = summarize_unified_increment_block(
        panel,
        block,
        columns=columns,
        history_len=2,
        future_len=3,
        iv_count=1,
    )

    assert summary["source_panel_channel_count"] == 5
    assert summary["target_state_channel_count"] == 3
    assert summary["iv_state_count"] == 1
    assert summary["factor_state_count"] == 2
    assert summary["no_duplicate_target_names"] is True
    assert summary["no_return_like_targets"] is True
    assert summary["reconstruction_max_abs_error"] < 1e-5
    assert summary["reference_increment_count"] == 2
    assert summary["reference_increment_max_abs_error"] < 1e-6


def test_clean_nonpositive_log_level_factors_fills_zeros_and_preserves_negative_fallback() -> None:
    panel = np.array(
        [
            [1.0, 0.0, 10.0, 0.0, 0.0],
            [1.1, 100.0, -2.0, 0.1, -12.0],
            [1.2, 110.0, 11.0, 0.095, 13.0],
        ],
        dtype=np.float32,
    )
    columns = [
        "iv:00",
        "factor:gold",
        "factor:crude_oil",
        "factor:gold_logret",
        "factor:crude_oil_logret",
    ]

    cleaned, report = clean_nonpositive_log_level_factors(panel, columns, iv_count=1)
    specs = build_unified_variable_specs(columns, panel=cleaned, iv_count=1)

    assert report["cleaned_columns"] == {"factor:gold": 1}
    assert report["diff_fallback_columns"] == ["factor:crude_oil"]
    assert cleaned[0, 1] == 100.0
    assert [spec.transform for spec in specs] == ["log_level", "log_level", "diff_level"]
