import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_window_selection_manifest import (
    build_window_records,
    select_manifest_windows,
    split_selected_records,
)


def _spec_names() -> list[str]:
    return [f"iv:{idx:02d}" for idx in range(25)] + [
        "factor:spx",
        "factor:usdcad",
        "factor:usdjpy",
        "factor:dxy",
        "factor:copper",
        "factor:wheat",
        "factor:crude_oil",
        "factor:us2y",
        "factor:us10y",
        "factor:aaa_oas",
        "factor:bbb_oas",
        "factor:nikkei",
        "factor:gold",
        "factor:vix",
    ]


def _history(delta_spx: float, delta_vix: float, delta_oas: float) -> np.ndarray:
    history = np.zeros((30, 39), dtype=np.float32)
    history[:, 25] = np.linspace(0.0, delta_spx, 30)
    history[:, 35] = np.linspace(0.0, delta_oas, 30)
    history[:, 38] = np.linspace(0.0, delta_vix, 30)
    return history


def _metadata(n: int) -> list[dict[str, object]]:
    return [
        {
            "source_index": idx,
            "calendar_start_date": f"2020-01-{idx + 1:02d}",
            "calendar_end_date": f"2020-02-{idx + 1:02d}",
            "forecast_start_date": f"2020-02-{idx + 2:02d}",
            "forecast_end_date": f"2020-03-{idx + 2:02d}",
        }
        for idx in range(n)
    ]


def test_build_window_records_scores_stress_above_calm() -> None:
    histories = np.stack(
        [
            _history(-1.2, 1.3, 0.9),
            _history(0.03, 0.02, 0.01),
        ],
        axis=0,
    )
    records = build_window_records(histories, _spec_names(), _metadata(2))

    assert records[0]["salience_score"] > records[1]["salience_score"]
    assert "equity" in records[0]["tags"]
    assert "vol" in records[0]["tags"]


def test_select_manifest_windows_includes_weekly_eventful_and_calm_buckets() -> None:
    histories = np.stack(
        [
            _history(-1.4, 1.1, 0.8),
            _history(-0.9, 1.0, 0.7),
            _history(0.04, 0.02, 0.01),
            _history(0.03, 0.01, 0.01),
            _history(0.8, -0.7, -0.5),
            _history(0.7, -0.6, -0.4),
        ],
        axis=0,
    )
    records = build_window_records(histories, _spec_names(), _metadata(6))
    selected = select_manifest_windows(
        records,
        weekly_count=2,
        eventful_count=2,
        calm_count=2,
        min_gap=0,
        diversity_threshold=0.95,
    )

    reasons_by_index = {row["window_index"]: set(row["selection_reasons"]) for row in selected}
    all_reasons = set().union(*(row["selection_reasons"] for row in selected))
    assert {"weekly_anchor", "eventful", "calm"} <= all_reasons
    assert 0 in reasons_by_index
    assert "eventful" in reasons_by_index[0]
    assert any("calm" in reasons for reasons in reasons_by_index.values())


def test_split_selected_records_applies_temporal_embargo() -> None:
    records = [
        {
            "window_index": idx,
            "window_id": f"joint39_val_{idx:04d}",
            "selection_reasons": ["weekly_anchor"],
        }
        for idx in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    ]
    split = split_selected_records(
        records,
        train_fraction=0.6,
        validation_fraction=0.2,
        embargo=5,
    )

    assert [row["window_index"] for row in split["train"]] == [0, 5, 10, 15, 20]
    assert [row["window_index"] for row in split["validation"]] == [30]
    assert [row["window_index"] for row in split["test"]] == [40, 45]
    assert [row["window_index"] for row in split["excluded_embargo"]] == [25, 35]
