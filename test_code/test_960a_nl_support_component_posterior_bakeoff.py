import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_support_component_posterior_bakeoff import (
    _select_components,
)


def _components() -> list[dict]:
    return [
        {"component_no": 0, "window_index": 10, "weight": 0.55, "sample_slice": [0, 4]},
        {"component_no": 1, "window_index": 20, "weight": 0.25, "sample_slice": [4, 6]},
        {"component_no": 2, "window_index": 30, "weight": 0.15, "sample_slice": [6, 7]},
        {"component_no": 3, "window_index": 40, "weight": 0.05, "sample_slice": [7, 8]},
    ]


def test_top2_80_posterior_selects_until_weight_threshold() -> None:
    selected = _select_components(_components(), posterior_mode="top2_80")

    assert [row["window_index"] for row in selected] == [10, 20]
    assert abs(sum(row["sparse_weight"] for row in selected) - 1.0) < 1e-9


def test_top1_posterior_keeps_one_component() -> None:
    selected = _select_components(_components(), posterior_mode="top1")

    assert [row["window_index"] for row in selected] == [10]
    assert selected[0]["sparse_weight"] == 1.0


def test_full_posterior_keeps_all_components_and_normalizes_weights() -> None:
    selected = _select_components(_components(), posterior_mode="full")

    assert [row["window_index"] for row in selected] == [10, 20, 30, 40]
    assert abs(sum(row["sparse_weight"] for row in selected) - 1.0) < 1e-9
