import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_portfolio_response_label_seed_stability import (
    compare_label_seed_stability,
)


def _report(values_by_query):
    rows = []
    row_no = 0
    for base, values in values_by_query.items():
        for idx, value in enumerate(values, start=1):
            rows.append(
                {
                    "query_id": f"{base}__mixture_{idx:03d}__support",
                    "window_id": base,
                    "window_index": 100 + row_no,
                    "block_window_index": 200 + row_no,
                    "row_no": row_no,
                    "methods": {
                        "narrative_generator_topk": {
                            "portfolio_reliable_path_score_z": value
                        }
                    },
                }
            )
            row_no += 1
    return {"window_scores": rows}


def test_compare_label_seed_stability_detects_stable_rankings():
    first = _report({"a": [0.1, 0.2, 0.3], "b": [0.4, 0.6, 0.8]})
    second = _report({"a": [0.11, 0.21, 0.31], "b": [0.42, 0.62, 0.82]})

    report = compare_label_seed_stability(first, second)

    assert report["matched_rows"] == 6
    assert report["query_count"] == 2
    assert report["result_status"] == "seed_stable_labels"
    assert report["within_query"]["weighted_pairwise_accuracy"] == pytest.approx(1.0)
    assert report["within_query"]["top1_match_rate"] == pytest.approx(1.0)


def test_compare_label_seed_stability_detects_seed_sensitive_rankings():
    first = _report({"a": [0.1, 0.2, 0.3], "b": [0.4, 0.6, 0.8]})
    second = _report({"a": [0.3, 0.2, 0.1], "b": [0.8, 0.6, 0.4]})

    report = compare_label_seed_stability(first, second)

    assert report["result_status"] == "seed_sensitive_labels"
    assert report["within_query"]["weighted_pairwise_accuracy"] == pytest.approx(0.0)
    assert report["within_query"]["top1_match_rate"] == pytest.approx(0.0)


def test_compare_label_seed_stability_requires_matches():
    first = _report({"a": [0.1]})
    second = _report({"b": [0.1]})

    with pytest.raises(ValueError, match="no matched"):
        compare_label_seed_stability(first, second)
