import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.audit_576a_unified_increment_panel import (
    build_unified_variable_specs,
    clean_nonpositive_log_level_factors,
)


def _columns() -> list[str]:
    return [f"iv:{idx:02d}" for idx in range(25)] + [
        "factor:aaa_oas",
        "factor:aaa_oas_diff",
        "factor:crude_oil",
        "factor:crude_oil_diff",
    ]


def test_observed_positive_policy_logs_positive_diff_referenced_factor() -> None:
    columns = _columns()
    panel = np.ones((4, len(columns)), dtype=np.float32)
    panel[:, 25] = np.array([0.7, 0.8, 0.9, 1.0], dtype=np.float32)
    panel[:, 27] = np.array([20.0, -5.0, 10.0, 15.0], dtype=np.float32)

    default_specs = build_unified_variable_specs(
        columns,
        panel=panel,
        iv_count=25,
    )
    observed_specs = build_unified_variable_specs(
        columns,
        panel=panel,
        iv_count=25,
        positive_level_policy="observed_positive",
    )

    assert default_specs[25].name == "factor:aaa_oas"
    assert default_specs[25].transform == "diff_level"
    assert observed_specs[25].transform == "log_level"
    assert observed_specs[26].name == "factor:crude_oil"
    assert observed_specs[26].transform == "diff_level"


def test_observed_positive_cleaning_extends_to_non_logret_level_candidates() -> None:
    columns = _columns()
    panel = np.ones((4, len(columns)), dtype=np.float32)
    panel[:, 25] = np.array([0.7, 0.0, np.nan, 1.0], dtype=np.float32)
    panel[:, 27] = np.array([20.0, -5.0, 10.0, 15.0], dtype=np.float32)

    cleaned, report = clean_nonpositive_log_level_factors(
        panel,
        columns,
        iv_count=25,
        positive_level_policy="observed_positive",
    )
    specs = build_unified_variable_specs(
        columns,
        panel=cleaned,
        iv_count=25,
        positive_level_policy="observed_positive",
    )

    assert report["cleaned_columns"] == {"factor:aaa_oas": 2}
    assert report["diff_fallback_columns"] == ["factor:crude_oil"]
    assert np.all(np.isfinite(cleaned[:, 25]))
    assert np.all(cleaned[:, 25] > 0.0)
    assert specs[25].transform == "log_level"
