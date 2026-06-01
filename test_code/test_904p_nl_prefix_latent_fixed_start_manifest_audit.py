import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_fixed_start_manifest_audit import (
    build_manifest_audit,
    discover_repeat_cases,
    parse_case_slug,
)


def _synthetic_case(
    *,
    narrative: str,
    start_index: int,
    direction: float,
    seed: int | None = None,
    noise: float = 0.0,
) -> dict:
    start = np.zeros(39, dtype=np.float32)
    start[25] = float(start_index)
    states = np.broadcast_to(start[None, None, :], (6, 4, 39)).copy()
    time = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    sample_tilt = np.linspace(-0.2, 0.2, 6, dtype=np.float32)
    states[:, :, 25] += direction * time[None, :] + sample_tilt[:, None] + noise
    return {
        "case_name": f"{narrative}_start{start_index}"
        + ("" if seed is None else f"#seed_{seed}"),
        "narrative": narrative,
        "start_index": start_index,
        "seed": seed,
        "label": narrative,
        "color": "#1565C0",
        "states": states,
        "start": start,
        "sample_count": int(states.shape[0]),
        "run_report": "synthetic.json",
        "validation_operational_status": "pass",
        "validation_operational_warnings": [],
        "validation_operational_failures": [],
        "validation_start_distance_z": 0.0,
    }


def test_parse_case_slug_extracts_narrative_and_start() -> None:
    assert parse_case_slug("fragile_risk_on_start18") == ("fragile_risk_on", 18)


def test_manifest_audit_compares_narratives_within_fixed_start() -> None:
    observed = []
    repeats = []
    for start_index in [18, 40]:
        observed.append(
            _synthetic_case(
                narrative="risk_on",
                start_index=start_index,
                direction=1.0,
            )
        )
        observed.append(
            _synthetic_case(
                narrative="risk_off",
                start_index=start_index,
                direction=-1.0,
            )
        )
        for narrative, direction in [("risk_on", 1.0), ("risk_off", -1.0)]:
            repeats.append(
                _synthetic_case(
                    narrative=narrative,
                    start_index=start_index,
                    direction=direction,
                    seed=1,
                    noise=0.001,
                )
            )
            repeats.append(
                _synthetic_case(
                    narrative=narrative,
                    start_index=start_index,
                    direction=direction,
                    seed=2,
                    noise=-0.001,
                )
            )

    report = build_manifest_audit(
        observed,
        repeats,
        [],
        max_start_abs_diff=1e-6,
        max_start_only_ratio=0.50,
        max_repeat_ratio=0.75,
        max_bootstrap_ratio=10.0,
    )

    assert report["case_counts"]["start_count"] == 2
    assert report["case_counts"]["observed_pair_count"] == 2
    assert report["case_counts"]["observed_sample_matched_pair_count"] == 8
    assert report["case_counts"]["repeat_pair_count"] == 4
    assert report["failures"] == []
    assert "start_only_null_absent" in report["warnings"]
    assert report["ratios"]["repeat_to_observed_path_energy"] < 0.75
    assert report["ratios"][
        "bootstrap_to_sample_matched_observed_path_energy"
    ] is not None
    assert "observed_narrative_sample_matched" in report["summaries"]
    assert len(report["observed_sample_matched_pairwise"]) == 8


def test_manifest_audit_accepts_start_only_null_cases() -> None:
    observed = []
    start_only = []
    repeats = []
    for narrative, direction in [("risk_on", 1.0), ("risk_off", -1.0)]:
        observed.append(
            _synthetic_case(
                narrative=narrative,
                start_index=18,
                direction=direction,
            )
        )
        start_only.append(
            _synthetic_case(
                narrative=narrative,
                start_index=18,
                direction=0.0,
            )
        )
        repeats.append(
            _synthetic_case(
                narrative=narrative,
                start_index=18,
                direction=direction,
                seed=1,
                noise=0.001,
            )
        )
        repeats.append(
            _synthetic_case(
                narrative=narrative,
                start_index=18,
                direction=direction,
                seed=2,
                noise=-0.001,
            )
        )

    report = build_manifest_audit(
        observed,
        repeats,
        start_only,
        max_start_abs_diff=1e-6,
        max_start_only_ratio=0.50,
        max_repeat_ratio=0.75,
        max_bootstrap_ratio=10.0,
    )

    assert report["case_counts"]["start_only_pair_count"] == 1
    assert "start_only_null_absent" not in report["warnings"]
    assert report["ratios"]["start_only_to_observed_path_energy"] == 0.0


def test_manifest_audit_reports_start_reliability_promotion_warning() -> None:
    observed = [
        _synthetic_case(narrative="risk_on", start_index=18, direction=1.0),
        _synthetic_case(narrative="risk_off", start_index=18, direction=-1.0),
    ]
    observed[0]["validation_operational_status"] = "warning"
    observed[0]["validation_operational_warnings"] = ["large_start_distance"]
    observed[0]["validation_start_distance_z"] = 22.0

    report = build_manifest_audit(
        observed,
        [],
        [],
        max_start_abs_diff=1e-6,
        max_start_only_ratio=0.50,
        max_repeat_ratio=10.0,
        max_bootstrap_ratio=10.0,
        enforce_start_reliability=True,
    )

    assert report["promotion_status"] == "warning"
    assert "observed_operational_start_warnings" in report["promotion_warnings"]
    assert report["start_reliability"]["warning_counts"] == {
        "large_start_distance": 1
    }


def test_discover_repeat_cases_accepts_comma_separated_observed_roots(monkeypatch):
    def fake_discover_observed_cases(root, *, variant_dir, fan_scale):
        suffix = str(root).split("_")[-1]
        return [
            {
                "case_name": "risk_on_start18",
                "label": "risk on",
                "narrative": "risk_on",
                "start_index": 18,
                "seed": None,
                "states": np.zeros((2, 2, 39), dtype=np.float32),
                "start": np.zeros(39, dtype=np.float32),
                "sample_count": 2,
                "run_report": f"{suffix}.json",
            }
        ]

    monkeypatch.setattr(
        "experiments.backfill.block_ar.nl_prefix_latent_fixed_start_manifest_audit.discover_observed_cases",
        fake_discover_observed_cases,
    )

    rows = discover_repeat_cases(
        "repeat_seed_1,repeat_seed_2",
        variant_dir="variant",
        fan_scale=1.0,
    )

    assert [row["seed"] for row in rows] == [1, 2]
    assert [row["case_name"] for row in rows] == [
        "risk_on_start18#seed_1",
        "risk_on_start18#seed_2",
    ]
