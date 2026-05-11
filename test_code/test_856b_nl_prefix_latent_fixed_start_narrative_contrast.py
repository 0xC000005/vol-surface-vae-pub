import json
import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_fixed_start_narrative_contrast import (
    build_fixed_start_contrast,
)


def _write_case(tmp_path, name: str, terminal_shift: float) -> dict[str, object]:
    case_dir = tmp_path / name
    case_dir.mkdir()
    report_path = case_dir / "prefix_latent_story_smoke_report.json"
    report = {
        "selected_start_state": {
            "values_by_name": {
                "factor:spx": 100.0,
                "factor:vix": 20.0,
                "iv:00": 0.1,
            }
        },
        "generation": {
            "terminal_delta_summary": [
                {
                    "market": "SPX",
                    "mean_terminal_delta": terminal_shift,
                    "p10": terminal_shift - 1.0,
                    "p50": terminal_shift,
                    "p90": terminal_shift + 1.0,
                },
                {
                    "market": "VIX",
                    "mean_terminal_delta": -terminal_shift,
                    "p10": -terminal_shift - 1.0,
                    "p50": -terminal_shift,
                    "p90": -terminal_shift + 1.0,
                },
                {
                    "market": "IV_SURFACE",
                    "mean_terminal_delta": 0.1 * terminal_shift,
                    "p10": 0.1 * terminal_shift - 0.1,
                    "p50": 0.1 * terminal_shift,
                    "p90": 0.1 * terminal_shift + 0.1,
                },
            ]
        },
        "variant_rows": [{"is_operational": True}],
        "cached_query": {
            "memory_prior": {
                "candidate_details": [
                    {
                        "rank": 1,
                        "window_index": 7,
                        "weight": 1.0,
                        "memory_support_cosine": 0.9,
                        "history_end_date": "2020-01-01",
                    }
                ]
            }
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return {
        "case_name": name,
        "start_name": "same_start",
        "run_report": str(report_path),
        "memory_prior_direction_status": "pass",
        "memory_prior_support_weighted_match_rate": 1.0,
        "memory_prior_final_mixture_mismatch_count": 0,
    }


def test_build_fixed_start_contrast_detects_same_start_and_pairwise_gap(
    tmp_path,
) -> None:
    bakeoff = {
        "rows": [
            _write_case(tmp_path, "risk_on", 2.0),
            _write_case(tmp_path, "risk_off", -2.0),
        ]
    }

    report = build_fixed_start_contrast(bakeoff, markets=["SPX", "VIX", "IV_SURFACE"])

    assert report["status"] == "pass"
    assert report["start_max_abs_diff"] == 0.0
    assert report["case_count"] == 2
    assert report["pairwise_contrasts"][0]["standardized_l2_gap"] > 0.0
