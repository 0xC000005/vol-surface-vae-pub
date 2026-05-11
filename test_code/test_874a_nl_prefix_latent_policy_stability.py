import json
import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_policy_stability import (
    extract_method_metrics,
    parse_report_arg,
    summarize_policy_stability,
)


def _report(path, *, energy: float, crps: float, coverage: float = 0.6) -> str:
    path.write_text(
        json.dumps(
            {
                "summary": {
                    "narrative_generator_topk": {
                        "energy_score_z_improvement_vs_persistence": energy,
                        "ensemble_crps_z_improvement_vs_persistence": crps,
                        "coverage_80_mean": coverage,
                        "mean_path_mae_z_improvement_vs_persistence": -0.1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def test_parse_report_arg_reads_policy_seed_path() -> None:
    assert parse_report_arg("baseline:776:/tmp/a.json") == (
        "baseline",
        776,
        "/tmp/a.json",
    )


def test_extract_method_metrics_reads_generator_summary() -> None:
    metrics = extract_method_metrics(
        {
            "summary": {
                "narrative_generator_topk": {
                    "energy_score_z_improvement_vs_persistence": 0.2,
                    "ensemble_crps_z_improvement_vs_persistence": 0.1,
                    "coverage_80_mean": 0.7,
                    "mean_path_mae_z_improvement_vs_persistence": -0.1,
                }
            }
        }
    )

    assert metrics["energy"] == 0.2
    assert metrics["crps"] == 0.1
    assert metrics["coverage_80"] == 0.7


def test_summarize_policy_stability_fails_unstable_candidate(tmp_path) -> None:
    specs = [
        ("baseline", 1, _report(tmp_path / "b1.json", energy=0.20, crps=0.20)),
        ("baseline", 2, _report(tmp_path / "b2.json", energy=0.22, crps=0.22)),
        ("candidate", 1, _report(tmp_path / "c1.json", energy=0.21, crps=0.21)),
        ("candidate", 2, _report(tmp_path / "c2.json", energy=0.19, crps=0.19)),
    ]

    report = summarize_policy_stability(
        specs,
        baseline_policy="baseline",
        candidate_policy="candidate",
        min_energy_gain=0.001,
        min_crps_gain=0.001,
    )

    assert report["status"] == "diagnostic_only"
    assert report["decision"]["promote_candidate"] is False
    assert report["paired_summary"]["common_seed_count"] == 2
    assert report["paired_summary"]["energy_delta_positive_count"] == 1


def test_summarize_policy_stability_passes_consistent_candidate(tmp_path) -> None:
    specs = [
        ("baseline", 1, _report(tmp_path / "b1.json", energy=0.20, crps=0.20)),
        ("baseline", 2, _report(tmp_path / "b2.json", energy=0.20, crps=0.20)),
        ("candidate", 1, _report(tmp_path / "c1.json", energy=0.22, crps=0.22)),
        ("candidate", 2, _report(tmp_path / "c2.json", energy=0.23, crps=0.23)),
    ]

    report = summarize_policy_stability(
        specs,
        baseline_policy="baseline",
        candidate_policy="candidate",
        min_energy_gain=0.001,
        min_crps_gain=0.001,
    )

    assert report["status"] == "stable_candidate"
    assert report["decision"]["promote_candidate"] is True
