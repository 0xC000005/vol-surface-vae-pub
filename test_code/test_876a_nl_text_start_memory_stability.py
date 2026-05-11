import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_text_start_memory_stability import (
    extract_method_metrics,
    parse_report_arg,
    summarize_text_start_memory_stability,
)


def _report(base_cos: float, cand_cos: float, cand_gap: float = 0.88) -> dict:
    return {
        "results": {
            "baseline": {
                "summary": {
                    "heldout_mean_target_cosine": base_cos,
                    "heldout_hard_negative_mean_gap": 0.90,
                    "heldout_hard_negative_mean_margin": 0.70,
                    "heldout_recall_at_1_test_pool": 0.08,
                    "heldout_recall_at_3_test_pool": 0.18,
                    "heldout_mean_top_train_cosine": 0.95,
                }
            },
            "candidate": {
                "summary": {
                    "heldout_mean_target_cosine": cand_cos,
                    "heldout_hard_negative_mean_gap": cand_gap,
                    "heldout_hard_negative_mean_margin": 0.69,
                    "heldout_recall_at_1_test_pool": 0.08,
                    "heldout_recall_at_3_test_pool": 0.19,
                    "heldout_mean_top_train_cosine": 0.95,
                }
            },
        }
    }


def test_parse_report_arg_reads_seed_and_path():
    assert parse_report_arg("775:/tmp/report.json") == (775, "/tmp/report.json")


def test_extract_method_metrics_reads_summary_values():
    metrics = extract_method_metrics(_report(0.85, 0.84))

    assert metrics["baseline"]["heldout_mean_target_cosine"] == 0.85
    assert metrics["candidate"]["heldout_recall_at_3_test_pool"] == 0.19


def test_summarize_text_start_memory_stability_passes_all_seeds():
    summary = summarize_text_start_memory_stability(
        [(775, _report(0.85, 0.849)), (776, _report(0.86, 0.859))],
        baseline="baseline",
        candidate="candidate",
    )

    assert summary["status"] == "pass"
    assert summary["preservation_pass_count"] == 2
    assert (
        summary["candidate_minus_baseline"]["heldout_mean_target_cosine"]["mean"]
        == -0.001
    )


def test_summarize_text_start_memory_stability_fails_if_gap_degrades():
    summary = summarize_text_start_memory_stability(
        [(775, _report(0.85, 0.849, cand_gap=0.70))],
        baseline="baseline",
        candidate="candidate",
    )

    assert summary["status"] == "diagnostic_only"
    assert summary["preservation_pass_count"] == 0
