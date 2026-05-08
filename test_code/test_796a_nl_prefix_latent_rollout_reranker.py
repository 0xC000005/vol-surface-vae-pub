import json
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_rollout_reranker import (
    choose_best_candidate,
    parse_candidate_arg,
    rerank_casebooks,
    run_rollout_reranker,
)


def _write_casebook(path, *, case_name: str, mismatch_count: int, checked_count: int):
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_name": case_name,
                        "selected_start_status": "pass",
                        "overall_status": "warning",
                        "path_labels": ["Selected start: joint39_val_0001"],
                        "market_alignment": {
                            "status": "warning" if mismatch_count else "pass",
                            "checked_count": checked_count,
                            "match_count": checked_count - mismatch_count,
                            "mismatch_count": mismatch_count,
                            "skipped_count": 0,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_parse_candidate_arg_supports_named_and_unnamed_paths() -> None:
    assert parse_candidate_arg("balanced=/tmp/a.json") == ("balanced", "/tmp/a.json")
    name, path = parse_candidate_arg("/tmp/casebook.json")
    assert name == "tmp"
    assert path == "/tmp/casebook.json"


def test_choose_best_candidate_prefers_lower_mismatch_rate() -> None:
    best = choose_best_candidate(
        [
            {
                "candidate_name": "a",
                "mismatch_rate": 0.5,
                "mismatch_count": 2,
                "selected_start_status": "pass",
            },
            {
                "candidate_name": "b",
                "mismatch_rate": 0.25,
                "mismatch_count": 1,
                "selected_start_status": "warning",
            },
        ]
    )

    assert best["candidate_name"] == "b"


def test_rerank_casebooks_selects_per_case_best_candidate(tmp_path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_casebook(a, case_name="risk_on", mismatch_count=2, checked_count=4)
    _write_casebook(b, case_name="risk_on", mismatch_count=1, checked_count=4)

    rerank = rerank_casebooks([f"a={a}", f"b={b}"])

    assert rerank["selected_cases"][0]["case_name"] == "risk_on"
    assert rerank["selected_cases"][0]["chosen_candidate"] == "b"


def test_run_rollout_reranker_writes_summary(tmp_path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_casebook(a, case_name="risk_on", mismatch_count=2, checked_count=4)
    _write_casebook(b, case_name="risk_on", mismatch_count=1, checked_count=4)

    summary = run_rollout_reranker(
        SimpleNamespace(
            candidate=[f"a={a}", f"b={b}"],
            output_dir=str(tmp_path / "out"),
        )
    )

    assert summary["status"] == "ok"
    assert summary["case_count"] == 1
    assert summary["chosen_candidate_counts"] == {"b": 1}
    assert summary["total_mismatches"] == 1
    assert (tmp_path / "out" / "rollout_reranker_summary.json").exists()
