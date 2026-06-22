import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar import nl_14_view_variant_pilot as pilot
from experiments.backfill.block_ar import nl_stride5_14_view_bank as bank


def _card(idx: int) -> dict:
    return {
        "window_id": f"joint39_train_{idx:04d}",
        "archetype": "test_archetype",
        "scenario_title": f"test scenario {idx}",
    }


def _valid_pair(view_name: str, idx: int) -> dict:
    positives = [
        "Softer dollar tape lifts energy and gold while credit stays calm.",
        "Weekly desk read points to commodity strength with contained spreads.",
        "Currency weakness transmits into hard assets and modest rate pressure.",
        "Evidence read shows FX softness, firmer oil, stronger gold, and quiet BBB.",
        "SPX small up; VIX flat; BBB flat; DXY down; crude and gold higher.",
        "Committee severity is moderate, concentrated in FX and commodity exposure.",
        "Decision memo classifies this as commodity-led macro repricing with calm credit.",
        "Professional read centers on weak dollar pressure, hard-asset demand, and limited spread stress.",
        "DXY heavy, oil and gold bid, credit quiet.",
        "My commodity hedge is working while the dollar leg feels exposed.",
        "Dollar softness is feeding oil and gold.",
        "Credit is thin confirmation; commodities are carrying the read.",
        "Rates are only firmer at the margin; commodities are louder.",
        "Morning sheet: dollar lower, oil bid, gold bid, credit calm.",
    ]
    negatives = [
        "Funding pressure supports the dollar while metals and energy soften.",
        "Weekly monitor flags dollar strength, softer commodities, and spread strain.",
        "Liquidity tightening moves through FX pressure and commodity liquidation.",
        "Evidence read shows a firmer dollar, lower crude, lower gold, and wider BBB.",
        "SPX lower; VIX higher; BBB wider; DXY up; crude and gold lower.",
        "Committee severity is elevated, led by dollar funding and credit pressure.",
        "Decision memo classifies this as defensive dollar pressure with commodity weakness.",
        "Professional read highlights funding scarcity, lower energy prices, weaker bullion, and risk aversion.",
        "DXY firm, oil offered, gold sold, credit shaky.",
        "My concern is dollar funding stress hitting commodity exposure.",
        "Dollar strength is weighing on oil and gold.",
        "Credit is the active problem; commodity support is missing.",
        "Rates are easing while crude and gold lose momentum.",
        "Morning sheet: dollar higher, oil softer, gold lower, spreads wider.",
    ]
    return {
        "view_name": view_name,
        "positive_text": positives[idx],
        "negative_window_id": f"joint39_train_{500 + idx:04d}",
        "negative_text": negatives[idx],
        "quality_notes": ["Structured channels differ without direct training-language labels."],
    }


def _valid_report(target_window_id: str = "joint39_train_0000") -> dict:
    pairs = [
        _valid_pair(view_name, idx)
        for idx, view_name in enumerate(pilot.EXPECTED_VIEW_NAMES)
    ]
    negative_candidates = [
        {"window_id": pair["negative_window_id"]} for pair in pairs
    ]
    target = {
        "window_id": target_window_id,
        "scenario_title": "test target scenario",
    }
    batch = pilot.FourteenViewPilotBatch(
        target_window_id=target_window_id,
        target_title="test target scenario",
        pairs=[pilot.NarrativePair(**pair) for pair in pairs],
    )
    validation = pilot.validate_batch(
        batch=batch,
        target=target,
        negative_candidates=negative_candidates,
    )
    assert validation["status"] == "pass"
    return {
        "status": "pass",
        "target_window_id": target_window_id,
        "target": target,
        "pairs": pairs,
        "negative_candidates": negative_candidates,
        "assigned_negative_candidates": [],
        "validation": validation,
        "errors": [],
        "dry_run": False,
        "local_prose_generated": False,
        "artifact_paths": {"review": "/tmp/review.md", "report": "/tmp/report.json"},
    }


def test_select_stride_targets_uses_every_fifth_window():
    cards = [_card(idx) for idx in range(13)]

    targets = bank.select_stride_targets(cards=cards, stride=5)

    assert [row["window_id"] for row in targets] == [
        "joint39_train_0000",
        "joint39_train_0005",
        "joint39_train_0010",
    ]


def test_validate_generated_target_report_accepts_revalidated_report(tmp_path: Path):
    report_path = tmp_path / "fourteen_view_report.json"
    report_path.write_text(json.dumps(_valid_report(), sort_keys=True), encoding="utf-8")

    validation = bank.validate_generated_target_report(report_path)

    assert validation["status"] == "pass"
    assert validation["pair_count"] == 14
    assert validation["positive_count"] == 14
    assert validation["negative_count"] == 14


def test_validate_generated_target_report_rejects_parse_or_validation_failures(
    tmp_path: Path,
):
    report = _valid_report()
    report["pairs"] = report["pairs"][:-1]
    report["validation"] = {"status": "fail", "error_count": 1, "errors": []}
    report_path = tmp_path / "fourteen_view_report.json"
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    validation = bank.validate_generated_target_report(report_path)

    assert validation["status"] == "fail"
    assert any(error["code"] == "pair_count_mismatch" for error in validation["errors"])


def test_validate_generated_target_report_allows_stale_embedded_fail_status(
    tmp_path: Path,
):
    report = _valid_report()
    report["status"] = "fail"
    report["validation"] = {
        "status": "fail",
        "error_count": 1,
        "errors": [{"code": "old_validator_false_positive"}],
    }
    report_path = tmp_path / "fourteen_view_report.json"
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    validation = bank.validate_generated_target_report(report_path)

    assert validation["status"] == "pass"
    assert validation["embedded_report_status"] == "fail"


def test_existing_pass_record_reuses_valid_artifacts(tmp_path: Path):
    target = {"window_id": "joint39_train_0000", "window_number": 0}
    target_dir = tmp_path / "targets" / target["window_id"]
    target_dir.mkdir(parents=True)
    report_path = target_dir / "fourteen_view_report.json"
    review_path = target_dir / "fourteen_view_review.md"
    report = _valid_report(target["window_id"])
    report["artifact_paths"] = {"report": str(report_path), "review": str(review_path)}
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    review_path.write_text("# review\n", encoding="utf-8")

    record = bank.existing_pass_record(target=target, output_dir=tmp_path)

    assert record is not None
    assert record["status"] == "pass"
    assert record["reused_existing"] is True
    assert record["pair_count"] == 14


def test_build_batch_validation_summary_counts_failures():
    records = [
        {"target_window_id": "joint39_train_0000", "status": "pass"},
        {"target_window_id": "joint39_train_0005", "status": "fail"},
    ]

    summary = bank.build_batch_validation_summary(batch_index=0, records=records)

    assert summary["status"] == "fail"
    assert summary["pass_count"] == 1
    assert summary["fail_count"] == 1


def test_run_bank_generation_dry_run_writes_manifest(tmp_path: Path):
    cards_path = tmp_path / "cards.jsonl"
    cards_path.write_text(
        "".join(json.dumps(_card(idx), sort_keys=True) + "\n" for idx in range(12)),
        encoding="utf-8",
    )
    args = Namespace(
        cards_jsonl=cards_path,
        support_cards_jsonl=cards_path,
        output_dir=tmp_path / "out",
        stride=5,
        batch_size=2,
        max_targets=None,
        max_batches=None,
        start_batch=0,
        timeout_seconds=5,
        validation_retries=1,
        negative_candidate_count=200,
        reuse_existing_pass=False,
        stop_on_failure=True,
        dry_run=True,
    )

    report = bank.run_bank_generation(args)

    assert report["status"] == "dry_run"
    assert report["target_count_selected"] == 3
    assert report["batch_count"] == 2
    assert Path(report["artifact_paths"]["manifest"]).exists()


def test_run_bank_generation_resume_loads_prior_batch_validations(tmp_path: Path):
    cards_path = tmp_path / "cards.jsonl"
    cards_path.write_text(
        "".join(json.dumps(_card(idx), sort_keys=True) + "\n" for idx in range(12)),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    prior_records = [
        {
            "target_window_id": "joint39_train_0000",
            "status": "pass",
            "pair_count": 14,
            "positive_count": 14,
            "negative_count": 14,
        },
        {
            "target_window_id": "joint39_train_0005",
            "status": "pass",
            "pair_count": 14,
            "positive_count": 14,
            "negative_count": 14,
        },
    ]
    prior_payload = {
        "schema_version": "stride5_fourteen_view_batch_validation_v1",
        "summary": bank.build_batch_validation_summary(
            batch_index=0,
            records=prior_records,
        ),
        "records": prior_records,
    }
    prior_path = output_dir / "batch_validations" / "batch_0000_validation.json"
    prior_path.parent.mkdir(parents=True)
    prior_path.write_text(json.dumps(prior_payload, sort_keys=True), encoding="utf-8")
    args = Namespace(
        cards_jsonl=cards_path,
        support_cards_jsonl=cards_path,
        output_dir=output_dir,
        stride=5,
        batch_size=2,
        max_targets=None,
        max_batches=None,
        start_batch=1,
        timeout_seconds=5,
        validation_retries=1,
        negative_candidate_count=200,
        reuse_existing_pass=False,
        stop_on_failure=True,
        dry_run=True,
    )

    report = bank.run_bank_generation(args)

    assert report["summary"]["record_count"] == 3
    assert report["summary"]["pass_count"] == 2
    assert report["summary"]["dry_run_count"] == 1
    assert [batch["batch_index"] for batch in report["batch_summaries"]] == [0, 1]
    assert {record["target_window_id"] for record in report["records"]} == {
        "joint39_train_0000",
        "joint39_train_0005",
        "joint39_train_0010",
    }


def test_build_target_command_passes_negative_candidate_count(tmp_path: Path):
    args = Namespace(
        cards_jsonl=Path("cards.jsonl"),
        support_cards_jsonl=Path("support.jsonl"),
        timeout_seconds=1200,
        validation_retries=2,
        negative_candidate_count=200,
    )

    cmd = bank.build_target_command(
        target_id="joint39_train_0070",
        target_output_dir=tmp_path,
        args=args,
    )

    assert "--negative-candidate-count" in cmd
    assert cmd[cmd.index("--negative-candidate-count") + 1] == "200"
