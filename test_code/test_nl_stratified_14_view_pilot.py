import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar import nl_stratified_14_view_pilot as stratified


def test_select_stratified_targets_balances_archetypes_and_skips_infeasible():
    cards = []
    for archetype in ("risk_off", "rates", "commodity"):
        for idx in range(4):
            cards.append(
                {
                    "window_id": f"{archetype}_{idx}",
                    "archetype": archetype,
                    "scenario_title": f"{archetype} title {idx}",
                }
            )

    selected = stratified.select_stratified_targets(
        cards=cards,
        target_count=5,
        max_per_archetype=2,
        is_feasible=lambda card: not str(card["window_id"]).endswith("_0"),
    )

    assert len(selected) == 5
    assert all(not row["window_id"].endswith("_0") for row in selected)
    counts = {}
    for row in selected:
        counts[row["archetype"]] = counts.get(row["archetype"], 0) + 1
    assert max(counts.values()) <= 2
    assert len(counts) == 3


def test_build_stratified_summary_markdown_lists_review_paths():
    records = [
        {
            "target_window_id": "joint39_train_1000",
            "archetype": "risk_off",
            "scenario_title": "risk-off example",
            "status": "pass",
            "validation_error_count": 0,
            "review_path": "/tmp/risk_off.md",
            "report_path": "/tmp/risk_off.json",
        },
        {
            "target_window_id": "joint39_train_2000",
            "archetype": "rates",
            "scenario_title": "rates example",
            "status": "fail",
            "validation_error_count": 2,
            "review_path": "/tmp/rates.md",
            "report_path": "/tmp/rates.json",
        },
    ]

    summary = stratified.build_summary(records)
    markdown = stratified.build_summary_markdown(records, summary)

    assert summary["target_count"] == 2
    assert summary["pass_count"] == 1
    assert summary["fail_count"] == 1
    assert "joint39_train_1000" in markdown
    assert "/tmp/risk_off.md" in markdown
    assert "rates example" in markdown


def test_build_summary_does_not_count_dry_run_as_failure():
    records = [
        {
            "target_window_id": "joint39_train_1000",
            "archetype": "risk_off",
            "scenario_title": "risk-off example",
            "status": "dry_run",
            "validation_error_count": "",
            "review_path": "/tmp/risk_off.md",
            "report_path": "/tmp/risk_off.json",
        }
    ]

    summary = stratified.build_summary(records)

    assert summary["target_count"] == 1
    assert summary["pass_count"] == 0
    assert summary["fail_count"] == 0
    assert summary["dry_run_count"] == 1


def test_existing_pass_record_can_be_reused_without_rerun(tmp_path):
    target = {
        "window_id": "joint39_train_1000",
        "archetype": "risk_off",
        "scenario_title": "risk-off example",
    }
    target_dir = tmp_path / "joint39_train_1000"
    target_dir.mkdir()
    report_path = target_dir / "fourteen_view_report.json"
    review_path = target_dir / "fourteen_view_review.md"
    report_path.write_text(
        '{"status":"pass","validation":{"error_count":0}}\n',
        encoding="utf-8",
    )
    review_path.write_text("# review\n", encoding="utf-8")

    record = stratified.existing_pass_record(target=target, output_dir=tmp_path)

    assert record is not None
    assert record["status"] == "pass"
    assert record["validation_error_count"] == 0
    assert record["reused_existing"] is True
    assert record["review_path"] == str(review_path)
