import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_hard_negative_bank_validate import validate_bank


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _manifest_row(**overrides: object) -> dict:
    row = {
        "target_window_id": "joint39_train_0001",
        "positive_view": "sparse_user_query",
        "negative_window_id": "joint39_train_0100",
        "positive_text": "Equities are lower and volatility is higher in a defensive tape.",
        "contradiction_channels": ["SPX", "VIX"],
    }
    row.update(overrides)
    return row


def _bank_row(**overrides: object) -> dict:
    row = {
        "row_id": "joint39_train_0001__sparse_user_query",
        "target_window_id": "joint39_train_0001",
        "positive_view": "sparse_user_query",
        "negative_window_id": "joint39_train_0100",
        "hard_negative_text": (
            "Equities are firmer and volatility is lower, so the current prefix "
            "reads as relief rather than a defensive stress tape."
        ),
        "generated_negative_text": (
            "Equities are firmer and volatility is lower, so the current prefix "
            "reads as relief rather than a defensive stress tape."
        ),
        "quality_notes": ["Opposes the SPX and VIX stress channels."],
    }
    row.update(overrides)
    return row


def test_validate_bank_rejects_same_window_negative_link(tmp_path: Path) -> None:
    manifest = [_manifest_row(negative_window_id="joint39_train_0001")]
    bank = [_bank_row(negative_window_id="joint39_train_0001")]
    manifest_path = tmp_path / "manifest.jsonl"
    bank_path = tmp_path / "bank.jsonl"
    _write_jsonl(manifest_path, manifest)
    _write_jsonl(bank_path, bank)

    report = validate_bank(
        manifest_jsonl=manifest_path,
        bank_jsonl=bank_path,
        expected_count=1,
        allow_partial=True,
    )

    assert report["status"] == "fail"
    assert any(
        err.get("code") == "row_validation_failed"
        and any(
            row_err.get("code") == "same_window_negative_link"
            for row_err in err.get("errors", [])
        )
        for err in report["errors"]
    )


def test_validate_bank_rejects_exact_positive_text_copy(tmp_path: Path) -> None:
    positive = "Equities are lower and volatility is higher in a defensive tape."
    manifest = [_manifest_row(positive_text=positive)]
    bank = [
        _bank_row(
            hard_negative_text=positive,
            generated_negative_text=positive,
        )
    ]
    manifest_path = tmp_path / "manifest.jsonl"
    bank_path = tmp_path / "bank.jsonl"
    _write_jsonl(manifest_path, manifest)
    _write_jsonl(bank_path, bank)

    report = validate_bank(
        manifest_jsonl=manifest_path,
        bank_jsonl=bank_path,
        expected_count=1,
        allow_partial=True,
    )

    assert report["status"] == "fail"
    assert any(
        err.get("code") == "row_validation_failed"
        and any(
            row_err.get("code") == "copied_positive_text"
            for row_err in err.get("errors", [])
        )
        for err in report["errors"]
    )
