import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar import nl_paper_conditionality_evidence_pack as pack


def test_support_matrix_is_symmetric_and_preserves_labels() -> None:
    rows = [
        {
            "left_label": "Risk-on",
            "right_label": "Risk-off",
            "jaccard": 0.25,
        },
        {
            "left_label": "Risk-on",
            "right_label": "Inflation",
            "jaccard": 0.0,
        },
    ]

    labels = pack._support_labels(rows)
    matrix = pack._support_matrix(rows, labels)

    assert labels == ["Risk-on", "Risk-off", "Inflation"]
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == 1.0
    assert matrix[0, 1] == matrix[1, 0] == 0.25
    assert matrix[0, 2] == matrix[2, 0] == 0.0


def test_portfolio_case_rows_use_current_loss_fields() -> None:
    summary = {
        "case_summaries": [
            {
                "label": "Safe haven",
                "portfolio_stats": {
                    "terminal_p50": 3.5,
                    "terminal_p10": -4.0,
                    "terminal_p90": 8.0,
                    "terminal_var95_loss": 6.25,
                    "terminal_expected_shortfall95_loss": 9.5,
                },
                "top_support": [
                    {
                        "window_id": "joint39_val_0108",
                        "history_end_date": "2016-06-30",
                        "weight": 0.47,
                    }
                ],
            }
        ]
    }

    rows = pack._portfolio_case_rows(summary)

    assert rows == [
        {
            "label": "Safe haven",
            "terminal_p50": 3.5,
            "terminal_p10": -4.0,
            "terminal_p90": 8.0,
            "var95_loss": 6.25,
            "es95_loss": 9.5,
            "top_support": "historical support ending 2016-06-30 (47%)",
        }
    ]


def test_tex_escape_handles_percent_without_textbackslash_artifact() -> None:
    escaped = pack._tex_escape("95% loss range")

    assert escaped == r"95\% loss range"
    assert "textbackslash" not in escaped
