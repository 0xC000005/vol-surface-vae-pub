import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar import nl_window_density_pilot as density


def _card(idx: int, *, archetype: str, title: str, views_text: str, signs: list[int]) -> dict:
    return {
        "window_id": f"joint39_train_{idx:04d}",
        "archetype": archetype,
        "scenario_title": title,
        "views": {
            "sparse_user_query": views_text,
            "weekly_risk_monitor": views_text,
            "mechanism_first": views_text,
            "technical_factor_evidence": views_text,
            "factor_list_baseline": views_text,
            "institutional_risk_committee_note": views_text,
            "risk_manager_memo": views_text,
            "full_professional": views_text,
        },
        "_test_sign_vector": signs,
    }


def test_build_density_records_compares_requested_offsets():
    cards = [
        _card(
            100,
            archetype="risk_on",
            title="risk-on relief",
            views_text="equities higher volatility lower credit tighter",
            signs=[1, -1, -1, 0],
        ),
        _card(
            101,
            archetype="risk_on",
            title="risk-on relief still present",
            views_text="equities higher volatility lower credit stable",
            signs=[1, -1, 0, 0],
        ),
        _card(
            105,
            archetype="risk_off",
            title="risk-off pressure",
            views_text="equities lower volatility higher credit wider",
            signs=[-1, 1, 1, 0],
        ),
    ]

    records = density.build_density_records(
        cards=cards,
        anchors=["joint39_train_0100"],
        offsets=[0, 1, 5],
        sign_vector_fn=lambda card: card["_test_sign_vector"],
    )

    assert [row["offset"] for row in records] == [0, 1, 5]
    assert records[0]["calendar_overlap_fraction"] == 1.0
    assert records[1]["calendar_overlap_fraction"] == 29 / 30
    assert records[1]["opposite_sign_count"] == 0
    assert records[2]["opposite_sign_count"] == 3
    assert records[2]["same_archetype"] is False


def test_recommend_stride_prefers_five_when_daily_redundant_and_five_day_moves():
    summary = {
        "offsets": {
            1: {
                "mean_view_similarity": 0.82,
                "mean_sign_change_count": 1.2,
                "mean_opposite_sign_count": 0.2,
                "distinct_regime_share": 0.1,
            },
            5: {
                "mean_view_similarity": 0.50,
                "mean_sign_change_count": 4.0,
                "mean_opposite_sign_count": 1.7,
                "distinct_regime_share": 0.55,
            },
            15: {
                "mean_view_similarity": 0.37,
                "mean_sign_change_count": 5.5,
                "mean_opposite_sign_count": 2.4,
                "distinct_regime_share": 0.85,
            },
        }
    }

    recommendation = density.recommend_stride(summary)

    assert recommendation["recommended_stride_days"] == 5
    assert "daily" in recommendation["rationale"].lower()
    assert "5-day" in recommendation["rationale"]


def test_recommend_stride_uses_half_overlap_when_five_day_is_still_redundant():
    summary = {
        "offsets": {
            1: {
                "mean_view_similarity": 0.90,
                "mean_sign_change_count": 0.5,
                "mean_opposite_sign_count": 0.0,
                "distinct_regime_share": 0.0,
            },
            5: {
                "mean_view_similarity": 0.78,
                "mean_sign_change_count": 1.1,
                "mean_opposite_sign_count": 0.2,
                "distinct_regime_share": 0.1,
            },
            15: {
                "mean_view_similarity": 0.45,
                "mean_sign_change_count": 4.4,
                "mean_opposite_sign_count": 1.8,
                "distinct_regime_share": 0.6,
            },
        }
    }

    recommendation = density.recommend_stride(summary)

    assert recommendation["recommended_stride_days"] == 15
    assert "5-day windows still look redundant" in recommendation["rationale"]


def test_build_human_review_markdown_is_summary_first():
    report = {
        "recommendation": {
            "recommended_stride_days": 5,
            "decision": "Generate every 5 trading days first.",
            "rationale": "Daily windows are redundant; 5-day windows move enough.",
        },
        "summary": {
            "anchor_count": 1,
            "offsets": {
                1: {
                    "case_count": 1,
                    "mean_view_similarity": 0.82,
                    "mean_sign_change_count": 1.0,
                    "mean_opposite_sign_count": 0.0,
                    "distinct_regime_share": 0.0,
                }
            },
        },
        "records": [
            {
                "anchor_window_id": "joint39_train_0100",
                "offset_window_id": "joint39_train_0101",
                "offset": 1,
                "anchor_title": "risk-on relief",
                "offset_title": "risk-on continuation",
                "anchor_mechanical_summary": "SPX up; VIX down.",
                "offset_mechanical_summary": "SPX up; VIX down.",
                "view_similarity": 0.82,
                "sign_change_count": 1,
                "opposite_sign_count": 0,
                "same_archetype": True,
                "human_label": "mostly redundant",
                "notable_changes": ["Credit moves from tighter to flat."],
            }
        ],
        "review_records": [
            {
                "anchor_window_id": "joint39_train_0100",
                "offset_window_id": "joint39_train_0101",
                "offset": 1,
                "anchor_title": "risk-on relief",
                "offset_title": "risk-on continuation",
                "anchor_mechanical_summary": "SPX up; VIX down.",
                "offset_mechanical_summary": "SPX up; VIX down.",
                "view_similarity": 0.82,
                "sign_change_count": 1,
                "opposite_sign_count": 0,
                "same_archetype": True,
                "human_label": "mostly redundant",
                "notable_changes": ["Credit moves from tighter to flat."],
            }
        ],
        "artifact_paths": {},
    }

    markdown = density.build_human_review_markdown(report)

    assert markdown.startswith("# Window-Density Pilot")
    assert "## Decision Card" in markdown
    assert "## Offset Summary" in markdown
    assert "## Low-Load Review Tiles" in markdown
    assert markdown.index("## Decision Card") < markdown.index("## Low-Load Review Tiles")


def test_select_review_records_covers_multiple_offsets():
    records = []
    for offset in (1, 5, 15):
        for idx in range(3):
            records.append(
                {
                    "anchor_window_id": f"joint39_train_{100 + idx:04d}",
                    "offset_window_id": f"joint39_train_{100 + idx + offset:04d}",
                    "offset": offset,
                    "human_label": "distinct regime" if idx == 0 else "partial change",
                    "view_similarity": 0.2 + idx / 10,
                }
            )

    selected = density._select_review_records(records, max_records=6)

    assert {row["offset"] for row in selected} == {1, 5, 15}
    counts = {offset: 0 for offset in (1, 5, 15)}
    for row in selected:
        counts[row["offset"]] += 1
    assert counts == {1: 2, 5: 2, 15: 2}
