import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar import nl_paper_caption_example_appendix as appendix


def test_select_narrative_finds_requested_text() -> None:
    bundle = {
        "narratives": [
            {"id": "other", "text": "ignore"},
            {"id": "description_terse_trader", "text": "Risk-on window."},
        ]
    }

    assert appendix._select_narrative(bundle, "description_terse_trader") == "Risk-on window."


def test_tex_escape_handles_finance_symbols() -> None:
    escaped = appendix._tex_escape("SPX_1 & 95% P&L")

    assert escaped == r"SPX\_1 \& 95\% P\&L"


def test_public_evidence_line_removes_internal_factor_key() -> None:
    line = appendix._public_evidence_line(
        "SPX up large: spx_30d_change=42.96, confidence high."
    )

    assert line == "SPX up large: 30-day change 42.96, confidence high."
    assert "spx_30d_change" not in line


def test_tex_snippet_contains_public_caption_sections() -> None:
    bundle = {
        "calendar": {
            "calendar_start_date": "2017-06-15",
            "calendar_end_date": "2017-07-27",
            "forecast_start_date": "2017-07-28",
            "forecast_end_date": "2017-09-08",
        }
    }
    codex = {
        "training_caption": "Professional caption.",
        "scenario_title": "Late-July Risk-On Reflation",
        "archetype": "growth_upside_reflation",
        "archetype_confidence": "medium",
        "current_market_state": "Equities rose and VIX fell.",
        "mechanical_summary": "SPX up, VIX down.",
        "trigger": "Observed market pattern.",
        "transmission": "Risk appetite improved.",
        "cross_asset_reaction": "Credit tightened.",
        "sequence": "Equity strength first.",
        "portfolio_vulnerability": "Short equity beta.",
        "risk_manager_implication": "Review hedges.",
        "no_forecast_caveat": "Not a forecast.",
        "evidence_used": ["SPX up large."],
        "ambiguity_flags": ["No news catalyst supplied."],
    }
    tex = appendix._tex_snippet(
        figure_name="example.png",
        bundle=bundle,
        codex=codex,
        simple_text="Risk-on window.",
        market_summary={
            "SPX": {
                "history_start": 1.0,
                "conditioning_date": 2.0,
                "future_end": 3.0,
                "history_change": 1.0,
                "future_change": 1.0,
            }
        },
    )

    assert "Earlier simple narrative" in tex
    assert "Professional risk-manager narrative" in tex
    assert "Raw SPX and VIX levels" in tex
    assert "joint39" not in tex
