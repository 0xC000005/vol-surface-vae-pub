import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_scenario_descriptions import (
    ContrastiveDescriptions,
    FreeFormDescription,
    MarketMoveAudit,
    NarrativeCatalyst,
    ScenarioDescriptionBundle,
    build_description_messages,
    build_batch_request_rows,
    bundle_from_batch_response_row,
    read_jsonl,
    validate_description_bundle,
    write_jsonl,
)


def _good_bundle() -> ScenarioDescriptionBundle:
    return ScenarioDescriptionBundle(
        window_id="w1",
        panel_version="joint39",
        canonical_machine_text=(
            "US2Y: UP LARGE; SPX: DOWN LARGE; VIX: UP LARGE; "
            "BBB_OAS: WIDER MEDIUM"
        ),
        descriptions=[
            FreeFormDescription(
                style="terse_trader",
                text="Rates rose hard while equities sold off and volatility climbed.",
            ),
            FreeFormDescription(
                style="risk_manager",
                text=(
                    "A risk-off window with higher front-end rates, weaker equities, "
                    "higher VIX, and wider credit spreads."
                ),
            ),
        ],
        structured_audit=[
            MarketMoveAudit(
                market="US2Y",
                direction="up",
                magnitude="large",
                confidence="high",
                evidence=["us2y_change_z=2.1"],
                inferred=False,
            ),
            MarketMoveAudit(
                market="SPX",
                direction="down",
                magnitude="large",
                confidence="high",
                evidence=["spx_return_z=-2.4"],
                inferred=False,
            ),
        ],
        contrastive=ContrastiveDescriptions(
            opposite=(
                "US2Y: DOWN LARGE; SPX: UP LARGE; VIX: DOWN LARGE; "
                "BBB_OAS: TIGHTER MEDIUM"
            ),
            partial=[
                "US2Y: UP LARGE; SPX: UP LARGE; VIX: DOWN MEDIUM; "
                "BBB_OAS: WIDER MEDIUM"
            ],
            magnitude=[
                "US2Y: UP SMALL; SPX: DOWN SMALL; VIX: UP SMALL; "
                "BBB_OAS: WIDER SMALL"
            ],
        ),
        critique=[
            "Descriptions are direct market-state summaries and avoid unsupported news.",
        ],
        revised_description=(
            "A risk-off market window with front-end rates rising, equities falling, "
            "volatility increasing, and credit spreads widening."
        ),
    )


def test_validate_good_bundle_has_no_issues() -> None:
    issues = validate_description_bundle(_good_bundle())
    assert issues == []


def test_validate_flags_external_news_and_missing_contrast() -> None:
    bundle = _good_bundle()
    bundle.descriptions[0].text = (
        "Geopolitical tensions escalated and central banks caused rates to rise."
    )
    bundle.contrastive.opposite = ""
    issues = validate_description_bundle(bundle)
    codes = {issue.code for issue in issues}
    assert "external_catalyst_requires_grounding" in codes
    assert "missing_opposite_contrast" in codes


def test_validate_allows_grounded_hypothetical_catalyst_as_warning() -> None:
    bundle = _good_bundle()
    bundle.descriptions[0].text = "War-style oil shock analogy with equities lower."
    bundle.narrative_catalysts = [
        NarrativeCatalyst(
            label="war-style oil shock",
            category="geopolitical_supply_shock",
            grounding_status="historical_analogy",
            description="An analogy, not a confirmed historical cause.",
            market_linkage=["CRUDE_OIL: UP LARGE", "SPX: DOWN LARGE"],
        )
    ]
    issues = validate_description_bundle(bundle)
    assert [issue.severity for issue in issues] == ["warning"]


def test_validate_unsupported_catalyst_is_warning_not_rejection() -> None:
    bundle = _good_bundle()
    bundle.narrative_catalysts = [
        NarrativeCatalyst(
            label="no confirmed shock story",
            category="none",
            grounding_status="unsupported",
            description="The market facts do not support a named catalyst.",
        )
    ]
    issues = validate_description_bundle(bundle)
    assert [(issue.code, issue.severity) for issue in issues] == [
        ("unsupported_catalyst", "warning")
    ]


def test_build_description_messages_preserve_direct_market_guardrails() -> None:
    summary = {
        "window_id": "w1",
        "panel_version": "joint39",
        "numeric_summary": {"spx_return_z": -2.4, "us2y_change_z": 2.1},
    }
    messages = build_description_messages(summary)
    joined = "\n".join(str(part) for message in messages for part in message.values())
    assert "Separate observed market facts" in joined
    assert "Never present an uncited event as a confirmed cause" in joined
    assert "narrative_catalysts" in joined
    assert "joint39" in joined
    assert json.dumps(summary, sort_keys=True) in joined


def test_jsonl_roundtrip_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [{"window_id": "w1"}, {"window_id": "w2", "panel_version": "joint39"}]
    write_jsonl(path, rows)
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert read_jsonl(path) == rows


def test_build_batch_request_rows_use_responses_endpoint() -> None:
    summaries = [
        {
            "window_id": "w1",
            "panel_version": "joint39",
            "numeric_summary": {"spx_return_z": -2.4},
        }
    ]
    rows = build_batch_request_rows(summaries, model="gpt-5.4-mini")
    assert len(rows) == 1
    row = rows[0]
    assert row["custom_id"] == "w1"
    assert row["method"] == "POST"
    assert row["url"] == "/v1/responses"
    assert row["body"]["model"] == "gpt-5.4-mini"
    assert row["body"]["text"]["format"]["type"] == "json_schema"
    assert row["body"]["text"]["format"]["strict"] is True
    assert "ScenarioDescriptionBundle" in row["body"]["text"]["format"]["name"]
    schema = row["body"]["text"]["format"]["schema"]
    assert set(schema["required"]) == set(schema["properties"])
    assert set(schema["$defs"]["MarketMoveAudit"]["required"]) == set(
        schema["$defs"]["MarketMoveAudit"]["properties"]
    )


def test_bundle_from_batch_response_row_extracts_structured_output() -> None:
    bundle = _good_bundle()
    row = {
        "custom_id": "w1",
        "response": {
            "status_code": 200,
            "body": {
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": bundle.model_dump_json(),
                            }
                        ]
                    }
                ]
            },
        },
    }
    assert bundle_from_batch_response_row(row) == bundle
