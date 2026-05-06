import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_web_evidence_enrichment import (
    _response_sources,
    build_web_evidence_messages,
    build_evidence_review_report,
    score_bundle_salience,
    select_bundles_for_enrichment,
    summarize_enrichment,
)


def _bundle() -> dict:
    return {
        "window_id": "joint39_val_0000",
        "calendar": {
            "calendar_start_date": "2015-12-14",
            "calendar_end_date": "2016-01-27",
            "forecast_start_date": "2016-01-28",
            "forecast_end_date": "2016-03-10",
        },
        "market_implications": [
            {"market": "SPX", "direction": "down", "magnitude": "large"},
            {"market": "CRUDE_OIL", "direction": "down", "magnitude": "large"},
        ],
        "narrative_catalysts": [
            {
                "label": "oil-price collapse analogy",
                "category": "commodity_deflation",
                "grounding_status": "historical_analogy",
                "description": "Oil weakness with risk-off markets.",
                "market_linkage": ["CRUDE_OIL: DOWN LARGE", "SPX: DOWN LARGE"],
            }
        ],
        "narratives": [{"text": "Risk-off market state."}],
    }


def test_build_web_evidence_messages_keeps_dates_and_guardrails() -> None:
    messages = build_web_evidence_messages(_bundle(), max_events=2)
    joined = "\n".join(message["content"] for message in messages)
    assert "2015-12-14 to 2016-01-27" in joined
    assert "Return at most 2 cited catalyst events" in joined
    assert "Do not change the market implication target" in joined
    assert "oil-price collapse analogy" in joined


def test_response_sources_extracts_annotations_and_search_sources() -> None:
    class Item:
        def __init__(self, payload):
            self._payload = payload

        def model_dump(self):
            return self._payload

    response = type(
        "Response",
        (),
        {
            "output": [
                Item(
                    {
                        "action": {
                            "sources": [
                                {"type": "url", "url": "https://example.com/a", "title": "A"}
                            ]
                        }
                    }
                ),
                Item(
                    {
                        "content": [
                            {
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://example.com/a",
                                        "title": "A",
                                    },
                                    {
                                        "type": "url_citation",
                                        "url": "https://example.com/b",
                                        "title": "B",
                                    },
                                ]
                            }
                        ]
                    }
                ),
            ]
        },
    )()
    sources = _response_sources(response)
    assert sources == [
        {"type": "url", "url": "https://example.com/a", "title": "A"},
        {"type": "url", "url": "https://example.com/b", "title": "B"},
    ]


def test_summarize_enrichment_counts_cited_events_and_sources() -> None:
    summary = summarize_enrichment(
        [
            {
                "raw_sources": [{"url": "u1"}, {"url": "u2"}],
                "web_evidence": {
                    "cited_events": [
                        {"grounding_status": "cited_external_event"},
                        {"grounding_status": "historical_analogy"},
                    ]
                },
            }
        ]
    )
    assert summary == {
        "window_count": 1,
        "cited_external_event_count": 1,
        "event_status_counts": {
            "cited_external_event": 1,
            "historical_analogy": 1,
        },
        "raw_source_count": 2,
    }


def test_score_bundle_salience_prefers_cross_asset_stress_over_flat_state() -> None:
    stress = _bundle()
    stress["market_implications"] = [
        {"market": "SPX", "direction": "down", "magnitude": "large"},
        {"market": "VIX", "direction": "up", "magnitude": "large"},
        {"market": "BBB_OAS", "direction": "wider", "magnitude": "medium"},
        {"market": "US10Y", "direction": "down", "magnitude": "medium"},
    ]
    calm = _bundle()
    calm["market_implications"] = [
        {"market": "SPX", "direction": "flat", "magnitude": "flat"},
        {"market": "VIX", "direction": "flat", "magnitude": "flat"},
    ]

    stress_score = score_bundle_salience(stress)
    calm_score = score_bundle_salience(calm)

    assert stress_score["score"] > calm_score["score"]
    assert "equity" in stress_score["tags"]
    assert "vol" in stress_score["tags"]
    assert "credit" in stress_score["tags"]
    assert calm_score["score"] == 0.0


def test_select_bundles_for_enrichment_returns_eventful_and_calm_controls() -> None:
    bundles = []
    for idx, spx_magnitude in enumerate(["flat", "small", "medium", "large"]):
        bundle = _bundle()
        bundle["window_id"] = f"w{idx}"
        bundle["market_implications"] = [
            {
                "market": "SPX",
                "direction": "down" if spx_magnitude != "flat" else "flat",
                "magnitude": spx_magnitude,
            },
            {
                "market": "VIX",
                "direction": "up" if spx_magnitude in {"medium", "large"} else "flat",
                "magnitude": spx_magnitude,
            },
        ]
        bundles.append(bundle)

    selected, selection_report = select_bundles_for_enrichment(
        bundles,
        eventful_windows=2,
        calm_windows=1,
    )

    assert [bundle["window_id"] for bundle in selected] == ["w3", "w2", "w0"]
    assert selection_report["selection_mode"] == "salience"
    assert selection_report["eventful_window_ids"] == ["w3", "w2"]
    assert selection_report["calm_window_ids"] == ["w0"]


def test_build_evidence_review_report_flags_low_confidence_and_bad_date_alignment() -> None:
    rows = [
        {
            "window_id": "w0",
            "web_evidence": {
                "cited_events": [
                    {
                        "label": "supported inside-window event",
                        "grounding_status": "cited_external_event",
                        "date_match": "inside_window",
                        "relevance_score": 0.86,
                        "citations": [
                            {
                                "url": "https://www.reuters.com/markets/example",
                                "title": "Market stress",
                                "published_date": "2016-01-07",
                            }
                        ],
                    },
                    {
                        "label": "weak source",
                        "grounding_status": "cited_external_event",
                        "date_match": "unknown",
                        "relevance_score": 0.34,
                        "citations": [],
                    },
                    {
                        "label": "outside-window event",
                        "grounding_status": "cited_external_event",
                        "date_match": "outside_window",
                        "relevance_score": 0.91,
                        "citations": [
                            {
                                "url": "https://example.com/outside",
                                "title": "Outside",
                            }
                        ],
                    },
                ]
            },
        }
    ]

    review = build_evidence_review_report(rows, min_relevance=0.6)

    assert review["accepted_cited_event_count"] == 1
    assert review["flag_counts"] == {
        "low_relevance": 1,
        "missing_citation": 1,
        "weak_date_alignment": 2,
    }
    assert review["source_domain_counts"] == {
        "www.reuters.com": 1,
        "example.com": 1,
    }
