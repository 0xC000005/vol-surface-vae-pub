#!/usr/bin/env python
"""Run a risk-manager story through the saved narrative-conditioning artifacts.

The smoke test is designed for product demos: a story-like user narrative is
grounded into explicit market implications, embedded, mapped into the generator
condition-memory space, matched to historical analogues, and summarized with the
retrieval-conditioned scenario generator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    NarrativeAdapter,
    _load_joint39_block,
    _reconstruct_states,
    _spec_names,
    retrieval_diagnostics,
    retrieval_weights,
    sample_normal_generator_for_retrieved_analogues,
    summarize_retrieval_generated_states,
)
from experiments.backfill.block_ar.nl_scenario_descriptions import (  # noqa: E402
    load_dotenv_key,
)
from experiments.backfill.block_ar.nl_text_conditioning import (  # noqa: E402
    embed_texts_with_openai,
    normalize_rows,
)


DEFAULT_STORY = (
    "This has the shape of a fragile risk-on rebound: equities are recovering, "
    "volatility is compressing, spreads are stabilizing, and investors appear "
    "to be rotating back into carry. The forward risk is that a volatility "
    "reversal quickly unwinds the move."
)
DEFAULT_PIPELINE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_openai_schema_v2_representative_220/narrative_pipeline_report.json"
)
DEFAULT_PIPELINE_NPZ = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_openai_schema_v2_representative_220/narrative_pipeline_arrays.npz"
)
DEFAULT_BRIDGE_ADAPTER = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_openai_schema_v2_representative_220/bridge_adapter.pt"
)
DEFAULT_CASEBOOK = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_risk_manager_casebook_representative_220/casebook.json"
)
DEFAULT_HARD_CASE_MANIFEST = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_risk_manager_casebook_representative_220/bridge_hard_case_manifest/"
    "bridge_hard_case_manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_story_smoke_fragile_risk_on"
)


class StoryMarketImplication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: str = Field(min_length=1)
    direction: str = Field(min_length=1)
    magnitude: str = Field(min_length=1)
    confidence: Literal["low", "medium", "high"] | str = "medium"
    evidence: list[str] = Field(default_factory=list)
    inferred: bool = False


class StoryGroundingWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    severity: Literal["warning", "error"] | str = "warning"
    message: str = Field(min_length=1)


class StoryGroundingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    narrative_frame: str = Field(min_length=1)
    cleaned_conditioning_text: str = Field(min_length=12)
    market_implications: list[StoryMarketImplication] = Field(default_factory=list)
    grounding_warnings: list[StoryGroundingWarning] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    critique: list[str] = Field(default_factory=list)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def build_story_grounding_messages(story: str) -> list[dict[str, str]]:
    """Build the OpenAI prompt for grounding a free-form risk-manager story."""

    system = (
        "You convert risk-manager scenario stories into grounded market "
        "conditioning instructions for a financial scenario generator. Extract "
        "only the market implications that are stated or strongly implied by "
        "the user's story. Do not invent news events, dates, policy actions, "
        "or causal facts. If a phrase is interpretive, mark it as a grounding "
        "warning instead of treating it as observed fact."
    )
    user = (
        "Return a StoryGroundingResult JSON object. Use market names such as "
        "SPX, VIX, BBB_OAS, AAA_OAS, US2Y, US10Y, USDJPY, DXY, GOLD, "
        "CRUDE_OIL, IV_SURFACE, and IV_SKEW when supported. Direction should "
        "be up/down/flat for prices, rates, vol, or FX, and wider/tighter/flat "
        "for credit spreads. Magnitude should be small/medium/large/flat. "
        "Set inferred=true for implications that follow from a regime phrase "
        "rather than a direct market mention. Keep generic phrases such as "
        "'carry', 'liquidity', or 'risk appetite' as narrative-frame evidence; "
        "do not map them to US2Y, US10Y, USDJPY, DXY, GOLD, or CRUDE_OIL unless "
        "the story names rates, FX, safe havens, dollar, yen, commodities, or "
        "the relevant asset directly.\n\n"
        f"Risk-manager story:\n{story}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def ground_story_with_openai(
    story: str,
    *,
    model: str,
    dotenv_path: str | Path = ".env",
    max_output_tokens: int = 1200,
) -> StoryGroundingResult:
    """Call OpenAI structured outputs to ground a story into market implications."""

    load_dotenv_key(dotenv_path)
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.parse(
        model=model,
        input=build_story_grounding_messages(story),
        text_format=StoryGroundingResult,
        max_output_tokens=int(max_output_tokens),
    )
    parsed = response.output_parsed
    if not isinstance(parsed, StoryGroundingResult):
        raise TypeError("OpenAI did not return a StoryGroundingResult")
    return parsed


def build_story_query_text(story: str, grounding: StoryGroundingResult) -> str:
    """Build the exact text sent to the text embedding model."""

    implication_lines = []
    for item in grounding.market_implications:
        inferred = " inferred" if bool(item.inferred) else ""
        evidence = "; ".join(item.evidence) if item.evidence else "no direct quote"
        implication_lines.append(
            f"{item.market}: {item.direction} {item.magnitude} "
            f"confidence={item.confidence}{inferred} evidence={evidence}"
        )
    warning_lines = [
        f"{item.severity} {item.code}: {item.message}"
        for item in grounding.grounding_warnings
    ]
    unsupported_lines = [
        f"unsupported: {claim}" for claim in grounding.unsupported_claims
    ]
    return "\n".join(
        [
            f"NARRATIVE: {story}",
            f"NARRATIVE_FRAME: {grounding.narrative_frame}",
            f"CLEANED_CONDITIONING_TEXT: {grounding.cleaned_conditioning_text}",
            "EXPLICIT_MARKET_IMPLICATIONS:",
            *implication_lines,
            "GROUNDING_WARNINGS:",
            *(warning_lines or ["none"]),
            "UNSUPPORTED_CLAIMS:",
            *(unsupported_lines or ["none"]),
        ]
    )


def _casebook_by_window(casebook: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(case["window_id"]): case
        for case in _as_list(casebook.get("cases"))
        if isinstance(case, dict) and case.get("window_id")
    }


def _nearest_memory_indices(
    query: np.ndarray,
    targets: np.ndarray,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    q = np.asarray(query, dtype=np.float32)
    target_norm = normalize_rows(np.asarray(targets, dtype=np.float32))
    q_norm = q / max(float(np.linalg.norm(q)), 1e-8)
    sims = target_norm @ q_norm
    order = np.argsort(-sims)[: int(top_k)]
    return [{"index": int(idx), "cosine": float(sims[idx])} for idx in order]


def _scenario_metrics_from_case(case: dict[str, Any]) -> dict[str, Any]:
    scores = _as_dict(_as_dict(case.get("scores")).get("narrative_generator_topk"))
    return {
        key: scores.get(key)
        for key in (
            "energy_score_improvement_vs_persistence",
            "ensemble_crps_improvement_vs_persistence",
            "coverage_80",
            "mean_path_mae_improvement_vs_persistence",
        )
        if key in scores
    }


def _market_direction_map(rows: list[Any]) -> dict[str, str]:
    directions: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        market = str(row.get("market", "")).upper()
        direction = str(row.get("direction", "")).lower()
        if market and direction:
            directions[market] = direction
    return directions


def score_implication_alignment(
    grounding: StoryGroundingResult,
    analogue_implications: list[Any],
) -> dict[str, Any]:
    """Compare extracted story implications against an analogue's market facts."""

    story_directions = {
        str(item.market).upper(): str(item.direction).lower()
        for item in grounding.market_implications
    }
    analogue_directions = _market_direction_map(analogue_implications)
    matched: list[str] = []
    mismatches: list[dict[str, str]] = []
    missing: list[str] = []
    for market, story_direction in story_directions.items():
        analogue_direction = analogue_directions.get(market)
        if analogue_direction is None:
            missing.append(market)
            continue
        if analogue_direction == story_direction:
            matched.append(market)
            continue
        mismatches.append(
            {
                "market": market,
                "story_direction": story_direction,
                "analogue_direction": analogue_direction,
            }
        )
    compared = len(matched) + len(mismatches)
    match_rate = float(len(matched) / compared) if compared else 0.0
    status = "pass" if compared and match_rate >= 0.67 else "warning"
    return {
        "status": status,
        "compared_markets": compared,
        "matched_markets": matched,
        "missing_markets": missing,
        "mismatches": mismatches,
        "match_rate": match_rate,
    }


def _enrich_analogue(
    row: dict[str, Any],
    bundles: list[dict[str, Any]],
    casebook: dict[str, Any],
    grounding: StoryGroundingResult,
) -> dict[str, Any]:
    bundle = bundles[int(row["index"])]
    case = _casebook_by_window(casebook).get(str(bundle.get("window_id")), {})
    narrative = ""
    narratives = _as_list(bundle.get("narratives"))
    if narratives and isinstance(narratives[0], dict):
        narrative = str(narratives[0].get("text", ""))
    enriched = {
        **row,
        "window_id": str(bundle.get("window_id", "")),
        "source_index": bundle.get("source_index"),
        "window_index": bundle.get("window_index"),
        "manifest_split": bundle.get("manifest_split"),
        "calendar": bundle.get("calendar", {}),
        "primary_narrative": narrative,
        "market_implications": _as_list(bundle.get("market_implications")),
    }
    enriched["implication_alignment"] = score_implication_alignment(
        grounding,
        _as_list(enriched.get("market_implications")),
    )
    if case:
        enriched["casebook_narrative"] = str(case.get("input_narrative", ""))
        enriched["casebook_observed_facts"] = str(case.get("observed_fact_tokens", ""))
        enriched["casebook_scores"] = _scenario_metrics_from_case(case)
    return enriched


def _subset_hits(
    analogues: list[dict[str, Any]], hard_case_manifest: dict[str, Any]
) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    window_ids = [str(row.get("window_id", "")) for row in analogues]
    for subset_name, raw_ids in _as_dict(hard_case_manifest.get("subsets")).items():
        subset_ids = {str(item) for item in _as_list(raw_ids)}
        overlap = [window_id for window_id in window_ids if window_id in subset_ids]
        hits[str(subset_name)] = overlap
    return hits


def assess_hard_case_gate(
    analogues: list[dict[str, Any]],
    hard_case_manifest: dict[str, Any],
    *,
    ood_warning: bool,
) -> dict[str, Any]:
    """Summarize whether retrieved analogues trigger hard-case demo warnings."""

    hits = _subset_hits(analogues, hard_case_manifest)
    if ood_warning:
        return {
            "status": "fail",
            "reason": "nearest historical analogue below similarity threshold",
            "subset_hits": hits,
        }
    blocking = set(hits.get("label_repair", [])) | set(
        hits.get("bridge_model_hard_case", [])
    )
    if blocking:
        return {
            "status": "fail",
            "reason": "retrieved analogue overlaps label-repair or bridge-model hard cases",
            "blocking_window_ids": sorted(blocking),
            "subset_hits": hits,
        }
    warning_hits = {
        name: ids
        for name, ids in hits.items()
        if ids
        and name
        in {
            "bridge_hard_case_validation",
            "mixed_regime_contrastive",
            "rank_metric_review",
        }
    }
    if warning_hits:
        return {
            "status": "warning",
            "reason": "retrieved analogue overlaps bridge hard-case review subsets",
            "subset_hits": hits,
        }
    return {
        "status": "pass",
        "reason": "nearest analogues are inside similarity threshold and do not overlap blocking hard-case subsets",
        "subset_hits": hits,
    }


def _relevance_status(diagnostics: dict[str, Any]) -> dict[str, Any]:
    if diagnostics.get("ood_warning"):
        return {
            "status": "fail",
            "reason": diagnostics.get("reason"),
        }
    top_cosine = diagnostics.get("top_cosine")
    top_gap = diagnostics.get("top_gap")
    if top_gap is not None and 0.0 < float(top_gap) < 0.01:
        return {
            "status": "warning",
            "reason": "nearest analogues are very close to each other; story maps to a broad regime cluster",
            "top_cosine": top_cosine,
            "top_gap": top_gap,
        }
    return {
        "status": "pass",
        "reason": "story maps to a nearby historical condition cluster",
        "top_cosine": top_cosine,
        "top_gap": top_gap,
    }


def build_story_smoke_report(
    *,
    story: str,
    grounding: StoryGroundingResult,
    query_text: str,
    query_condition: np.ndarray,
    memory_targets: np.ndarray,
    bundles: list[dict[str, Any]],
    casebook: dict[str, Any],
    hard_case_manifest: dict[str, Any],
    top_k: int,
    ood_threshold: float,
    scenario_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build the product-facing smoke-test report from a query condition."""

    analogues = _nearest_memory_indices(
        query_condition,
        memory_targets,
        top_k=int(top_k),
    )
    weighted = retrieval_weights(analogues, temperature=0.05)
    weight_by_index = {int(row["index"]): row["weight"] for row in weighted}
    enriched = []
    for row in analogues:
        enriched_row = _enrich_analogue(row, bundles, casebook, grounding)
        enriched_row["weight"] = float(weight_by_index.get(int(row["index"]), 0.0))
        enriched.append(enriched_row)
    diagnostics = retrieval_diagnostics(
        analogues,
        ood_threshold=float(ood_threshold),
    )
    hard_case_gate = assess_hard_case_gate(
        enriched,
        hard_case_manifest,
        ood_warning=bool(diagnostics.get("ood_warning")),
    )
    return {
        "title": "Risk Manager Story Smoke Test",
        "status": "ok",
        "story": story,
        "grounding": grounding.model_dump(),
        "query_text": query_text,
        "embedding_metadata": {
            "query_text_length": len(query_text),
            "query_text_line_count": query_text.count("\n") + 1,
        },
        "condition_diagnostics": {
            "condition_dim": int(np.asarray(query_condition).shape[-1]),
            "query_condition_norm": float(np.linalg.norm(query_condition)),
            **diagnostics,
        },
        "relevance": _relevance_status(diagnostics),
        "hard_case_gate": hard_case_gate,
        "historical_analogues": enriched,
        "generation": scenario_summary,
    }


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _short(text: str, limit: int = 180) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= int(limit):
        return compact
    return compact[: int(limit) - 3].rstrip() + "..."


def render_story_smoke_markdown(report: dict[str, Any]) -> str:
    """Render the smoke-test report as Markdown for demos/review."""

    grounding = _as_dict(report.get("grounding"))
    condition = _as_dict(report.get("condition_diagnostics"))
    relevance = _as_dict(report.get("relevance"))
    hard_case_gate = _as_dict(report.get("hard_case_gate"))
    lines = [
        "# Risk Manager Story Smoke Test",
        "",
        "## Story",
        "",
        str(report.get("story", "")),
        "",
        "## Extracted Market Implications",
        "",
        f"- Narrative frame: {grounding.get('narrative_frame', '')}",
    ]
    for item in _as_list(grounding.get("market_implications")):
        if not isinstance(item, dict):
            continue
        inferred = " inferred" if item.get("inferred") else ""
        lines.append(
            "- "
            f"{item.get('market')}: {item.get('direction')} {item.get('magnitude')} "
            f"confidence={item.get('confidence')}{inferred}"
        )
    lines.extend(["", "## Grounding Warnings", ""])
    warnings = _as_list(grounding.get("grounding_warnings"))
    if not warnings:
        lines.append("- none")
    for item in warnings:
        if isinstance(item, dict):
            lines.append(
                f"- `{item.get('severity')}` `{item.get('code')}`: {item.get('message')}"
            )
    lines.extend(
        [
            "",
            "## Condition Diagnostics",
            "",
            f"- Condition dimension: {condition.get('condition_dim')}",
            f"- Query condition norm: {_fmt_float(condition.get('query_condition_norm'))}",
            f"- Top analogue cosine: {_fmt_float(condition.get('top_cosine'))}",
            f"- Top analogue gap: {_fmt_float(condition.get('top_gap'))}",
            f"- Relevance status: `{relevance.get('status')}` - {relevance.get('reason')}",
            f"- Hard-case gate: `{hard_case_gate.get('status')}` - {hard_case_gate.get('reason')}",
            "",
            "## Historical Analogues",
            "",
            "| Rank | Window | Cosine | Weight | Implication Match | Split | Story Evidence |",
            "| ---: | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for rank, row in enumerate(_as_list(report.get("historical_analogues")), start=1):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(row.get("window_id", "")),
                    _fmt_float(row.get("cosine")),
                    _fmt_float(row.get("weight")),
                    _fmt_float(
                        _as_dict(row.get("implication_alignment")).get("match_rate")
                    ),
                    str(row.get("manifest_split", "")),
                    _short(
                        str(
                            row.get("casebook_narrative")
                            or row.get("primary_narrative")
                            or ""
                        )
                    ),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Scenario Summary", ""])
    generation = _as_dict(report.get("generation"))
    if not generation:
        lines.append("Scenario generator was not run for this smoke test.")
    else:
        lines.append(
            f"- Generated state shape: {generation.get('generated_state_shape', 'n/a')}"
        )
        lines.append(f"- Finite rate: {_fmt_float(generation.get('finite_rate'))}")
        lines.append("")
        lines.append("| Market | Mean Terminal Delta | P10 | P90 |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in _as_list(generation.get("terminal_delta_summary")):
            if not isinstance(row, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("market", "")),
                        _fmt_float(row.get("mean_terminal_delta")),
                        _fmt_float(row.get("p10")),
                        _fmt_float(row.get("p90")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def _load_bridge_adapter(
    path: str | Path,
    *,
    embedding_dim: int,
    condition_dim: int,
) -> NarrativeAdapter:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    adapter = NarrativeAdapter(
        int(embedding_dim),
        int(condition_dim),
    )
    adapter.load_state_dict(checkpoint["state_dict"])
    adapter.eval()
    return adapter


def _selected_generator_arrays(
    pipeline_report: dict[str, Any],
    *,
    checkpoint: str,
    device: torch.device,
) -> dict[str, Any]:
    from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (
        load_model,
    )

    args = argparse.Namespace(
        checkpoint=checkpoint,
        device=str(device),
        state_scope="joint38",
        eval_split="val",
        test_start=4511,
        val_size=441,
        max_windows=441,
        iv_count=25,
        clean_nonpositive_log_levels=True,
        positive_level_policy="reference_based",
        iv_transform="log_level",
        iv_lower_bound=1e-4,
        iv_upper_bound=1.0,
        scale_half_life=0.0,
        scale_floor=1e-4,
        center_mode="zero",
        drift_feature_mode="none",
    )
    model, payload = load_model(checkpoint, device)
    (
        history_level,
        history_norm,
        center,
        scale,
        drift_feature,
        history_raw,
        specs,
        _block,
    ) = _load_joint39_block(args, payload)
    selected = np.asarray(pipeline_report["window_indices"], dtype=np.int64)
    return {
        "model": model,
        "history_level": history_level[selected],
        "history_norm": history_norm[selected],
        "center": center[selected],
        "scale": scale[selected],
        "drift_feature": drift_feature[selected],
        "history_raw": history_raw[selected],
        "specs": specs,
        "spec_names": _spec_names(specs),
    }


def _run_retrieval_generator(
    *,
    pipeline_report: dict[str, Any],
    checkpoint: str,
    analogues: list[dict[str, Any]],
    samples: int,
    n_steps: int,
    chunk_size: int,
    temperature: float,
    device: str,
) -> dict[str, Any]:
    dev = torch.device(
        device if torch.cuda.is_available() or str(device) == "cpu" else "cpu"
    )
    arrays = _selected_generator_arrays(
        pipeline_report,
        checkpoint=checkpoint,
        device=dev,
    )
    sampled = sample_normal_generator_for_retrieved_analogues(
        arrays["model"],
        analogues,
        arrays["history_level"],
        arrays["history_norm"],
        arrays["center"],
        arrays["scale"],
        arrays["drift_feature"],
        n_samples=int(samples),
        n_steps=int(n_steps),
        chunk_size=int(chunk_size),
        temperature=float(temperature),
        device=dev,
    )
    retrieved_indices = sampled["indices"]
    generated_states = _reconstruct_states(
        arrays["history_raw"][retrieved_indices, -1, :],
        sampled["increments"],
        arrays["specs"],
    )
    return summarize_retrieval_generated_states(
        generated_states,
        arrays["history_raw"][retrieved_indices, -1, :],
        arrays["spec_names"],
    )


def _load_story_grounding(
    *,
    story: str,
    grounding_json: str | Path | None,
    model: str,
    dotenv_path: str | Path,
    max_output_tokens: int,
) -> StoryGroundingResult:
    if grounding_json:
        return StoryGroundingResult.model_validate(_load_json(grounding_json))
    return ground_story_with_openai(
        story,
        model=model,
        dotenv_path=dotenv_path,
        max_output_tokens=int(max_output_tokens),
    )


def run_story_smoke(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_report = _load_json(args.pipeline_report)
    pipeline_arrays = np.load(args.pipeline_npz)
    casebook = _load_json(args.casebook)
    hard_case_manifest = _load_json(args.hard_case_manifest)
    grounding = _load_story_grounding(
        story=args.story,
        grounding_json=args.grounding_json,
        model=args.grounding_model,
        dotenv_path=args.dotenv,
        max_output_tokens=int(args.grounding_max_output_tokens),
    )
    query_text = build_story_query_text(args.story, grounding)
    query_embedding = embed_texts_with_openai(
        [query_text],
        model=args.embedding_model,
        dotenv_path=args.dotenv,
        batch_size=1,
    )
    memory_targets = np.asarray(pipeline_arrays["memory_targets"], dtype=np.float32)
    adapter = _load_bridge_adapter(
        args.bridge_adapter,
        embedding_dim=int(query_embedding.shape[1]),
        condition_dim=int(memory_targets.shape[1]),
    )
    with torch.no_grad():
        query_condition = (
            adapter(torch.from_numpy(normalize_rows(query_embedding)).float())
            .cpu()
            .numpy()[0]
            .astype(np.float32)
        )
    analogue_rows = _nearest_memory_indices(
        query_condition,
        memory_targets,
        top_k=int(args.top_k),
    )
    scenario_summary: dict[str, Any] = {}
    if not bool(args.skip_generator):
        scenario_summary = _run_retrieval_generator(
            pipeline_report=pipeline_report,
            checkpoint=args.checkpoint or pipeline_report["checkpoint"],
            analogues=analogue_rows,
            samples=int(args.samples),
            n_steps=int(args.n_steps),
            chunk_size=int(args.chunk_size),
            temperature=float(args.temperature),
            device=args.device,
        )
    report = build_story_smoke_report(
        story=args.story,
        grounding=grounding,
        query_text=query_text,
        query_condition=query_condition,
        memory_targets=memory_targets,
        bundles=_as_list(pipeline_report.get("narrative_bundles")),
        casebook=casebook,
        hard_case_manifest=hard_case_manifest,
        top_k=int(args.top_k),
        ood_threshold=float(args.ood_threshold),
        scenario_summary=scenario_summary,
    )
    report["artifact_inputs"] = {
        "pipeline_report": str(args.pipeline_report),
        "pipeline_npz": str(args.pipeline_npz),
        "bridge_adapter": str(args.bridge_adapter),
        "casebook": str(args.casebook),
        "hard_case_manifest": str(args.hard_case_manifest),
    }
    report["embedding_metadata"].update(
        {
            "embedding_model": str(args.embedding_model),
            "embedding_dim": int(query_embedding.shape[1]),
            "grounding_model": str(args.grounding_model),
        }
    )
    output_dir = Path(args.output_dir)
    json_path = output_dir / "story_smoke_report.json"
    markdown_path = output_dir / "story_smoke_report.md"
    _write_json(json_path, report)
    _write_text(markdown_path, render_story_smoke_markdown(report))
    return {
        **report,
        "artifact_paths": {
            "json": str(json_path),
            "markdown": str(markdown_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--story", default=DEFAULT_STORY)
    parser.add_argument("--grounding-json")
    parser.add_argument("--grounding-model", default="gpt-5.4-mini")
    parser.add_argument("--grounding-max-output-tokens", type=int, default=1200)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--pipeline-report", default=DEFAULT_PIPELINE_REPORT)
    parser.add_argument("--pipeline-npz", default=DEFAULT_PIPELINE_NPZ)
    parser.add_argument("--bridge-adapter", default=DEFAULT_BRIDGE_ADAPTER)
    parser.add_argument("--casebook", default=DEFAULT_CASEBOOK)
    parser.add_argument("--hard-case-manifest", default=DEFAULT_HARD_CASE_MANIFEST)
    parser.add_argument("--checkpoint")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--ood-threshold", type=float, default=0.75)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-generator", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = run_story_smoke(args)
    print(
        json.dumps(
            {
                "json": report["artifact_paths"]["json"],
                "markdown": report["artifact_paths"]["markdown"],
                "top_analogue": report["historical_analogues"][0]["window_id"],
                "top_cosine": report["condition_diagnostics"]["top_cosine"],
                "relevance_status": report["relevance"]["status"],
                "hard_case_gate": report["hard_case_gate"]["status"],
                "generator_ran": bool(report["generation"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
