#!/usr/bin/env python
"""Run fixed-start rollout comparison for narrative support policies.

This is the rollout-level follow-up to the support-policy bake-off. It calls the
existing prefix-latent story-smoke generator, keeps the same fixed historical
start, and compares whether different support policies produce materially
different raw-factor and portfolio-tail distributions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_live_casebook import (  # noqa: E402
    default_casebook_stories,
)


DEFAULT_OUTPUT_ROOT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "fixed_start_rollout_policy_comparison_931a"
)
DEFAULT_BRIDGE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_full_906b_all_windows/bridge_eval_report.json"
)
DEFAULT_BRIDGE_ARRAYS = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_full_906b_all_windows/bridge_eval_arrays.npz"
)
DEFAULT_CONDITION_ROOT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_condition_only_full906b_914a"
)
STORY_SMOKE_SCRIPT = "experiments/backfill/block_ar/nl_prefix_latent_story_smoke.py"

CASE_SPECS = {
    "fragile_risk_on": {
        "condition_report_name": "fragile_risk_on_start18",
        "label": "Fragile risk-on",
        "color": "#1565C0",
    },
    "defensive_risk_off": {
        "condition_report_name": "defensive_risk_off_start18",
        "label": "Defensive risk-off",
        "color": "#C62828",
    },
    "commodity_inflation": {
        "condition_report_name": "commodity_inflation_start18",
        "label": "Commodity inflation",
        "color": "#EF6C00",
    },
    "dollar_liquidity": {
        "condition_report_name": "dollar_liquidity_start18",
        "label": "Dollar liquidity",
        "color": "#6A1B9A",
    },
    "rates_selloff": {
        "condition_report_name": "rates_selloff_start18",
        "label": "Rates selloff",
        "color": "#00838F",
    },
    "safe_haven_gold": {
        "condition_report_name": "safe_haven_gold_start18",
        "label": "Safe-haven gold",
        "color": "#2E7D32",
    },
}
CASE_NAMES = tuple(CASE_SPECS)
LEGACY_CASE_ALIASES = {
    str(spec["condition_report_name"]): name for name, spec in CASE_SPECS.items()
}
PROFESSIONAL_STORY_NAMES = {
    "fragile_risk_on": "fragile_risk_on_rebound",
    "defensive_risk_off": "defensive_risk_off_shock",
    "commodity_inflation": "commodity_inflation_pressure",
    "dollar_liquidity": "dollar_liquidity_squeeze",
    "rates_selloff": "rates_selloff_tightening_fear",
    "safe_haven_gold": "safe_haven_gold_bid",
}
DEFAULT_POLICY_NAMES = ("current_start_checked_gap30", "start_only_topk")

FACTOR_INDEX = {
    "SPX": 25,
    "VIX": 38,
    "DXY": 28,
    "Crude": 31,
    "US10Y": 33,
    "BBB_OAS": 35,
    "Gold": 37,
    "IV_ATM_1Y": 17,
}
SUMMARY_FACTORS = ("SPX", "VIX", "Crude", "US10Y", "BBB_OAS", "Gold", "IV_ATM_1Y")
PORTFOLIO_EXPOSURES = (
    {"market": "SPX", "index": 25, "sensitivity": 1.00},
    {"market": "VIX", "index": 38, "sensitivity": -0.55},
    {"market": "BBB_OAS", "index": 35, "sensitivity": -0.45},
    {"market": "US10Y", "index": 33, "sensitivity": -0.35},
    {"market": "DXY", "index": 28, "sensitivity": -0.25},
    {"market": "Crude", "index": 31, "sensitivity": 0.20},
    {"market": "Gold", "index": 37, "sensitivity": 0.15},
    {"market": "IV_ATM_1Y", "index": 17, "sensitivity": -0.30},
)


def normalize_case_name(case_name: str) -> str:
    """Return the public narrative case id, accepting legacy cache aliases."""

    raw = str(case_name)
    normalized = LEGACY_CASE_ALIASES.get(raw, raw)
    if normalized not in CASE_SPECS:
        raise ValueError(f"unknown case {case_name!r}")
    return normalized


def condition_report_case_name(case_name: str) -> str:
    return str(CASE_SPECS[normalize_case_name(case_name)]["condition_report_name"])


def public_case_label(case_name: str) -> str:
    return str(CASE_SPECS[normalize_case_name(case_name)]["label"])


def professional_story_for_case(case_name: str) -> str:
    case = normalize_case_name(case_name)
    story_name = PROFESSIONAL_STORY_NAMES[case]
    stories = {
        str(item["name"]): str(item["story"]) for item in default_casebook_stories()
    }
    if story_name not in stories:
        raise KeyError(f"professional story {story_name!r} missing from default deck")
    return stories[story_name]


@dataclass(frozen=True)
class RolloutPolicy:
    name: str
    memory_prior_mode: str
    start_distance_penalty: float
    implication_alignment_weight: float
    diverse_min_index_gap: int
    description: str
    rollout_mixture_mode: str = "component_prefix_mixture"
    response_preview_objective: str = "narrative_channels"
    memory_prior_temperature: float | None = None


POLICIES = {
    "current_start_checked_gap30": RolloutPolicy(
        name="current_start_checked_gap30",
        memory_prior_mode="diverse_topk_narrative_start_checked",
        start_distance_penalty=0.02,
        implication_alignment_weight=0.25,
        diverse_min_index_gap=30,
        description=(
            "Current selector: text-memory support, start-distance penalty, hard "
            "direction gate, and 30-window non-overlap."
        ),
    ),
    "narrative_first_hard_direction_gap30": RolloutPolicy(
        name="narrative_first_hard_direction_gap30",
        memory_prior_mode="diverse_topk_narrative_start_checked",
        start_distance_penalty=0.0,
        implication_alignment_weight=0.25,
        diverse_min_index_gap=30,
        description=(
            "Start-penalty ablation: removes start-distance ranking pressure "
            "while keeping hard direction consistency and 30-window non-overlap."
        ),
    ),
    "cohesive_support_gap30": RolloutPolicy(
        name="cohesive_support_gap30",
        memory_prior_mode="cohesive_topk_narrative_start_checked",
        start_distance_penalty=0.0,
        implication_alignment_weight=0.25,
        diverse_min_index_gap=30,
        description=(
            "Support-coherence selector: starts from the top direction-safe "
            "narrative hit and fills the support set with nearest latent-family "
            "members, preserving non-overlap."
        ),
    ),
    "cluster_family_support_gap30": RolloutPolicy(
        name="cluster_family_support_gap30",
        memory_prior_mode="cluster_family_narrative_start_checked",
        start_distance_penalty=0.0,
        implication_alignment_weight=0.25,
        diverse_min_index_gap=30,
        description=(
            "Support-family selector: chooses a coherent latent family with "
            "strong aggregate narrative score instead of mixing broad, weakly "
            "related support regimes."
        ),
    ),
    "kernel_similarity_support_gap30": RolloutPolicy(
        name="kernel_similarity_support_gap30",
        memory_prior_mode="kernel_topk_narrative_start_checked",
        start_distance_penalty=0.0,
        implication_alignment_weight=0.25,
        diverse_min_index_gap=30,
        memory_prior_temperature=0.05,
        description=(
            "Similarity-kernel selector: uses direction-safe top support but "
            "applies a low-temperature memory-similarity posterior so one "
            "coherent regime is not averaged away."
        ),
    ),
    "narrative_first_soft_direction_gap30": RolloutPolicy(
        name="narrative_first_soft_direction_gap30",
        memory_prior_mode="diverse_topk_combined",
        start_distance_penalty=0.0,
        implication_alignment_weight=0.25,
        diverse_min_index_gap=30,
        description=(
            "Narrative-first candidate: removes start-distance ranking pressure "
            "and uses direction as a soft score instead of a hard support gate."
        ),
    ),
    "start_only_topk": RolloutPolicy(
        name="start_only_topk",
        memory_prior_mode="soft_topk_start_only",
        start_distance_penalty=0.02,
        implication_alignment_weight=0.0,
        diverse_min_index_gap=0,
        description=(
            "Start-only null: support selected from starting-level similarity, "
            "with no narrative ranking contribution."
        ),
    ),
    "narrative_book_response_guard_gap30": RolloutPolicy(
        name="narrative_book_response_guard_gap30",
        memory_prior_mode="narrative_book_quality_guard_926b",
        start_distance_penalty=0.02,
        implication_alignment_weight=0.25,
        diverse_min_index_gap=30,
        description=(
            "Response-aware selector: chooses candidate support mixtures with "
            "train-only portfolio response priors blended by the narrative's "
            "grounded risk-channel books, while keeping hard direction checks "
            "and 30-window non-overlap."
        ),
    ),
    "portfolio_quality_guard_gap30": RolloutPolicy(
        name="portfolio_quality_guard_gap30",
        memory_prior_mode="portfolio_quality_guard_924e",
        start_distance_penalty=0.02,
        implication_alignment_weight=0.25,
        diverse_min_index_gap=30,
        description=(
            "Deployable response-aware selector: scores candidate support "
            "mixtures with train-only portfolio-response, CRPS, and energy "
            "support priors, while keeping hard direction checks and "
            "30-window non-overlap."
        ),
    ),
    "portfolio_direction_first_guard_gap30": RolloutPolicy(
        name="portfolio_direction_first_guard_gap30",
        memory_prior_mode="portfolio_direction_first_quality_guard_938a",
        start_distance_penalty=0.02,
        implication_alignment_weight=0.25,
        diverse_min_index_gap=30,
        description=(
            "Direction-first response-aware selector: filters candidate support "
            "mixtures through the final direction check before portfolio/quality "
            "scoring, then uses the best direction-safe candidate mixture as an "
            "atomic support set."
        ),
    ),
    "broad_replay_response_guard_gap30": RolloutPolicy(
        name="broad_replay_response_guard_gap30",
        memory_prior_mode="broad_replay_response_guard_940a",
        start_distance_penalty=0.02,
        implication_alignment_weight=0.25,
        diverse_min_index_gap=30,
        description=(
            "Broad-bank learned response-utility selector: scores direction-safe "
            "candidate support mixtures with support-level priors trained from "
            "broad support-bank historical response labels."
        ),
    ),
    "broad_response_preview_gap30": RolloutPolicy(
        name="broad_response_preview_gap30",
        memory_prior_mode="broad_replay_response_guard_940a",
        start_distance_penalty=0.02,
        implication_alignment_weight=0.25,
        diverse_min_index_gap=30,
        rollout_mixture_mode="response_preview_component_mixture",
        description=(
            "Broad-bank response-preview selector: first builds the broad "
            "direction-safe response-utility support pool, then runs small "
            "frozen-generator previews to reweight components by the actual "
            "narrative-channel rollout response before the final scenario deck."
        ),
    ),
    "broad_portfolio_response_preview_gap30": RolloutPolicy(
        name="broad_portfolio_response_preview_gap30",
        memory_prior_mode="broad_replay_response_guard_940a",
        start_distance_penalty=0.02,
        implication_alignment_weight=0.25,
        diverse_min_index_gap=30,
        rollout_mixture_mode="response_preview_component_mixture",
        response_preview_objective="factor_portfolio",
        description=(
            "Broad-bank portfolio-aware response-preview selector: first builds "
            "the broad direction-safe response-utility support pool, then "
            "reweights components with a preview objective that combines "
            "narrative-channel response and portfolio-tail activation."
        ),
    ),
    "broad_channel_portfolio_preview_gap30": RolloutPolicy(
        name="broad_channel_portfolio_preview_gap30",
        memory_prior_mode="broad_replay_response_guard_940a",
        start_distance_penalty=0.02,
        implication_alignment_weight=0.25,
        diverse_min_index_gap=30,
        rollout_mixture_mode="response_preview_component_mixture",
        response_preview_objective="channel_portfolio",
        description=(
            "Broad-bank channel-portfolio response-preview selector: starts "
            "from the broad direction-safe support pool, then reweights "
            "components by joint movement in the signed risk channels named by "
            "the narrative."
        ),
    ),
    "narrative_book_direction_first_guard_gap30": RolloutPolicy(
        name="narrative_book_direction_first_guard_gap30",
        memory_prior_mode="narrative_book_direction_first_quality_guard_938c",
        start_distance_penalty=0.02,
        implication_alignment_weight=0.25,
        diverse_min_index_gap=30,
        description=(
            "Direction-first narrative-channel selector: filters candidate "
            "support mixtures through the final direction check, scores them "
            "with the narrative's grounded risk-book priors, and uses the best "
            "direction-safe candidate mixture as an atomic support set."
        ),
    ),
    "response_preview_gap30": RolloutPolicy(
        name="response_preview_gap30",
        memory_prior_mode="diverse_topk_narrative_start_checked",
        start_distance_penalty=0.02,
        implication_alignment_weight=0.25,
        diverse_min_index_gap=30,
        rollout_mixture_mode="response_preview_component_mixture",
        description=(
            "Response-preview selector: starts from the current hard direction "
            "support pool, runs small frozen-generator previews for each support "
            "component, reweights components by narrative-channel response, and "
            "then runs the final component-preserving rollout."
        ),
    ),
}

PUBLIC_POLICY_LABELS = {
    "current_start_checked_gap30": "Hard direction\nstart-aware",
    "narrative_first_hard_direction_gap30": "Hard direction\nnarrative-first",
    "cohesive_support_gap30": "Cohesive\nsupport",
    "cluster_family_support_gap30": "Cluster-family\nsupport",
    "kernel_similarity_support_gap30": "Similarity-kernel\nsupport",
    "narrative_first_soft_direction_gap30": "Soft direction\nnarrative-first",
    "start_only_topk": "Start-only\nnull",
    "narrative_book_response_guard_gap30": "Response-aware\nbook guard",
    "portfolio_quality_guard_gap30": "Portfolio quality\nsupport guard",
    "portfolio_direction_first_guard_gap30": "Direction-first\nquality guard",
    "broad_replay_response_guard_gap30": "Broad learned\nresponse guard",
    "broad_response_preview_gap30": "Broad response\npreview guard",
    "broad_portfolio_response_preview_gap30": "Portfolio-aware\npreview guard",
    "broad_channel_portfolio_preview_gap30": "Channel-portfolio\npreview guard",
    "narrative_book_direction_first_guard_gap30": "Direction-first\nrisk book",
    "response_preview_gap30": "Response preview\nsupport weights",
}


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


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def build_story_smoke_command(
    *,
    case_name: str,
    policy_name: str,
    output_root: str | Path,
    samples: int,
    decoder_steps: int,
    seed: int,
    bridge_report: str | Path = DEFAULT_BRIDGE_REPORT,
    bridge_arrays: str | Path = DEFAULT_BRIDGE_ARRAYS,
    support_bank_report: str | Path | None = None,
    support_bank_arrays: str | Path | None = None,
    condition_root: str | Path = DEFAULT_CONDITION_ROOT,
    start_window_index: int = 18,
    memory_prior_top_k: int = 8,
    memory_prior_temperature: float = 0.2,
    diverse_max_pairwise_cosine: float = 0.95,
    quality_guard_candidate_pool_size: int = 12,
    quality_guard_mixture_size: int = 3,
    quality_guard_max_mixtures: int = 64,
    quality_guard_min_candidate_mixtures: int = 4,
    quality_guard_max_candidate_entropy_quantile: float = -1.0,
    quality_guard_min_support_weight_max_quantile: float = 0.25,
    quality_guard_probability_temperature: float = -1.0,
    response_preview_samples_per_component: int = 8,
    response_preview_alpha: float = 0.75,
    response_preview_temperature: float = 1.0,
    response_preview_blend: float = 1.0,
    device: str = "cuda",
    live_story_text: str | None = None,
    grounding_model: str = "gpt-5.4-mini",
    embedding_model: str = "text-embedding-3-small",
) -> list[str]:
    policy = POLICIES[str(policy_name)]
    case = normalize_case_name(case_name)
    output_dir = Path(output_root) / case / policy.name
    command = [
        sys.executable,
        STORY_SMOKE_SCRIPT,
        "--bridge-report",
        str(bridge_report),
        "--bridge-arrays",
        str(bridge_arrays),
        "--output-dir",
        str(output_dir),
        "--start-mode",
        "explicit_start_window",
        "--explicit-start-window-index",
        str(int(start_window_index)),
        "--memory-prior-mode",
        policy.memory_prior_mode,
        "--memory-prior-top-k",
        str(int(memory_prior_top_k)),
        "--memory-prior-temperature",
        str(
            float(
                policy.memory_prior_temperature
                if policy.memory_prior_temperature is not None
                else memory_prior_temperature
            )
        ),
        "--memory-prior-diverse-max-pairwise-cosine",
        str(float(diverse_max_pairwise_cosine)),
        "--memory-prior-diverse-min-index-gap",
        str(int(policy.diverse_min_index_gap)),
        "--memory-prior-quality-guard-candidate-pool-size",
        str(int(quality_guard_candidate_pool_size)),
        "--memory-prior-quality-guard-mixture-size",
        str(int(quality_guard_mixture_size)),
        "--memory-prior-quality-guard-max-mixtures",
        str(int(quality_guard_max_mixtures)),
        "--memory-prior-quality-guard-min-candidate-mixtures",
        str(int(quality_guard_min_candidate_mixtures)),
        "--memory-prior-quality-guard-max-candidate-entropy-quantile",
        str(float(quality_guard_max_candidate_entropy_quantile)),
        "--memory-prior-quality-guard-min-support-weight-max-quantile",
        str(float(quality_guard_min_support_weight_max_quantile)),
        "--memory-prior-quality-guard-probability-temperature",
        str(float(quality_guard_probability_temperature)),
        "--start-distance-penalty",
        str(float(policy.start_distance_penalty)),
        "--implication-alignment-weight",
        str(float(policy.implication_alignment_weight)),
        "--prefix-prior-mode",
        "decoder",
        "--rollout-mixture-mode",
        str(policy.rollout_mixture_mode),
        "--response-preview-samples-per-component",
        str(int(response_preview_samples_per_component)),
        "--response-preview-alpha",
        str(float(response_preview_alpha)),
        "--response-preview-temperature",
        str(float(response_preview_temperature)),
        "--response-preview-blend",
        str(float(response_preview_blend)),
        "--response-preview-objective",
        str(policy.response_preview_objective),
        "--samples",
        str(int(samples)),
        "--steps",
        str(int(decoder_steps)),
        "--seed",
        str(int(seed)),
        "--rollout-seed",
        str(int(seed)),
        "--device",
        str(device),
    ]
    if live_story_text:
        command.extend(
            [
                "--live-story",
                "--story",
                str(live_story_text),
                "--grounding-model",
                str(grounding_model),
                "--embedding-model",
                str(embedding_model),
            ]
        )
    else:
        condition_report = (
            Path(condition_root)
            / condition_report_case_name(case)
            / "condition_only_report.json"
        )
        command.extend(["--condition-report", str(condition_report)])
    if support_bank_report or support_bank_arrays:
        if not support_bank_report or not support_bank_arrays:
            raise ValueError(
                "support_bank_report and support_bank_arrays must be provided together"
            )
        command.extend(
            [
                "--support-bank-report",
                str(support_bank_report),
                "--support-bank-arrays",
                str(support_bank_arrays),
            ]
        )
    return command


def _run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _case_output_dir(output_root: str | Path, case_name: str, policy_name: str) -> Path:
    case = normalize_case_name(case_name)
    primary = Path(output_root) / case / str(policy_name)
    if primary.exists():
        return primary
    legacy = Path(output_root) / condition_report_case_name(case) / str(policy_name)
    if legacy.exists():
        return legacy
    return primary


def _case_report_path(
    output_root: str | Path, case_name: str, policy_name: str
) -> Path:
    return _case_output_dir(output_root, case_name, policy_name) / (
        "prefix_latent_story_smoke_report.json"
    )


def _case_arrays_path(
    output_root: str | Path, case_name: str, policy_name: str
) -> Path:
    return _case_output_dir(output_root, case_name, policy_name) / (
        "prefix_latent_story_smoke_arrays.npz"
    )


def _operational_variant_index(report: dict[str, Any], arrays: dict[str, Any]) -> int:
    cached = report.get("cached_query", {})
    if isinstance(cached, dict):
        raw = cached.get("operational_memory_prior_variant_index")
        if raw is not None:
            idx = int(raw)
            states = np.asarray(arrays["generated_states"])
            if 0 <= idx < states.shape[0]:
                return idx
    return 0


def _support_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    cached = report.get("cached_query", {})
    prior = cached.get("memory_prior", {}) if isinstance(cached, dict) else {}
    if not isinstance(prior, dict):
        return []
    weights = prior.get("weights", [])
    details = prior.get("candidate_details", [])
    rows = []
    for pos, row in enumerate(details if isinstance(details, list) else []):
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "rank": int(pos + 1),
                "window_index": int(row.get("window_index", -1)),
                "window_id": str(row.get("window_id", "")),
                "history_end_date": str(row.get("history_end_date", "")),
                "weight": (
                    float(weights[pos])
                    if isinstance(weights, list) and pos < len(weights)
                    else 0.0
                ),
                "memory_cosine": float(row.get("memory_support_cosine", 0.0)),
                "start_distance_z": float(row.get("start_distance_z", 0.0)),
                "direction_mismatches": int(
                    row.get("recent_prefix_mismatches", 0) or 0
                ),
                "direction_checked": int(row.get("recent_prefix_checked", 0) or 0),
            }
        )
    return rows


def _support_jaccard(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    left_set = {int(row["window_index"]) for row in left}
    right_set = {int(row["window_index"]) for row in right}
    return float(len(left_set & right_set) / max(len(left_set | right_set), 1))


def _ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    a = np.sort(np.asarray(left, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(right, dtype=np.float64).reshape(-1))
    if a.size == 0 or b.size == 0:
        return 0.0
    grid = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, grid, side="right") / float(a.size)
    cdf_b = np.searchsorted(b, grid, side="right") / float(b.size)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _terminal_summary(
    states: np.ndarray,
    factors: dict[str, int],
) -> dict[str, dict[str, float]]:
    arr = np.asarray(states, dtype=np.float64)
    terminal = arr[:, -1, :]
    out: dict[str, dict[str, float]] = {}
    for name, idx in factors.items():
        values = terminal[:, int(idx)]
        out[name] = {
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return out


def _portfolio_pnl(states: np.ndarray, start: np.ndarray) -> np.ndarray:
    arr = np.asarray(states, dtype=np.float64)
    start_arr = np.asarray(start, dtype=np.float64).reshape(-1)
    pnl_terms = []
    for exposure in PORTFOLIO_EXPOSURES:
        idx = int(exposure["index"])
        scale = max(abs(float(start_arr[idx])), 1.0)
        pnl_terms.append(
            (arr[:, :, idx] - float(start_arr[idx]))
            / scale
            * float(exposure["sensitivity"])
            * 100.0
        )
    return np.sum(np.stack(pnl_terms, axis=-1), axis=-1)


def _portfolio_stats(states: np.ndarray, start: np.ndarray) -> dict[str, float]:
    pnl = _portfolio_pnl(states, start)
    terminal = pnl[:, -1]
    q05 = float(np.percentile(terminal, 5))
    return {
        "terminal_p10": float(np.percentile(terminal, 10)),
        "terminal_p50": float(np.percentile(terminal, 50)),
        "terminal_p90": float(np.percentile(terminal, 90)),
        "terminal_var95_loss": float(-q05),
        "terminal_expected_shortfall95_loss": float(
            -np.mean(terminal[terminal <= q05])
        ),
        "terminal_std": float(np.std(terminal)),
    }


def _direction_status(report: dict[str, Any]) -> str:
    prior = (
        report.get("cached_query", {}).get("memory_prior", {})
        if isinstance(report.get("cached_query", {}), dict)
        else {}
    )
    check = prior.get("direction_check", {}) if isinstance(prior, dict) else {}
    return str(check.get("status", "")) if isinstance(check, dict) else ""


def load_case_result(
    *,
    output_root: str | Path,
    case_name: str,
    policy_name: str,
) -> dict[str, Any]:
    report_path = _case_report_path(output_root, case_name, policy_name)
    arrays_path = _case_arrays_path(output_root, case_name, policy_name)
    report = _load_json(report_path)
    arrays = dict(np.load(arrays_path))
    op_idx = _operational_variant_index(report, arrays)
    states = np.asarray(arrays["generated_states"], dtype=np.float32)[op_idx]
    start = np.asarray(arrays["requested_raw"], dtype=np.float32)[op_idx]
    support = _support_rows(report)
    return {
        "case": normalize_case_name(case_name),
        "case_label": public_case_label(case_name),
        "policy": str(policy_name),
        "report_path": str(report_path),
        "arrays_path": str(arrays_path),
        "generated_shape": [int(v) for v in states.shape],
        "start": start,
        "states": states,
        "support": support,
        "support_count": int(len(support)),
        "direction_status": _direction_status(report),
        "terminal_raw_levels": _terminal_summary(
            states, {name: FACTOR_INDEX[name] for name in SUMMARY_FACTORS}
        ),
        "portfolio_stats": _portfolio_stats(states, start),
    }


def _pairwise_case_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for i, left in enumerate(case_results):
        for right in case_results[i + 1 :]:
            factor_ks = {}
            for factor in SUMMARY_FACTORS:
                idx = FACTOR_INDEX[factor]
                factor_ks[factor] = _ks_statistic(
                    left["states"][:, -1, idx],
                    right["states"][:, -1, idx],
                )
            pnl_left = _portfolio_pnl(left["states"], left["start"])[:, -1]
            pnl_right = _portfolio_pnl(right["states"], right["start"])[:, -1]
            rows.append(
                {
                    "left": str(left["case"]),
                    "right": str(right["case"]),
                    "support_jaccard": _support_jaccard(
                        left["support"], right["support"]
                    ),
                    "mean_factor_terminal_ks": float(np.mean(list(factor_ks.values()))),
                    "factor_terminal_ks": factor_ks,
                    "portfolio_terminal_ks": _ks_statistic(pnl_left, pnl_right),
                }
            )
    return {
        "pair_count": int(len(rows)),
        "mean_support_jaccard": (
            float(np.mean([r["support_jaccard"] for r in rows])) if rows else 0.0
        ),
        "mean_factor_terminal_ks": (
            float(np.mean([r["mean_factor_terminal_ks"] for r in rows]))
            if rows
            else 0.0
        ),
        "mean_portfolio_terminal_ks": (
            float(np.mean([r["portfolio_terminal_ks"] for r in rows])) if rows else 0.0
        ),
        "rows": rows,
    }


def summarize_outputs(
    *,
    output_root: str | Path,
    cases: tuple[str, ...] = CASE_NAMES,
    policies: tuple[str, ...] = tuple(POLICIES),
) -> dict[str, Any]:
    policy_results = []
    normalized_cases = tuple(normalize_case_name(case) for case in cases)
    for policy_name in policies:
        case_results = [
            load_case_result(
                output_root=output_root,
                case_name=case_name,
                policy_name=policy_name,
            )
            for case_name in normalized_cases
        ]
        support_counts = [int(case["support_count"]) for case in case_results]
        direction_counts = Counter(
            str(case["direction_status"]) for case in case_results
        )
        portfolio_var95 = [
            float(case["portfolio_stats"]["terminal_var95_loss"])
            for case in case_results
        ]
        portfolio_es95 = [
            float(case["portfolio_stats"]["terminal_expected_shortfall95_loss"])
            for case in case_results
        ]
        start_stack = np.stack([case["start"] for case in case_results], axis=0)
        policy_results.append(
            {
                "policy": policy_name,
                "description": POLICIES[policy_name].description,
                "case_count": int(len(case_results)),
                "support_count_mean": float(np.mean(support_counts)),
                "support_count_min": int(np.min(support_counts)),
                "support_count_max": int(np.max(support_counts)),
                "direction_status_counts": dict(sorted(direction_counts.items())),
                "max_abs_start_difference": float(
                    np.max(np.abs(start_stack - start_stack[0:1]))
                ),
                "portfolio_var95_loss_range": float(
                    np.max(portfolio_var95) - np.min(portfolio_var95)
                ),
                "portfolio_es95_loss_range": float(
                    np.max(portfolio_es95) - np.min(portfolio_es95)
                ),
                "pairwise": _pairwise_case_metrics(case_results),
                "cases": [
                    {
                        key: value
                        for key, value in case.items()
                        if key not in {"states", "start"}
                    }
                    for case in case_results
                ],
            }
        )
    return {
        "status": "ok",
        "scope_note": (
            "Fixed-start rollout-level comparison. All policies use the same "
            "full 906b support bank, same fixed start, and component-preserving "
            "SNI rollout."
        ),
        "output_root": str(output_root),
        "cases": list(normalized_cases),
        "policies": policy_results,
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Fixed-Start Rollout Policy Comparison",
        "",
        str(summary["scope_note"]),
        "",
        "| Policy | Mean supports | Direction statuses | Start diff | Mean support Jaccard | Mean factor KS | Mean portfolio KS | VaR95 range | ES95 range |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in summary["policies"]:
        pairwise = policy["pairwise"]
        lines.append(
            "| {policy} | {support:.2f} | `{direction}` | {start:.3g} | "
            "{support_j:.3f} | {factor_ks:.3f} | {portfolio_ks:.3f} | "
            "{var_range:.3f} | {es_range:.3f} |".format(
                policy=policy["policy"],
                support=float(policy["support_count_mean"]),
                direction=policy["direction_status_counts"],
                start=float(policy["max_abs_start_difference"]),
                support_j=float(pairwise["mean_support_jaccard"]),
                factor_ks=float(pairwise["mean_factor_terminal_ks"]),
                portfolio_ks=float(pairwise["mean_portfolio_terminal_ks"]),
                var_range=float(policy["portfolio_var95_loss_range"]),
                es_range=float(policy["portfolio_es95_loss_range"]),
            )
        )
    lines.append("")
    for policy in summary["policies"]:
        lines.extend([f"## {policy['policy']}", "", policy["description"], ""])
        lines.append(
            "| Case | Supports | Direction | Portfolio VaR95 loss | Portfolio ES95 loss | Top support |"
        )
        lines.append("| --- | ---: | --- | ---: | ---: | --- |")
        for case in policy["cases"]:
            top = "; ".join(
                f"{row['history_end_date']} w={row['weight']:.2f}"
                for row in case["support"][:4]
            )
            stats = case["portfolio_stats"]
            lines.append(
                f"| {case.get('case_label', case['case'])} | {case['support_count']} | "
                f"`{case['direction_status']}` | "
                f"{stats['terminal_var95_loss']:.3f} | "
                f"{stats['terminal_expected_shortfall95_loss']:.3f} | {top} |"
            )
        lines.append("")
    return "\n".join(lines)


def plot_rollout_policy_comparison(
    *,
    output_root: str | Path,
    cases: tuple[str, ...],
    policies: tuple[str, ...],
    factor_output: str | Path,
    portfolio_output: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    normalized_cases = tuple(normalize_case_name(case) for case in cases)
    factors = SUMMARY_FACTORS
    days = np.arange(0, 31)
    fig, axes = plt.subplots(
        len(policies),
        len(factors),
        figsize=(3.9 * len(factors), 3.3 * len(policies)),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(
        "Fixed-start raw-level factor fans by support policy",
        fontsize=14,
        fontweight="bold",
    )
    for row_idx, policy_name in enumerate(policies):
        for col_idx, factor in enumerate(factors):
            ax = axes[row_idx, col_idx]
            idx = FACTOR_INDEX[factor]
            for case_name in normalized_cases:
                result = load_case_result(
                    output_root=output_root,
                    case_name=case_name,
                    policy_name=policy_name,
                )
                states = np.asarray(result["states"], dtype=np.float64)
                start = float(np.asarray(result["start"], dtype=np.float64)[idx])
                paths = np.concatenate(
                    [
                        np.full((states.shape[0], 1), start, dtype=np.float64),
                        states[:, :, idx],
                    ],
                    axis=1,
                )
                q10, q50, q90 = np.percentile(paths, [10, 50, 90], axis=0)
                color = str(CASE_SPECS[case_name]["color"])
                ax.fill_between(days, q10, q90, color=color, alpha=0.045)
                ax.plot(
                    days,
                    q50,
                    color=color,
                    linewidth=1.4,
                    label=(
                        public_case_label(case_name)
                        if row_idx == 0 and col_idx == 0
                        else None
                    ),
                )
            if row_idx == 0:
                ax.set_title(factor, fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(
                    PUBLIC_POLICY_LABELS.get(policy_name, policy_name), fontsize=8
                )
            ax.grid(alpha=0.15)
            ax.set_xlim(0, 30)
    axes[-1, 1].set_xlabel("Forward day")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    Path(factor_output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(factor_output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        len(policies),
        figsize=(4.6 * len(policies), 4.8),
        sharey=True,
        squeeze=False,
    )
    fig.suptitle(
        "Portfolio terminal distribution by narrative and support policy",
        fontsize=14,
        fontweight="bold",
    )
    for col_idx, policy_name in enumerate(policies):
        ax = axes[0, col_idx]
        data = []
        tick_labels = []
        box_colors = []
        for case_name in normalized_cases:
            result = load_case_result(
                output_root=output_root,
                case_name=case_name,
                policy_name=policy_name,
            )
            pnl = _portfolio_pnl(result["states"], result["start"])[:, -1]
            data.append(pnl)
            tick_labels.append(public_case_label(case_name).replace(" ", "\n"))
            box_colors.append(str(CASE_SPECS[case_name]["color"]))
        parts = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, color in zip(parts["boxes"], box_colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.28)
            patch.set_edgecolor(color)
        for median in parts["medians"]:
            median.set_color("#212121")
            median.set_linewidth(1.2)
        ax.axhline(0.0, color="#9E9E9E", linestyle=":", linewidth=0.9)
        ax.set_title(PUBLIC_POLICY_LABELS.get(policy_name, policy_name), fontsize=9)
        ax.set_xticks(np.arange(1, len(tick_labels) + 1))
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=0)
        ax.grid(axis="y", alpha=0.15)
        if col_idx == 0:
            ax.set_ylabel("Terminal portfolio P&L units")
    Path(portfolio_output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(portfolio_output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    cases = tuple(
        CASE_NAMES
        if not args.case
        else [normalize_case_name(case) for case in args.case]
    )
    policies = tuple(DEFAULT_POLICY_NAMES if not args.policy else args.policy)
    for policy in policies:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}")
    if not bool(args.summarize_only):
        for case_name in cases:
            for policy_name in policies:
                report_path = _case_report_path(output_root, case_name, policy_name)
                if report_path.exists() and not bool(args.force):
                    print(f"[reuse] {case_name} / {policy_name}: {report_path}")
                    continue
                command = build_story_smoke_command(
                    case_name=case_name,
                    policy_name=policy_name,
                    output_root=output_root,
                    samples=int(args.samples),
                    decoder_steps=int(args.decoder_steps),
                    seed=int(args.seed),
                    bridge_report=args.bridge_report,
                    bridge_arrays=args.bridge_arrays,
                    support_bank_report=args.support_bank_report,
                    support_bank_arrays=args.support_bank_arrays,
                    condition_root=args.condition_root,
                    start_window_index=int(args.start_window_index),
                    memory_prior_top_k=int(args.memory_prior_top_k),
                    memory_prior_temperature=float(args.memory_prior_temperature),
                    diverse_max_pairwise_cosine=float(args.diverse_max_pairwise_cosine),
                    quality_guard_candidate_pool_size=int(
                        args.quality_guard_candidate_pool_size
                    ),
                    quality_guard_mixture_size=int(args.quality_guard_mixture_size),
                    quality_guard_max_mixtures=int(args.quality_guard_max_mixtures),
                    quality_guard_min_candidate_mixtures=int(
                        args.quality_guard_min_candidate_mixtures
                    ),
                    quality_guard_max_candidate_entropy_quantile=float(
                        args.quality_guard_max_candidate_entropy_quantile
                    ),
                    quality_guard_min_support_weight_max_quantile=float(
                        args.quality_guard_min_support_weight_max_quantile
                    ),
                    quality_guard_probability_temperature=float(
                        args.quality_guard_probability_temperature
                    ),
                    response_preview_samples_per_component=int(
                        args.response_preview_samples_per_component
                    ),
                    response_preview_alpha=float(args.response_preview_alpha),
                    response_preview_temperature=float(
                        args.response_preview_temperature
                    ),
                    response_preview_blend=float(args.response_preview_blend),
                    device=str(args.device),
                    live_story_text=(
                        professional_story_for_case(case_name)
                        if bool(args.use_professional_story_deck)
                        else None
                    ),
                    grounding_model=str(args.grounding_model),
                    embedding_model=str(args.embedding_model),
                )
                if bool(args.dry_run):
                    print(" ".join(command))
                else:
                    print(f"[run] {case_name} / {policy_name}", flush=True)
                    _run_command(command)
    if bool(args.dry_run):
        return {"status": "dry_run", "output_root": str(output_root)}
    summary = summarize_outputs(output_root=output_root, cases=cases, policies=policies)
    summary["artifact_paths"] = {
        "json": str(output_root / "fixed_start_rollout_policy_comparison.json"),
        "markdown": str(output_root / "fixed_start_rollout_policy_comparison.md"),
        "factor_fans": str(output_root / "fixed_start_rollout_factor_fans.png"),
        "portfolio_tail": str(output_root / "fixed_start_rollout_portfolio_tail.png"),
    }
    summary["run_config"] = {
        "samples": int(args.samples),
        "decoder_steps": int(args.decoder_steps),
        "seed": int(args.seed),
        "start_window_index": int(args.start_window_index),
        "response_preview_samples_per_component": int(
            args.response_preview_samples_per_component
        ),
        "quality_guard_max_candidate_entropy_quantile": float(
            args.quality_guard_max_candidate_entropy_quantile
        ),
        "quality_guard_min_support_weight_max_quantile": float(
            args.quality_guard_min_support_weight_max_quantile
        ),
        "quality_guard_probability_temperature": float(
            args.quality_guard_probability_temperature
        ),
        "response_preview_alpha": float(args.response_preview_alpha),
        "response_preview_temperature": float(args.response_preview_temperature),
        "response_preview_blend": float(args.response_preview_blend),
        "bridge_report": str(args.bridge_report),
        "bridge_arrays": str(args.bridge_arrays),
        "support_bank_report": (
            None if args.support_bank_report is None else str(args.support_bank_report)
        ),
        "support_bank_arrays": (
            None if args.support_bank_arrays is None else str(args.support_bank_arrays)
        ),
        "condition_root": str(args.condition_root),
        "use_professional_story_deck": bool(args.use_professional_story_deck),
        "grounding_model": str(args.grounding_model),
        "embedding_model": str(args.embedding_model),
    }
    if not bool(args.no_plots):
        plot_rollout_policy_comparison(
            output_root=output_root,
            cases=cases,
            policies=policies,
            factor_output=summary["artifact_paths"]["factor_fans"],
            portfolio_output=summary["artifact_paths"]["portfolio_tail"],
        )
    _write_json(summary["artifact_paths"]["json"], summary)
    _write_text(summary["artifact_paths"]["markdown"], _render_markdown(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bridge-report", default=DEFAULT_BRIDGE_REPORT)
    parser.add_argument("--bridge-arrays", default=DEFAULT_BRIDGE_ARRAYS)
    parser.add_argument("--support-bank-report")
    parser.add_argument("--support-bank-arrays")
    parser.add_argument("--condition-root", default=DEFAULT_CONDITION_ROOT)
    parser.add_argument("--start-window-index", type=int, default=18)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--decoder-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=931)
    parser.add_argument("--memory-prior-top-k", type=int, default=8)
    parser.add_argument("--memory-prior-temperature", type=float, default=0.2)
    parser.add_argument("--diverse-max-pairwise-cosine", type=float, default=0.95)
    parser.add_argument("--quality-guard-candidate-pool-size", type=int, default=12)
    parser.add_argument("--quality-guard-mixture-size", type=int, default=3)
    parser.add_argument("--quality-guard-max-mixtures", type=int, default=64)
    parser.add_argument("--quality-guard-min-candidate-mixtures", type=int, default=4)
    parser.add_argument(
        "--quality-guard-max-candidate-entropy-quantile",
        type=float,
        default=-1.0,
        help="Negative disables the train-set entropy gate.",
    )
    parser.add_argument(
        "--quality-guard-min-support-weight-max-quantile",
        type=float,
        default=0.25,
        help=(
            "Train-set support-concentration quantile gate. Negative disables "
            "the support-max fallback for research probes."
        ),
    )
    parser.add_argument(
        "--quality-guard-probability-temperature",
        type=float,
        default=-1.0,
        help=(
            "Optional live support-policy probability temperature override. "
            "Negative uses the policy context default."
        ),
    )
    parser.add_argument("--response-preview-samples-per-component", type=int, default=8)
    parser.add_argument("--response-preview-alpha", type=float, default=0.75)
    parser.add_argument("--response-preview-temperature", type=float, default=1.0)
    parser.add_argument("--response-preview-blend", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--use-professional-story-deck",
        action="store_true",
        help=(
            "Use the validated professional risk-manager story deck directly "
            "instead of replaying older condition-only reports."
        ),
    )
    parser.add_argument("--grounding-model", default="gpt-5.4-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--case", action="append")
    parser.add_argument("--policy", action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    summary = run_comparison(args)
    compact = {
        "status": summary["status"],
        "output_root": summary.get("output_root", args.output_root),
        "artifact_paths": summary.get("artifact_paths", {}),
        "policies": {
            row["policy"]: {
                "mean_support_count": row["support_count_mean"],
                "mean_support_jaccard": row["pairwise"]["mean_support_jaccard"],
                "mean_factor_terminal_ks": row["pairwise"]["mean_factor_terminal_ks"],
                "mean_portfolio_terminal_ks": row["pairwise"][
                    "mean_portfolio_terminal_ks"
                ],
                "portfolio_var95_loss_range": row["portfolio_var95_loss_range"],
                "direction_status_counts": row["direction_status_counts"],
            }
            for row in summary.get("policies", [])
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
