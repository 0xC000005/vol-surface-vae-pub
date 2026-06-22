#!/usr/bin/env python
"""Local Gradio demo for narrative-conditioned scenario generation."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_risk_manager_story_smoke import (  # noqa: E402
    DEFAULT_BRIDGE_ADAPTER,
    DEFAULT_CASEBOOK,
    DEFAULT_HARD_CASE_MANIFEST,
    DEFAULT_PIPELINE_NPZ,
    DEFAULT_PIPELINE_REPORT,
    DEFAULT_STORY,
    path_quantiles_for_generated_states,
    render_story_smoke_markdown,
    run_story_smoke,
)
from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    DEFAULT_BRIDGE_ARRAYS as _LEGACY_PREFIX_BRIDGE_ARRAYS,
    DEFAULT_CHECKPOINT as DEFAULT_PREFIX_CHECKPOINT,
    DEFAULT_BRIDGE_REPORT as _LEGACY_PREFIX_BRIDGE_REPORT,
    run_prefix_latent_story_smoke,
    window_metadata_by_bridge_local_index,
)
from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    load_model,
)
from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    build_val_block,
)
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    summarize_retrieval_generated_states,
    _spec_names,
)
from experiments.backfill.block_ar.nl_prefix_latent_oracle_autoencoder import (  # noqa: E402
    selected_bridge_window_indices,
)
from experiments.backfill.block_ar.nl_prefix_latent_condition_only_report import (  # noqa: E402
    run_condition_only_report,
)
from experiments.backfill.block_ar.nl_prefix_latent_live_casebook import (  # noqa: E402
    default_casebook_stories,
)
from experiments.backfill.block_ar.nl_prefix_latent_hard_case_decomposition import (  # noqa: E402
    decompose_report,
    production_decision,
)
from experiments.backfill.block_ar.nl_prefix_latent_temporal_grounding_testflight import (  # noqa: E402
    PROMPT_VERSION as CONDITION_ONLY_PROMPT_VERSION,
    condition_query_text_from_grounding,
    ground_condition_only_story_with_openai,
    split_story_for_conditioning,
    validate_condition_only_grounding_result,
)
from experiments.backfill.block_ar.nl_prefix_latent_run_record import (  # noqa: E402
    write_prefix_run_record,
)
from experiments.backfill.block_ar.nl_narrative_ensemble_calibration import (  # noqa: E402
    apply_directional_delta_calibration,
    direction_vector_from_grounding,
    _support_evidence_gate_from_report,
)
from experiments.backfill.block_ar.nl_component_pooling_diagnostic import (  # noqa: E402
    component_slices_for_variant,
)
from experiments.backfill.block_ar.nl_sparse_component_family_view import (  # noqa: E402
    select_sparse_components,
)


DEFAULT_APP_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_story_gradio_demo"
)
DEFAULT_PREFIX_VALIDATION_GATE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_validation_gate_790b/"
    "prefix_latent_validation_gate_report.json"
)
DEFAULT_PREFIX_APP_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_story_gradio_demo/prefix_latent_live_smoke"
)
DEFAULT_PREFIX_SUPPORT_BANK_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_report.json"
)
DEFAULT_PREFIX_SUPPORT_BANK_ARRAYS = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
)
# ---------------------------------------------------------------------------
# RETRIEVAL QUERY BRIDGE — clean legacy-oracle backend (verified 2026-06-16)
# ---------------------------------------------------------------------------
# The 14x14 projected-memory bridge (990f/991a_seed1) was trained on the contaminated 982g
# corpus (joint39 factor-mapping bug: USDJPY→col 29, AAA_OAS→col 36 instead of canonical 27/34
# per nl_joint39_anchor_map.py) and is NOT used. Crucially, the clean-restamp 14x14 bridge —
# retrained on the CLEAN bank (stride5_14x14_retrieval_training_clean_restamp_20260615) — was
# tried and FAILED its fit gate (heldout recall@10 ~0.007 vs 0.10; the 992b exact-window
# information ceiling, reproduced on clean data). So there is no trained text bridge that beats
# the default; this is not an "interim hold pending retrain" — the retrain was done and lost.
#
# CLEAN BACKEND IN USE: the legacy oracle-era query bridge
# (manifest_bridge_eval_openai_schema_v2_representative_220, 1536→128, built from val/test window
# embeddings via the 734a encoder — it carries NO contaminated episode-card text). It projects the
# user's narrative embedding into the 734a 128-d space; retrieval + scenario generation then run
# over the clean 939a numeric support bank (top3/90 nearest-similar). This IS narrative-conditioned
# (the query reflects the user's narrative) — it is NOT a start-only fallback. Both the query bridge
# and the 939a targets are 734a-encoded and clean; the contamination affected only TEXT authoring,
# not the numeric encoder outputs used for retrieval.
DEFAULT_PREFIX_BRIDGE_REPORT: str = _LEGACY_PREFIX_BRIDGE_REPORT  # legacy clean oracle bridge (query projection / val-block export)
DEFAULT_PREFIX_BRIDGE_ARRAYS: str = _LEGACY_PREFIX_BRIDGE_ARRAYS  # legacy clean oracle arrays
DEFAULT_PREFIX_BRIDGE_ADAPTER: str = DEFAULT_BRIDGE_ADAPTER  # legacy oracle adapter (embedding_dim=1536, condition_dim=128) — coherent with legacy report/arrays above; was 991a_seed1 (CONTAMINATED)
# ---------------------------------------------------------------------------
# FULL TRAIN-REGION START BRIDGE (clean, 2026-06-20) — the demo START pool.
# ---------------------------------------------------------------------------
# Built by build_train_region_full_start_bridge.py from the clean 939a numeric
# support bank: 4010 train windows whose day-0 dates span Feb 2000 .. Jan 2016
# (incl. the 2008 GFC). window_indices are positional [0..4009] so the smoke takes the
# support-bank start branch (start pool == retrieval pool == 939a, consistent).
# condition_vectors are a zeros placeholder (held-out eval path only; the live
# demo gets query memory from the legacy-oracle adapter + 939a, unchanged).
# This is NOT the contaminated 990f/991a bridge — it carries no episode-card text.
DEFAULT_PREFIX_FULL_START_BRIDGE_REPORT: str = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_full_start_bridge_train_region_939a/full_start_bridge_report.json"
)
DEFAULT_PREFIX_FULL_START_BRIDGE_ARRAYS: str = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_full_start_bridge_train_region_939a/full_start_bridge_arrays.npz"
)
_HOW_IT_WORKS_NOTE = (
    "\n\n> **How it works:** your narrative describes the market conditions *now*. "
    "The demo matches them to the most similar real historical setups and rolls those "
    "forward, so the fan shows what has historically *followed* conditions like these — "
    "which can differ from simply projecting those moves forward."
)
DEFAULT_PREFIX_ROLLOUT_TEMPERATURE = 0.5
DEFAULT_PREFIX_ROLLOUT_FAN_SCALE = 3.5
DEFAULT_PREFIX_ENSEMBLE_CALIBRATION_BETA = 0.25
DEFAULT_PREFIX_ENSEMBLE_CALIBRATION_ALPHA = 1.0
DEFAULT_PREFIX_ENSEMBLE_CALIBRATION_BETA_BOUND = 0.25
TOP3_90_ANALOGUE_KEY = "TOP3_90"
TOP3_90_ENSEMBLE_LABEL = "Built from the most similar historical regimes"
TOP3_90_MAX_COMPONENTS = 3
TOP3_90_MIN_WEIGHT_MASS = 0.90
JOINT39_SPEC_NAMES = [
    "iv:00",
    "iv:01",
    "iv:02",
    "iv:03",
    "iv:04",
    "iv:05",
    "iv:06",
    "iv:07",
    "iv:08",
    "iv:09",
    "iv:10",
    "iv:11",
    "iv:12",
    "iv:13",
    "iv:14",
    "iv:15",
    "iv:16",
    "iv:17",
    "iv:18",
    "iv:19",
    "iv:20",
    "iv:21",
    "iv:22",
    "iv:23",
    "iv:24",
    "factor:spx",
    "factor:usdcad",
    "factor:usdjpy",
    "factor:dxy",
    "factor:copper",
    "factor:wheat",
    "factor:crude_oil",
    "factor:us2y",
    "factor:us10y",
    "factor:aaa_oas",
    "factor:bbb_oas",
    "factor:nikkei",
    "factor:gold",
    "factor:vix",
]
DEFAULT_USER_START_STATE_JSON = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_story_gradio_demo/prefix_latent_live_smoke/"
    "user_start_state_18.json"
)
DEFAULT_PREFIX_START_RELIABILITY_MANIFEST = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_start_reliability_gate_865d_full_s192_symmetric/"
    "start_reliability_gate.json"
)
DEFAULT_BOSS_DEMO_PACK_JSON = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_boss_demo_pack_829a_live_casebook/"
    "boss_demo_pack.json"
)
NARRATIVE_EXAMPLE_FAMILIES = [
    (
        "dollar_squeeze_liquidation",
        "Dollar squeeze/liquidation",
        {
            "short": (
                "Dollar funding is tight: DXY is surging, USDJPY is breaking "
                "lower, equities and BBB credit are under pressure, and oil plus "
                "gold are being sold."
            ),
            "medium": (
                "The current tape is a dollar-liquidity squeeze. Equities are "
                "falling, volatility is elevated, BBB spreads are wider, DXY is "
                "bid, USDJPY is lower, and both crude and gold are being "
                "liquidated rather than acting like clean hedges."
            ),
            "full": (
                "Professional read: this is a dollar-liquidity squeeze with "
                "commodity and gold liquidation. The active condition is broad "
                "de-risking through lower equities, higher volatility, wider "
                "lower-quality credit, a stronger DXY, weaker USDJPY, and heavy "
                "selling in crude and gold. The curve signal is secondary; the "
                "main portfolio sensitivity is dollar funding, high-beta credit, "
                "commodity beta, and positions that assume gold is providing a "
                "normal haven offset."
            ),
        },
    ),
    (
        "safe_haven_risk_off",
        "Safe-haven risk-off",
        {
            "short": (
                "Risk assets are soft, volatility is up, Treasury yields are "
                "lower, gold is bid, and credit is showing stress."
            ),
            "medium": (
                "The tape is defensive but not a pure dollar squeeze. Equities "
                "are selling off, VIX is firmer, BBB spreads are wider, Treasury "
                "yields are lower, and gold is catching a haven bid while the "
                "dollar channel is less dominant."
            ),
            "full": (
                "Professional read: the condition is a safety bid around an "
                "equity-credit drawdown. Transmission runs from weaker equities "
                "and firmer volatility into lower-quality credit stress, lower "
                "Treasury yields, and gold demand. The important distinction is "
                "that gold and duration are confirming defense, while FX is not "
                "the only channel carrying the stress."
            ),
        },
    ),
    (
        "weak_dollar_commodity_bid",
        "Weak-dollar commodity bid",
        {
            "short": (
                "DXY is heavy, USDJPY is firm, crude and gold are bid, rates are "
                "a touch higher, and credit is mostly calm."
            ),
            "medium": (
                "The active condition is a weak-dollar commodity repricing. DXY "
                "is lower while USDJPY, crude, and gold are higher; Treasury "
                "yields are only modestly firmer and BBB credit is quiet."
            ),
            "full": (
                "Professional read: this is a weak-dollar hard-asset bid with "
                "restrained credit stress. The trigger is a softer DXY alongside "
                "a firmer USDJPY cross, stronger crude, and stronger gold. "
                "Transmission reaches rates through a small backup, but BBB "
                "credit remains anchored, so the scenario is commodity/FX-led "
                "rather than broad credit deterioration."
            ),
        },
    ),
    (
        "rates_tightening_pressure",
        "Rates tightening pressure",
        {
            "short": (
                "Treasury yields are backing up, the dollar is firm, equities are "
                "struggling, and volatility is grinding higher."
            ),
            "medium": (
                "The current condition is rates-led tightening pressure. The "
                "front end and long end are firmer, DXY is supported, equities "
                "are under duration pressure, and volatility is rising without a "
                "full credit accident."
            ),
            "full": (
                "Professional read: the market is repricing around higher rates "
                "and a firmer dollar. Equities are struggling with duration "
                "pressure, volatility is grinding higher, and credit is fragile "
                "but not the first mover. The key exposure is growth-sensitive "
                "equity beta and duration-sensitive carry, not a classic "
                "safe-haven liquidation."
            ),
        },
    ),
    (
        "post_stress_reflation_relief",
        "Post-stress reflation relief",
        {
            "short": (
                "Equities are rebounding, volatility is compressing, crude is "
                "firmer, and credit is healing unevenly."
            ),
            "medium": (
                "The tape is post-stress reflation relief. SPX is higher, VIX is "
                "lower, crude is participating, BBB credit is improving, and "
                "high-grade spreads remain the main unresolved split."
            ),
            "full": (
                "Professional read: risk appetite is recovering after stress, "
                "but the confirmation is uneven. Equity beta and volatility "
                "compression lead the move, crude provides reflation support, "
                "and BBB spreads improve. The main ambiguity is high-grade "
                "credit basis pressure, so the condition is relief with a credit "
                "quality split rather than a clean broad-risk rally."
            ),
        },
    ),
    (
        "split_credit_quality_stress",
        "Split credit-quality stress",
        {
            "short": (
                "BBB credit is widening while high-grade behaves differently; "
                "risk tone is mixed and the spread signal is the issue."
            ),
            "medium": (
                "The active condition is split credit-quality stress. Lower-"
                "quality spreads are under pressure, high-grade spreads do not "
                "confirm in the same direction, and cross-asset risk signals are "
                "mixed rather than one-way."
            ),
            "full": (
                "Professional read: this is not a simple risk-on or risk-off "
                "state. The key condition is divergence inside credit quality: "
                "BBB spreads point to stress while high-grade credit is moving "
                "differently. Equities, FX, rates, and commodities provide "
                "partial context, but portfolio sensitivity should be framed "
                "around credit-quality basis and hedges that assume spread "
                "cohorts move together."
            ),
        },
    ),
]
RECOMMENDED_NARRATIVE_EXAMPLES = [
    (
        f"{family_key}__{length}",
        f"{family_label} - {length}",
        narrative,
    )
    for family_key, family_label, variants in NARRATIVE_EXAMPLE_FAMILIES
    for length, narrative in variants.items()
]
APP_DEFAULT_STORY = RECOMMENDED_NARRATIVE_EXAMPLES[0][2]
DEFAULT_AUTH_USER_ENV = "NARRATIVE_DEMO_AUTH_USER"
DEFAULT_AUTH_PASSWORD_ENV = "NARRATIVE_DEMO_AUTH_PASSWORD"
DEMO_TABLE_CLASS = "demo-scroll-table"
APP_CSS = """
.gradio-container {
  width: 100% !important;
  max-width: 1180px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  overflow-x: hidden;
}
.gradio-container,
.gradio-container * {
  box-sizing: border-box;
  min-width: 0;
}
.gradio-container code,
.gradio-container pre,
.gradio-container .prose,
.gradio-container .markdown {
  overflow-wrap: anywhere;
  word-break: break-word;
}
.gradio-container p,
.gradio-container li {
  max-width: 100%;
  white-space: normal;
}
.gradio-container p code,
.gradio-container li code {
  display: inline;
  white-space: normal !important;
}
.demo-scroll-table {
  width: 100% !important;
  min-width: 0 !important;
  max-width: 100%;
  overflow-x: auto !important;
}
.demo-scroll-table > div {
  width: 100% !important;
  min-width: 0 !important;
  max-width: 100% !important;
}
.demo-scroll-table table {
  width: max-content;
  max-width: none;
}
.demo-shell {
  width: 100%;
  max-width: 1120px;
  margin-left: auto;
  margin-right: auto;
}
.demo-hero {
  margin-bottom: 0.35rem;
  text-align: center;
}
.demo-status-strip {
  padding: 0.55rem 0.75rem;
  border-left: 4px solid #2563eb;
  background: #eff6ff;
  border-radius: 6px;
  font-size: 0.95rem;
}
.demo-status-strip p {
  margin: 0;
}
.scenario-summary-wrap {
  width: 100%;
  max-width: 100%;
  overflow-x: auto;
}
.scenario-summary-table {
  width: 100%;
  min-width: 680px;
  border-collapse: collapse;
  font-size: 0.92rem;
}
.scenario-summary-table th,
.scenario-summary-table td {
  padding: 0.55rem 0.65rem;
  border-bottom: 1px solid #e5e7eb;
  text-align: left;
  vertical-align: middle;
}
.scenario-summary-table th {
  color: #374151;
  background: #f9fafb;
  font-weight: 650;
}
.demo-dir {
  font-weight: 650;
  white-space: nowrap;
}
.demo-dir-arrow {
  display: inline-block;
  min-width: 0.9em;
  font-weight: 850;
}
.demo-dir-up .demo-dir-arrow {
  color: #16a34a;
}
.demo-dir-down .demo-dir-arrow {
  color: #dc2626;
}
.demo-dir-flat .demo-dir-arrow {
  color: #6b7280;
}
.demo-dir-moderate .demo-dir-arrow {
  color: #f59e0b;
}
.demo-responsive-row {
  gap: 0.75rem;
  align-items: stretch;
}
@media (max-width: 900px) {
  .demo-responsive-row {
    flex-direction: column !important;
  }
  .demo-responsive-row > div {
    width: 100% !important;
    min-width: 0 !important;
  }
}
@media (max-width: 640px) {
  html,
  body {
    max-width: 100vw;
    overflow-x: hidden;
  }
  .gradio-container {
    max-width: 100vw !important;
    padding-left: 14px !important;
    padding-right: 14px !important;
  }
  .gradio-container h1 {
    font-size: 1.55rem !important;
    line-height: 1.18 !important;
  }
  .gradio-container h2 {
    font-size: 1.25rem !important;
    line-height: 1.2 !important;
  }
  .gradio-container textarea,
  .gradio-container input,
  .gradio-container label {
    font-size: 0.92rem !important;
  }
  .gradio-container .wrap,
  .gradio-container .contain {
    min-width: 0 !important;
  }
  .gradio-container p,
  .gradio-container li,
  .gradio-container .prose p,
  .gradio-container .markdown p,
  .gradio-container .prose li,
  .gradio-container .markdown li {
    width: 100% !important;
    max-width: calc(100vw - 64px) !important;
  }
}
.ess-strip,
.calendar-start-strip {
  padding: 8px 12px;
  margin: 6px 0;
  border-radius: 6px;
  background: rgba(33, 150, 243, 0.06);
  font-size: 0.95em;
}
.calendar-start-strip {
  background: rgba(69, 90, 100, 0.06);
}
.ess-strip-na {
  border-left: 4px solid #9E9E9E;
}
.ess-note {
  color: #607D8B;
  font-size: 0.85em;
}
"""
CACHED_PREFIX_CASEBOOK_CONFIG = [
    (
        "commodity_inflation_pressure",
        "Commodity inflation pressure / start 18",
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_822a_commodity/condition_only_report.json",
        18,
    ),
    (
        "commodity_inflation_pressure",
        "Commodity inflation pressure / balanced start 40",
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_822a_commodity/condition_only_report.json",
        40,
    ),
    (
        "commodity_inflation_pressure",
        "Commodity inflation pressure / rates start 178",
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_822a_commodity/condition_only_report.json",
        178,
    ),
    (
        "dollar_liquidity_squeeze",
        "Dollar liquidity squeeze / defensive start 22",
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_823a_dollar/condition_only_report.json",
        22,
    ),
    (
        "dollar_liquidity_squeeze",
        "Dollar liquidity squeeze / balanced start 77",
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_823a_dollar/condition_only_report.json",
        77,
    ),
    (
        "dollar_liquidity_squeeze",
        "Dollar liquidity squeeze / memory-nearest start 0",
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_823a_dollar/condition_only_report.json",
        0,
    ),
    (
        "safe_haven_gold_bid",
        "Safe-haven gold bid / start 18",
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_823b_safe_haven/condition_only_report.json",
        18,
    ),
    (
        "safe_haven_gold_bid",
        "Safe-haven gold bid / balanced start 77",
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_823b_safe_haven/condition_only_report.json",
        77,
    ),
    (
        "safe_haven_gold_bid",
        "Safe-haven gold bid / rates start 178",
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_823b_safe_haven/condition_only_report.json",
        178,
    ),
]


# ---------------------------------------------------------------------------
# Corpus path safety (contamination gate)
# ---------------------------------------------------------------------------
_CONTAMINATED_PATH_FRAGMENTS: tuple[str, ...] = (
    "episode_card_v3_full_codex_multiformat_982g_sharded",
    "stride5_fourteen_view_bank_988b",
    "stride5_14x14_support_audit_990f",
    "stride5_14x14_retrieval_training_openai_holdout_990a",
    "stride5_14x14_retrieval_training_openai_holdout_991a_seed",
    "990f",
    "991a_seed",
)
_CLEAN_CORPUS_PATHS: tuple[str, ...] = (
    "prefix_latent_support_bank_train_all_939a",
    "episode_card_v3_codex_multiformat_982g_clean_stride5_20260612",
    "manifest_bridge_eval_openai_schema_v2_representative_220",
    "manifest_openai_schema_v2_representative_220",
)


def assert_clean_corpus_paths(*paths: "str | Path | None") -> None:
    """Raise RuntimeError if any non-None path matches a known-contaminated fragment.

    The joint39 factor-mapping contamination (2026-06-11) affected corpora in
    the 982g_sharded, 988b, 990f, 990a, and 991a_seed* output directories.
    Support paths (939a) were regenerated clean and are safe to serve.
    """
    for raw_path in paths:
        if raw_path is None:
            continue
        path_str = str(raw_path)
        for fragment in _CONTAMINATED_PATH_FRAGMENTS:
            if fragment in path_str:
                raise RuntimeError(
                    f"[contamination-gate] BLOCKED: path contains known-contaminated "
                    f"fragment '{fragment}': {path_str!r}\n"
                    "The joint39 factor-mapping bug caused USDJPY and AAA_OAS "
                    "narratives to be generated from the wrong data columns. "
                    "Repoint to a clean corpus before starting this app."
                )


# Startup check — fires at import time so misconfiguration fails loudly before
# the Gradio server starts.  Active paths must NOT be from the contaminated
# 990f/991a_seed chain.  The legacy oracle bridge (manifest_bridge_eval_*) and
# the 939a support bank are clean.
assert_clean_corpus_paths(
    DEFAULT_PREFIX_BRIDGE_REPORT,   # legacy oracle bridge — clean
    DEFAULT_PREFIX_BRIDGE_ARRAYS,   # legacy oracle arrays — clean
    DEFAULT_PREFIX_BRIDGE_ADAPTER,  # legacy oracle adapter — clean
    DEFAULT_PREFIX_SUPPORT_BANK_REPORT,   # 939a — clean
    DEFAULT_PREFIX_SUPPORT_BANK_ARRAYS,   # 939a — clean
    DEFAULT_PREFIX_FULL_START_BRIDGE_REPORT,  # 939a-derived full start bridge — clean
    DEFAULT_PREFIX_FULL_START_BRIDGE_ARRAYS,  # 939a-derived full start arrays — clean
)


def resolve_launch_auth(
    *,
    env: dict[str, str] | None = None,
    user_env: str = DEFAULT_AUTH_USER_ENV,
    password_env: str = DEFAULT_AUTH_PASSWORD_ENV,
    require_auth: bool = False,
) -> tuple[str, str] | None:
    env_map = os.environ if env is None else env
    user = str(env_map.get(user_env, "")).strip()
    password = str(env_map.get(password_env, "")).strip()
    if user and password:
        return user, password
    if require_auth:
        missing = []
        if not user:
            missing.append(user_env)
        if not password:
            missing.append(password_env)
        raise RuntimeError(
            "missing required Gradio auth environment variable(s): "
            + ", ".join(missing)
        )
    return None


IMPLICATION_COLUMNS = [
    "Market",
    "Direction",
    "Magnitude",
    "Confidence",
    "Inferred",
    "Evidence",
]
WARNING_COLUMNS = ["Severity", "Code", "Message"]
# Primary (non-audit) warnings display drops the machine "Code"; the code stays
# in the audit JSON / markdown report for traceability.
PREFIX_WARNING_DISPLAY_COLUMNS = ["Severity", "Message"]
PREFIX_CONDITION_COLUMNS = [
    "Market",
    "Direction",
    "Magnitude",
    "Confidence",
    "Horizon",
    "Evidence",
]
PREFIX_WARNING_COMPONENT_COLUMNS = ["Check", "Status", "Metric", "Value"]
PREFIX_SHIFT_FACTOR_COLUMNS = [
    "Start Mode",
    "Factor",
    "Terminal Abs Z",
    "Signed Terminal Z",
    "Path Abs Z",
]
ANALOGUE_COLUMNS = [
    "Rank",
    "Window",
    "Source",
    "History End",
    "Cosine",
    "Weight",
    "Implication Match",
    "Fit Status",
    "Split",
    "Narrative",
]
SCENARIO_COLUMNS = [
    "Market",
    "Baseline View",
    "Baseline Path Share",
    "Baseline Mean Move",
    "Narrative View",
    "Narrative Path Share",
    "Narrative Mean Move",
    "30d Change vs Baseline",
]
VALIDATION_GATE_COLUMNS = [
    "Variant",
    "Role",
    "Query",
    "Start",
    "Status",
    "Memory Cosine",
    "Distance from start (σ)",
    "Terminal Shift",
    "Warnings",
    "Failures",
]
PREFIX_VARIANT_COLUMNS = [
    "Variant",
    "Query Window",
    "Start Window",
    "Distance from start (σ)",
    "Memory Support",
    "Selection",
    "Start Split",
]
# Primary selected-start view shows only the human-meaningful day-0 date and the
# narrative match. The raw window Index/Source live in the audit JSON, and the
# "distance from start" is structurally 0 in explicit-start mode (the requested
# start IS the selected start) so it is omitted rather than shown as a misleading
# distance-to-the-query-stub.
PREFIX_SELECTED_START_COLUMNS = [
    "Starting Level",
    "Narrative match (cosine)",
]
PREFIX_START_CANDIDATE_COLUMNS = [
    "Used For",
    "Rank",
    "Episode date",
    "Weight",
    "Narrative match (cosine)",
    "Distance from start (σ)",
    "Narrative directions",
]
PREFIX_USER_START_COLUMNS = [
    "Label",
    "Source",
    "Format",
    "Dimension",
    "Nearest Train",
    "Distance from start (σ)",
    "Max Abs Z",
]
PREFIX_START_PREVIEW_COLUMNS = ["Field", "Value"]
BOSS_DEMO_CASEBOOK_COLUMNS = [
    "Case",
    "Start",
    "Status",
    "Condition",
    "Warnings",
    "Support",
    "Summary",
]
START_PREVIEW_FIELDS = [
    ("SPX", "factor:spx"),
    ("VIX", "factor:vix"),
    ("BBB OAS", "factor:bbb_oas"),
    ("AAA OAS", "factor:aaa_oas"),
    ("US 2Y", "factor:us2y"),
    ("US 10Y", "factor:us10y"),
    ("USD/JPY", "factor:usdjpy"),
    ("DXY", "factor:dxy"),
    ("Gold", "factor:gold"),
    ("Crude oil", "factor:crude_oil"),
    ("IV ATM 3M", "iv:07"),
    ("IV ATM 1Y", "iv:17"),
]
SCENARIO_MARKET_TO_START_SPEC = {
    "SPX": "factor:spx",
    "VIX": "factor:vix",
    "BBB_OAS": "factor:bbb_oas",
    "AAA_OAS": "factor:aaa_oas",
    "US2Y": "factor:us2y",
    "US10Y": "factor:us10y",
    "USDJPY": "factor:usdjpy",
    "DXY": "factor:dxy",
    "GOLD": "factor:gold",
    "CRUDE_OIL": "factor:crude_oil",
    "IV_ATM_3M": "iv:07",
    "IV_ATM_1Y": "iv:17",
}
FAN_MARKET_CHOICES = [
    ("SPX", "SPX"),
    ("VIX", "VIX"),
    ("BBB OAS", "BBB_OAS"),
    ("AAA OAS", "AAA_OAS"),
    ("US 2Y", "US2Y"),
    ("US 10Y", "US10Y"),
    ("USD/JPY", "USDJPY"),
    ("DXY", "DXY"),
    ("Gold", "GOLD"),
    ("Crude oil", "CRUDE_OIL"),
    ("IV surface average", "IV_SURFACE"),
    ("IV ATM 3M, K=1.00", "IV_ATM_3M"),
    ("IV ATM 1Y, K=1.00", "IV_ATM_1Y"),
    ("IV OTM put 1Y, K=0.70", "IV_OTM_PUT_1Y"),
    ("IV wing 6M, K=1.30", "IV_WING_6M_K130"),
]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{100.0 * float(value):+.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _short(text: Any, limit: int = 160) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= int(limit):
        return compact
    return compact[: int(limit) - 3].rstrip() + "..."


def _slug(text: str) -> str:
    keep: list[str] = []
    for char in str(text).lower():
        if char.isalnum():
            keep.append(char)
        elif keep and keep[-1] != "_":
            keep.append("_")
    return "".join(keep).strip("_") or "case"


def _frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def cached_prefix_casebook_rows() -> list[dict[str, Any]]:
    """Return cached condition-only demo rows for the prefix-latent UI."""

    story_by_name = {
        str(item["name"]): str(item["story"]) for item in default_casebook_stories()
    }
    rows: list[dict[str, Any]] = []
    for (
        case_name,
        label,
        condition_report,
        start_index,
    ) in CACHED_PREFIX_CASEBOOK_CONFIG:
        rows.append(
            {
                "label": str(label),
                "value": f"{case_name}:{int(start_index)}",
                "case_name": str(case_name),
                "story": story_by_name.get(str(case_name), DEFAULT_STORY),
                "condition_report": str(condition_report),
                "start_index": int(start_index),
            }
        )
    return rows


def cached_prefix_casebook_choices() -> list[tuple[str, str]]:
    return [("Typed story / current controls", "")] + [
        (str(row["label"]), str(row["value"])) for row in cached_prefix_casebook_rows()
    ]


def recommended_narrative_choices() -> list[tuple[str, str]]:
    return [("Type my own narrative", "")] + [
        (label, key) for key, label, _story in RECOMMENDED_NARRATIVE_EXAMPLES
    ]


def recommended_narrative_text(choice: str | None, current_story: str) -> str:
    key = str(choice or "")
    if not key:
        return str(current_story or APP_DEFAULT_STORY)
    for example_key, _label, narrative in RECOMMENDED_NARRATIVE_EXAMPLES:
        if str(example_key) == key:
            return str(narrative)
    return str(current_story or APP_DEFAULT_STORY)


def cached_prefix_casebook_update(choice: str | None) -> tuple[
    str,
    bool,
    int,
    bool,
    bool,
    str,
    str,
]:
    """Populate story/start/report controls from a saved demo setup."""

    value = str(choice or "")
    if not value:
        return (
            DEFAULT_STORY,
            False,
            22,
            True,
            False,
            "",
            "## Saved Demo Setup\n\n- Selection: `typed story / current controls`",
        )
    for row in cached_prefix_casebook_rows():
        if str(row["value"]) != value:
            continue
        report_path = str(row["condition_report"])
        exists_text = "available" if Path(report_path).exists() else "missing"
        return (
            str(row["story"]),
            True,
            int(row["start_index"]),
            False,
            False,
            report_path,
            "\n".join(
                [
                    "## Saved Demo Setup",
                    "",
                    f"- Selection: `{row['label']}`",
                    f"- Narrative family: `{row['case_name']}`",
                    f"- Fixed start index: `{row['start_index']}`",
                    f"- Cached condition report: `{exists_text}`",
                    "- OpenAI calls: `none for this cached run`",
                ]
            ),
        )
    return (
        DEFAULT_STORY,
        False,
        22,
        True,
        False,
        "",
        f"## Saved Demo Setup\n\n- Selection: `unknown ({value})`",
    )


def _operational_variant_row(report: dict[str, Any]) -> dict[str, Any]:
    for row in _as_list(report.get("variant_rows")):
        if isinstance(row, dict) and bool(row.get("is_operational")):
            return row
    for row in _as_list(report.get("variant_rows")):
        if isinstance(row, dict) and str(row.get("variant", "")) != "original":
            return row
    return {}


def _operational_score_metrics(report: dict[str, Any]) -> dict[str, Any]:
    generation = _as_dict(report.get("generation"))
    operational = _operational_variant_row(report)
    variant = str(operational.get("variant", ""))
    start_index = operational.get("start_window_index")
    for row in _as_list(generation.get("window_scores")):
        if not isinstance(row, dict):
            continue
        if variant and str(row.get("variant", "")) != variant:
            continue
        if start_index is not None and row.get("start_window_index") is not None:
            try:
                if int(row.get("start_window_index")) != int(start_index):
                    continue
            except (TypeError, ValueError):
                continue
        methods = _as_dict(row.get("methods"))
        model = _as_dict(methods.get("text_memory_plus_start_prefix_decoder"))
        if model:
            return model
    return {}


def prefix_trust_interpretation(report: dict[str, Any]) -> str:
    gate = _as_dict(report.get("validation_gate"))
    status = str(
        gate.get("selected_start_status", gate.get("operational_status", "unknown"))
    )
    metrics = _operational_score_metrics(report)
    crps_improvement = metrics.get("ensemble_crps_z_improvement_vs_persistence")
    try:
        crps_value = float(crps_improvement)
    except (TypeError, ValueError):
        crps_value = None
    if status == "pass":
        return "supported narrative scenario"
    if status == "warning" and crps_value is not None and crps_value > 0.0:
        return (
            "usable with support/shift caveats; scenario CRPS improved vs persistence"
        )
    if status == "warning":
        return "usable only with caveats; inspect support and rollout-shift warnings"
    if status == "fail":
        return "do not use without changing the narrative or starting state"
    return "status unavailable"


def _float_series(value: Any) -> list[float]:
    values = []
    for item in _as_list(value):
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return values


def implications_table(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grounding = _as_dict(report.get("grounding"))
    for item in _as_list(grounding.get("market_implications")):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Market": str(item.get("market", "")),
                "Direction": str(item.get("direction", "")),
                "Magnitude": str(item.get("magnitude", "")),
                "Confidence": str(item.get("confidence", "")),
                "Inferred": bool(item.get("inferred", False)),
                "Evidence": "; ".join(str(x) for x in _as_list(item.get("evidence"))),
            }
        )
    return _frame(rows, IMPLICATION_COLUMNS)


def warnings_table(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grounding = _as_dict(report.get("grounding"))
    for item in _as_list(grounding.get("grounding_warnings")):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Severity": str(item.get("severity", "")),
                "Code": str(item.get("code", "")),
                "Message": str(item.get("message", "")),
            }
        )
    return _frame(
        rows or [{"Severity": "none", "Code": "none", "Message": "none"}],
        WARNING_COLUMNS,
    )


def _prefix_grounding(report: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(_as_dict(report.get("cached_query")).get("grounding"))


def prefix_condition_implications_table(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grounding = _prefix_grounding(report)
    for item in _as_list(grounding.get("market_implications")):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Market": str(item.get("market", "")),
                "Direction": str(item.get("direction", "")),
                "Magnitude": str(item.get("magnitude", "")),
                "Confidence": str(item.get("confidence", "")),
                "Horizon": str(item.get("horizon", "")),
                "Evidence": "; ".join(str(x) for x in _as_list(item.get("evidence"))),
            }
        )
    return _frame(rows, PREFIX_CONDITION_COLUMNS)


def prefix_condition_warnings_table(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grounding = _prefix_grounding(report)
    for item in _as_list(grounding.get("non_conditioning_forward_language")):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Severity": str(item.get("severity", "warning")),
                "Code": "non_conditioning_forward_language",
                "Message": (
                    f"{item.get('phrase', '')} - "
                    f"{item.get('reason', item.get('handling', 'warning only'))}"
                ).strip(" -"),
            }
        )
    for item in _as_list(grounding.get("grounding_warnings")):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Severity": str(item.get("severity", "warning")),
                "Code": str(item.get("code", "grounding_warning")),
                "Message": str(item.get("message", "")),
            }
        )
    return _frame(
        rows or [{"Severity": "none", "Code": "none", "Message": "none"}],
        WARNING_COLUMNS,
    )


def prefix_condition_warnings_display_table(report: dict[str, Any]) -> pd.DataFrame:
    """Primary (non-audit) warnings view: severity + plain message only.

    The machine ``Code`` is intentionally dropped from the displayed table (it
    stays in the full ``prefix_condition_warnings_table`` / audit JSON) so the
    risk-manager-facing warnings panel reads as plain guidance.
    """

    full = prefix_condition_warnings_table(report)
    return _frame(
        full[PREFIX_WARNING_DISPLAY_COLUMNS].to_dict("records"),
        PREFIX_WARNING_DISPLAY_COLUMNS,
    )


def prefix_visible_warning_lines(report: dict[str, Any]) -> list[str]:
    """Return product-facing warning lines that should not be hidden in audit details."""

    grounding = _prefix_grounding(report)
    lines: list[str] = []
    for item in _as_list(grounding.get("non_conditioning_forward_language")):
        if not isinstance(item, dict):
            continue
        phrase = _short(item.get("phrase", ""), limit=150)
        reason = _short(
            item.get(
                "reason",
                "future-looking language is warning-only, not a conditioning fact",
            ),
            limit=150,
        )
        if phrase:
            lines.append(
                "Forward-looking language excluded from conditioning: "
                f'"{phrase}" ({reason}).'
            )
    for item in _as_list(grounding.get("grounding_warnings")):
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", "grounding_warning"))
        message = _short(item.get("message", ""), limit=170)
        if not message:
            continue
        if code == "FORWARD_LOOKING_EXCLUDED" and any(
            "Forward-looking language excluded" in line for line in lines
        ):
            continue
        lines.append(f"{code}: {message}")
    return lines


def analogues_table(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(_as_list(report.get("historical_analogues")), start=1):
        if not isinstance(item, dict):
            continue
        alignment = _as_dict(item.get("implication_alignment"))
        calendar = _as_dict(item.get("calendar"))
        rows.append(
            {
                "Rank": rank,
                "Window": str(item.get("window_id", "")),
                "Source": str(item.get("source_index", "")),
                "History End": str(calendar.get("history_end", "")),
                "Cosine": _fmt_float(item.get("cosine")),
                "Weight": _fmt_float(item.get("weight")),
                "Implication Match": _fmt_float(alignment.get("match_rate")),
                "Fit Status": str(alignment.get("status", "")),
                "Split": str(item.get("manifest_split", "")),
                "Narrative": _short(
                    item.get("casebook_narrative") or item.get("primary_narrative")
                ),
            }
        )
    return _frame(rows, ANALOGUE_COLUMNS)


def _selected_start_values_by_spec(report: dict[str, Any]) -> dict[str, Any]:
    selected = _as_dict(report.get("selected_start_state"))
    values = _as_dict(selected.get("values_by_name"))
    if values:
        return values
    return _as_dict(report.get("selected_start_values_by_name"))


def _market_start_level(report: dict[str, Any], market: str) -> float | None:
    start_values = _selected_start_values_by_spec(report)
    market_key = str(market).upper()
    if market_key == "IV_SURFACE":
        iv_values = [
            float(value)
            for key, value in start_values.items()
            if str(key).startswith("iv:") and _is_finite(value)
        ]
        return float(np.nanmean(iv_values)) if iv_values else None
    start_spec = SCENARIO_MARKET_TO_START_SPEC.get(market_key, "")
    if start_spec and _is_finite(start_values.get(start_spec)):
        return float(start_values[start_spec])
    return None


def _add_start_level(value: Any, start_level: float | None) -> float | None:
    if not (_is_finite(value) and _is_finite(start_level)):
        return None
    return float(value) + float(start_level)


def _terminal_mean_delta(row: dict[str, Any]) -> float | None:
    value = row.get("mean_terminal_delta")
    return float(value) if _is_finite(value) else None


def _terminal_band_width(row: dict[str, Any]) -> float | None:
    p10 = row.get("p10")
    p90 = row.get("p90")
    if not (_is_finite(p10) and _is_finite(p90)):
        return None
    return abs(float(p90) - float(p10))


def _terminal_direction_view(row: dict[str, Any]) -> str:
    mean = _terminal_mean_delta(row)
    if mean is None:
        return "n/a"
    width = _terminal_band_width(row)
    threshold = max(0.05 * float(width or abs(mean) or 1.0), 1.0e-8)
    if abs(float(mean)) <= threshold:
        return "Flat/mixed"
    return "Up" if float(mean) > 0.0 else "Down"


def _direction_display_label(value: str) -> str:
    text = str(value or "").strip()
    if text == "Flat/mixed":
        return "-"
    return text or "n/a"


def _baseline_change_display_label(value: str) -> str:
    return str(value or "").strip() or "n/a"


def _direction_html(value: str) -> str:
    text = str(value or "").strip()
    if text == "Up":
        return (
            '<span class="demo-dir demo-dir-up">'
            '<span class="demo-dir-arrow">↑</span> Up</span>'
        )
    if text == "Down":
        return (
            '<span class="demo-dir demo-dir-down">'
            '<span class="demo-dir-arrow">↓</span> Down</span>'
        )
    if text in {"Flat/mixed", "-"}:
        return (
            '<span class="demo-dir demo-dir-flat">'
            '<span class="demo-dir-arrow">-</span></span>'
        )
    return html.escape(text or "n/a")


def _baseline_change_html(value: str) -> str:
    # This column is a RELATIVE tilt vs the start-only baseline, not an absolute
    # direction — so it carries no up/down arrow glyph (the words carry it); only
    # the colour class conveys the tilt. (The View column keeps its ↑/↓ arrows.)
    text = str(value or "").strip()
    if text in {"Higher than baseline", "More up than baseline"}:
        return f'<span class="demo-dir demo-dir-up">{html.escape(text)}</span>'
    if text in {"Lower than baseline", "More down than baseline"}:
        return f'<span class="demo-dir demo-dir-down">{html.escape(text)}</span>'
    if text in {"Less down than baseline", "Less up than baseline"}:
        return f'<span class="demo-dir demo-dir-moderate">{html.escape(text)}</span>'
    if text == "Similar to baseline":
        return '<span class="demo-dir demo-dir-flat">Similar to baseline</span>'
    return html.escape(text or "n/a")


def _terminal_probability_pair(row: dict[str, Any]) -> tuple[float, float] | None:
    up = row.get("terminal_probability_up")
    down = row.get("terminal_probability_down")
    if not (_is_finite(up) and _is_finite(down)):
        return None
    up_f = max(0.0, min(1.0, float(up)))
    down_f = max(0.0, min(1.0, float(down)))
    return up_f, down_f


def _typical_direction_view(row: dict[str, Any]) -> str:
    probabilities = _terminal_probability_pair(row)
    if probabilities is None:
        return _terminal_direction_view(row)
    up, down = probabilities
    threshold = 0.60
    if up >= threshold and up > down:
        return "Up"
    if down >= threshold and down > up:
        return "Down"
    return "Flat/mixed"


def _path_share_label(row: dict[str, Any]) -> str:
    probabilities = _terminal_probability_pair(row)
    if probabilities is None:
        return "n/a"
    up, down = probabilities
    if abs(up - down) <= 0.005:
        return f"{max(up, down):.0%} split"
    if up >= down:
        return f"{up:.0%} up"
    return f"{down:.0%} down"


def _terminal_sigma_score(row: dict[str, Any]) -> float | None:
    mean = _terminal_mean_delta(row)
    if mean is None:
        return None
    explicit = row.get("terminal_mean_sigma")
    if _is_finite(explicit):
        return float(explicit)
    width = _terminal_band_width(row)
    if width is None or float(width) <= 1.0e-12:
        return None
    # For older reports, estimate one standard deviation from the 10%-90% band
    # under a normal approximation: q90 - q10 ~= 2 * 1.28155 * sigma.
    sigma = float(width) / 2.5631031310892007
    if sigma <= 1.0e-12:
        return None
    return float(mean) / sigma


def _signed_number(value: float, digits: int) -> str:
    rounded = round(float(value), int(digits))
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:+.{int(digits)}f}"


def _mean_move_raw_part(market: str, mean: float) -> str:
    market_key = str(market or "").upper()
    if market_key == "IV_SURFACE" or market_key.startswith("IV_"):
        return f"{_signed_number(float(mean) * 100.0, 1)} vol pts"
    if market_key in {"US2Y", "US10Y", "BBB_OAS", "AAA_OAS"}:
        return f"{_signed_number(float(mean) * 100.0, 1)} bp"
    if market_key in {"SPX", "GOLD"}:
        return f"{_signed_number(float(mean), 0)} pts"
    return f"{_signed_number(float(mean), 1)} pts"


def _mean_move_label(row: dict[str, Any], *, market: str) -> str:
    mean = _terminal_mean_delta(row)
    if mean is None:
        return "n/a"
    raw = _mean_move_raw_part(market, float(mean))
    sigma = _terminal_sigma_score(row)
    if sigma is None:
        return raw
    return f"{raw} / {_signed_number(float(sigma), 1)}σ"


def _terminal_change_vs_baseline(
    narrative_row: dict[str, Any],
    baseline_row: dict[str, Any] | None,
) -> str:
    narrative_mean = _terminal_mean_delta(narrative_row)
    baseline_mean = _terminal_mean_delta(baseline_row or {})
    if narrative_mean is None or baseline_mean is None:
        return "Baseline unavailable"
    narrative_width = _terminal_band_width(narrative_row) or 0.0
    baseline_width = _terminal_band_width(baseline_row or {}) or 0.0
    threshold = max(0.10 * max(float(narrative_width), float(baseline_width)), 1.0e-8)
    delta = float(narrative_mean) - float(baseline_mean)
    if abs(delta) <= threshold:
        return "Similar to baseline"
    narrative_view = _terminal_direction_view(narrative_row)
    if delta > 0.0:
        if narrative_view == "Down":
            return "Less down than baseline"
        if narrative_view == "Up":
            return "More up than baseline"
        return "Higher than baseline"
    if narrative_view == "Down":
        return "More down than baseline"
    if narrative_view == "Up":
        return "Less up than baseline"
    return "Lower than baseline"


def _terminal_rows_by_market(rows: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _as_list(rows):
        if not isinstance(row, dict):
            continue
        market = str(row.get("market", "")).strip()
        if market:
            result[market] = row
    return result


def scenario_table(report: dict[str, Any]) -> pd.DataFrame:
    generation = _as_dict(report.get("generation"))
    baseline = _as_dict(generation.get("start_only_baseline"))
    baseline_by_market = _terminal_rows_by_market(
        baseline.get("terminal_delta_summary")
    )
    rows: list[dict[str, Any]] = []
    for item in _as_list(generation.get("terminal_delta_summary")):
        if not isinstance(item, dict):
            continue
        market = str(item.get("market", ""))
        baseline_row = baseline_by_market.get(market)
        rows.append(
            {
                "Market": market,
                "Baseline View": _direction_display_label(
                    _typical_direction_view(baseline_row or {})
                ),
                "Baseline Path Share": _path_share_label(baseline_row or {}),
                "Baseline Mean Move": _mean_move_label(
                    baseline_row or {}, market=market
                ),
                "Narrative View": _direction_display_label(
                    _typical_direction_view(item)
                ),
                "Narrative Path Share": _path_share_label(item),
                "Narrative Mean Move": _mean_move_label(item, market=market),
                "30d Change vs Baseline": _baseline_change_display_label(
                    _terminal_change_vs_baseline(item, baseline_row)
                ),
            }
        )
    return _frame(rows, SCENARIO_COLUMNS)


def scenario_summary_html(report: dict[str, Any]) -> str:
    df = scenario_table(report)
    headers = "".join(f"<th>{html.escape(column)}</th>" for column in SCENARIO_COLUMNS)
    if df.empty:
        cells = "".join(
            "<td>&nbsp;</td>" if column != "Market" else "<td>No scenario yet</td>"
            for column in SCENARIO_COLUMNS
        )
        body = f"<tr>{cells}</tr>"
    else:
        row_html: list[str] = []
        for _, row in df.iterrows():
            cells = []
            for column in SCENARIO_COLUMNS:
                value = str(row.get(column, ""))
                if column in {"Baseline View", "Narrative View"}:
                    cells.append(f"<td>{_direction_html(value)}</td>")
                elif column == "30d Change vs Baseline":
                    cells.append(f"<td>{_baseline_change_html(value)}</td>")
                else:
                    cells.append(f"<td>{html.escape(value)}</td>")
            row_html.append(f"<tr>{''.join(cells)}</tr>")
        body = "".join(row_html)
    return (
        '<div class="scenario-summary-wrap">'
        '<table class="scenario-summary-table">'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table></div>"
    )


# --- Track D zero-risk "validated spine" product items ---------------------
# Five additive, gracefully-degrading product surfaces for the risk-manager
# demo (ESS, terminal Day-30 table, calendar start label, baseline-median fan
# overlay, standing disclaimer).  None of these touch generation logic, demo
# defaults, or the frozen 734a path; each guards on missing fields.

ESS_FLOOR: float = 3.0

TERMINAL_DAY30_COLUMNS = [
    "Factor",
    "Baseline P10",
    "Baseline P50",
    "Baseline P90",
    "Conditioned P10",
    "Conditioned P50",
    "Conditioned P90",
]


def _support_pool_weights(report: dict[str, Any]) -> list[float]:
    """Return the support-pool weights used for the displayed scenario fan.

    Mirrors the source ``prefix_start_candidates_table`` reads: the top3/90
    posterior-ensemble ``selected_support`` weights when present, otherwise the
    raw ``memory_prior`` candidate weights.  Returns ``[]`` if no usable
    weights are found so callers can degrade gracefully.
    """

    generation = _as_dict(report.get("generation"))
    posterior = _as_dict(generation.get("posterior_ensemble"))
    candidates = _as_list(posterior.get("selected_support"))
    if not candidates:
        memory_prior = _as_dict(_as_dict(report.get("cached_query")).get("memory_prior"))
        candidates = _as_list(memory_prior.get("candidate_details"))
        weights: list[float] = []
        for item in candidates:
            if isinstance(item, dict) and _is_finite(item.get("weight")):
                weights.append(float(item.get("weight")))
        if weights:
            return weights
        # Final fallback: a bare weights list on the memory prior.
        bare = _float_series(memory_prior.get("weights"))
        return [float(w) for w in bare if _is_finite(w)]
    weights = []
    for item in candidates:
        if isinstance(item, dict) and _is_finite(item.get("weight")):
            weights.append(float(item.get("weight")))
    return weights


def effective_sample_size(weights: list[float]) -> float | None:
    """Kish effective sample size ``1 / sum(w**2)`` over normalized weights.

    Defensively renormalizes (the formula is only meaningful when the weights
    sum to one) and returns ``None`` when the input cannot yield a finite ESS.
    """

    finite = [float(w) for w in weights if _is_finite(w) and float(w) >= 0.0]
    total = sum(finite)
    if not finite or not _is_finite(total) or total <= 0.0:
        return None
    normalized = [w / total for w in finite]
    denominator = sum(w * w for w in normalized)
    if not _is_finite(denominator) or denominator <= 0.0:
        return None
    return 1.0 / denominator


def support_ess_html(report: dict[str, Any]) -> str:
    """Render the support-pool effective sample size with a fixed floor.

    ESS is free: the weights already exist in the report.  Degrades to a
    neutral placeholder when no weights are available (e.g. the error yield
    path or a baseline-only report).
    """

    weights = _support_pool_weights(report)
    ess = effective_sample_size(weights)
    floor = ESS_FLOOR  # used only for the colour band below, not as a denominator
    if ess is None:
        return (
            '<div class="ess-strip ess-strip-na">'
            "<strong>Effective analogues:</strong> n/a "
            '<span class="ess-note">(no support weights in this run)</span>'
            "</div>"
        )
    # Colour vs floor: green when the pool is genuinely diverse, orange when it
    # is collapsing toward a single analogue, red when near-degenerate.
    if ess >= max(2.5, 0.83 * floor):
        colour = "#2E7D32"  # green
        tone = "ess-strip-green"
    elif ess >= 1.5:
        colour = "#EF6C00"  # orange
        tone = "ess-strip-orange"
    else:
        colour = "#C62828"  # red
        tone = "ess-strip-red"
    pool_n = len(weights)
    return (
        f'<div class="ess-strip {tone}" style="border-left:4px solid {colour};">'
        f'<strong style="color:{colour};">Effective analogues: {ess:.1f}</strong> '
        f'<span class="ess-note">'
        f"(from {pool_n} historical analogue{'s' if pool_n != 1 else ''}; "
        "higher = more diverse)"
        "</span></div>"
    )


def support_hull_html(report: dict[str, Any]) -> str:
    """Thin honesty badge: is the narrative's implied move within historical analogue support?

    Framework-v1 section I hull gate over the 14 named anchors. Leads with the GRADED signal
    (the severity kappa at which the implied completion leaves the historical hull, plus the
    pool Mahalanobis density); raises a loud flag ONLY when the scenario is genuinely outside
    historical precedent. The directional fan above is the product -- this is a quiet support
    badge, not a hedge on the arrow. Degrades silently when grounding/anchors are unavailable.
    """

    try:
        from experiments.backfill.block_ar.nl_hull_gate_inputs import (
            hull_label_from_grounding,
        )

        grounding = _as_dict(report.get("grounding"))
        if not grounding:
            grounding = _as_dict(_as_dict(report.get("cached_query")).get("grounding"))
        if not grounding:
            return (
                '<div class="ess-strip ess-strip-na">'
                "<strong>Historical support:</strong> "
                'n/a <span class="ess-note">(no grounding in this run)</span></div>'
            )
        out = hull_label_from_grounding(grounding, kappas=(0.5, 1.0, 2.0))
        ok_rungs = [r for r in out.get("ladder", []) if r.get("status") == "ok"]
        if not ok_rungs:
            return (
                '<div class="ess-strip ess-strip-na">'
                "<strong>Historical support:</strong> no anchor-specific implications to test"
                '<span class="ess-note"> (narrative did not pin named factors)</span></div>'
            )
        if out.get("any_indeterminate") and not out.get("any_infeasible"):
            # LP could not assess -> "could not check", NEVER conflate with outside-support.
            return (
                '<div class="ess-strip ess-strip-na">'
                "<strong>Historical support:</strong> check unavailable "
                '<span class="ess-note">(support feasibility LP indeterminate for this scenario)</span></div>'
            )
        if out.get("any_infeasible"):
            leaves = out.get("leaves_hull_at_kappa")
            colour = "#C62828"  # red -- genuinely outside historical precedent (rare)
            return (
                f'<div class="ess-strip" style="border-left:4px solid {colour};">'
                f'<strong style="color:{colour};">&#9888; No close historical precedent</strong> '
                '<span class="ess-note">The mix of moves your narrative implies is more extreme, '
                "taken together, than any real 30-day period on record. This scenario is a "
                "<strong>stress extrapolation, not a blend of past episodes</strong> &mdash; weight "
                "it as a what-if. <em>(Technical: implied move exits the historical support hull at "
                f"&kappa;&ge;{leaves}&sigma;.)</em></span></div>"
            )
        colour = "#2E7D32"  # green
        return (
            f'<div class="ess-strip" style="border-left:4px solid {colour};">'
            f'<strong style="color:{colour};">&#10003; Historical support (14 anchors): '
            "within precedent</strong> "
            '<span class="ess-note">the mix of moves your narrative implies stays within '
            "the range of real historical 30-day episodes</span></div>"
        )
    except Exception:
        return (
            '<div class="ess-strip ess-strip-na">'
            "<strong>Historical support:</strong> check unavailable</div>"
        )


def _terminal_quantiles_for_row(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Return the LAST (Day-30 terminal) P10/P50/P90 of a raw-level fan row."""

    if not row:
        return (None, None, None)
    p10 = _float_series(row.get("p10"))
    p50 = _float_series(row.get("p50"))
    p90 = _float_series(row.get("p90"))
    last10 = p10[-1] if p10 else None
    last50 = p50[-1] if p50 else None
    last90 = p90[-1] if p90 else None
    return (last10, last50, last90)


def _terminal_level_text(market: str, value: Any) -> str:
    """Format a raw Day-30 LEVEL with per-factor units/decimals.

    Distinct from the move-formatter (``_mean_move_raw_part``): these are
    absolute levels, not deltas.  Index levels get a thousands separator; rates
    and OAS levels are already in percent magnitude (e.g. US2Y ``4.78`` = 4.78%)
    so they get a ``%`` suffix; IV cells are decimal vols scaled to percent.
    """

    if not _is_finite(value):
        return "n/a"
    val = float(value)
    market_key = str(market or "").upper()
    if market_key == "IV_SURFACE" or market_key.startswith("IV_"):
        # Implied vol cannot be negative — floor the RENDERED level at 0%
        # (display-only; does not touch sampling/fan_scale/734a).
        return f"{max(val, 0.0) * 100.0:.2f}%"
    if market_key in {"US2Y", "US10Y", "AAA_OAS", "BBB_OAS"}:
        return f"{val:.2f}%"
    if market_key in {"SPX", "NIKKEI"}:
        return f"{val:,.1f}"
    if market_key in {"USDJPY", "USDCAD", "DXY", "VIX"}:
        return f"{val:.2f}"
    if market_key in {"GOLD", "CRUDE_OIL", "COPPER", "WHEAT"}:
        return f"{val:,.2f}"
    return f"{val:,.2f}"


def terminal_day30_table(report: dict[str, Any]) -> pd.DataFrame:
    """Per-factor terminal Day-30 quantiles: baseline vs conditioned.

    Uses the same raw-level conversion as the fan chart so units match, and
    takes the last element of each quantile series.  Degrades to an empty
    frame when no path-quantile rows are present.
    """

    generation = _as_dict(report.get("generation"))
    scope = _default_analogue_scope(report)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for order_idx, raw_row in enumerate(_as_list(generation.get("path_quantiles"))):
        if not isinstance(raw_row, dict):
            continue
        market = str(raw_row.get("market", "")).strip()
        if not market or market in seen:
            continue
        seen.add(market)
        cond_row = _path_quantile_row_as_raw_level(
            report,
            _path_quantile_row(report, market, scope),
        )
        if not cond_row:
            continue
        baseline_row = _baseline_path_quantile_row_as_raw_level(report, market)
        c10, c50, c90 = _terminal_quantiles_for_row(cond_row)
        b10, b50, b90 = _terminal_quantiles_for_row(baseline_row)
        display = str(
            cond_row.get("display_name") or cond_row.get("market") or market
        )
        # Macro factors the narrative names (SPX, DXY, ...) lead; the IV-surface
        # cells follow. Stable sort preserves original order within each group.
        is_iv = 1 if market.upper().startswith("IV") else 0
        rows.append(
            (
                (is_iv, order_idx),
                {
                    "Factor": display,
                    "Baseline P10": _terminal_level_text(market, b10),
                    "Baseline P50": _terminal_level_text(market, b50),
                    "Baseline P90": _terminal_level_text(market, b90),
                    "Conditioned P10": _terminal_level_text(market, c10),
                    "Conditioned P50": _terminal_level_text(market, c50),
                    "Conditioned P90": _terminal_level_text(market, c90),
                },
            )
        )
    rows.sort(key=lambda item: item[0])
    return _frame([row for _key, row in rows], TERMINAL_DAY30_COLUMNS)


def calendar_start_label(
    report: dict[str, Any] | None,
    explicit_start_window_index: float | int | None = None,
    *,
    bridge_report_path: str | Path = DEFAULT_PREFIX_FULL_START_BRIDGE_REPORT,
) -> str:
    """Render a calendar 'as of YYYY-MM-DD (window W)' start label.

    Resolves the day-0 as-of date from the bridge-local window metadata keyed
    by the explicit start window index (the same key the start-state bank
    uses).  Degrades to 'as of n/a' when the index or metadata is unavailable.
    """

    start_index: int | None = None
    if explicit_start_window_index is not None:
        try:
            start_index = int(explicit_start_window_index)
        except (TypeError, ValueError):
            start_index = None
    if start_index is None:
        return (
            '<div class="calendar-start-strip">'
            "<strong>Day-0 start:</strong> as of n/a "
            '<span class="ess-note">(no start window index)</span></div>'
        )
    window_id = ""
    as_of = ""
    try:
        path = Path(bridge_report_path)
        if path.exists():
            bridge_report = json.loads(path.read_text(encoding="utf-8"))
            metadata = window_metadata_by_bridge_local_index(bridge_report)
            entry = _as_dict(metadata.get(int(start_index)))
            window_id = str(entry.get("window_id", "") or "")
            calendar = _as_dict(entry.get("calendar"))
            as_of = str(calendar.get("calendar_end_date", "") or "")
    except Exception:  # pragma: no cover - defensive UI path
        window_id = ""
        as_of = ""
    if as_of:
        # A calendar date is the human-recognizable day-0 label; the internal
        # window id is omitted from the primary strip (it stays in the audit
        # JSON for traceability).
        return (
            '<div class="calendar-start-strip">'
            f"<strong>Day-0 start:</strong> as of {html.escape(as_of)}</div>"
        )
    window_text = window_id if window_id else f"index {start_index}"
    return (
        '<div class="calendar-start-strip">'
        f"<strong>Day-0 start:</strong> as of n/a "
        f"(window {html.escape(window_text)})</div>"
    )


@lru_cache(maxsize=4)
def _bridge_window_metadata(bridge_report_path: str) -> dict[int, Any]:
    """Cached bridge-local window metadata (window_id + calendar) keyed by index.

    Reads the bridge report JSON once per path; used by the day-0 start bounds
    and the index→date hint so the live ``change`` handler does not re-parse the
    report on every keystroke.  Returns an empty mapping on any failure.
    """

    try:
        path = Path(bridge_report_path)
        if not path.exists():
            return {}
        bridge_report = json.loads(path.read_text(encoding="utf-8"))
        metadata = window_metadata_by_bridge_local_index(bridge_report)
        return {int(k): v for k, v in metadata.items()}
    except Exception:  # pragma: no cover - defensive UI path
        return {}


def _start_index_bounds(
    bridge_report_path: str | Path = DEFAULT_PREFIX_FULL_START_BRIDGE_REPORT,
) -> tuple[int, int]:
    """Inclusive [min, max] day-0 window index derived from the loaded start bank."""

    keys = list(_bridge_window_metadata(str(bridge_report_path)).keys())
    if not keys:
        return (0, 0)
    return (min(keys), max(keys))


def start_index_date_hint(
    explicit_start_window_index: float | int | None,
    *,
    bridge_report_path: str | Path = DEFAULT_PREFIX_FULL_START_BRIDGE_REPORT,
) -> str:
    """Resolve a day-0 window index to a calendar date for the live input hint.

    Validates against the loaded start bank's index range and renders a plain
    'as of <date>' line, or an out-of-range warning, so the risk manager sees
    what state they picked before spending the paid grounding call.
    """

    lo, hi = _start_index_bounds(bridge_report_path)
    idx: int | None = None
    if explicit_start_window_index is not None:
        try:
            idx = int(explicit_start_window_index)
        except (TypeError, ValueError):
            idx = None
    if idx is None:
        return (
            '<div class="calendar-start-strip"><strong>Starting market state:</strong> '
            f"enter a historical window from {lo} to {hi}.</div>"
        )
    if idx < lo or idx > hi:
        return (
            '<div class="calendar-start-strip" style="border-left:4px solid #C62828;">'
            f"<strong>Starting market state:</strong> index {idx} is out of range "
            f"(valid {lo}&ndash;{hi}). Pick a window in this range before generating.</div>"
        )
    entry = _as_dict(_bridge_window_metadata(str(bridge_report_path)).get(idx))
    as_of = str(_as_dict(entry.get("calendar")).get("calendar_end_date", "") or "")
    if as_of:
        return (
            '<div class="calendar-start-strip"><strong>Starting market state:</strong> '
            f"as of {html.escape(as_of)} (historical window {idx}).</div>"
        )
    return (
        '<div class="calendar-start-strip"><strong>Starting market state:</strong> '
        f"historical window {idx}.</div>"
    )


def _start_window_calendar_label(
    window_index: float | int | None,
    fallback_id: str,
    *,
    bridge_report_path: str | Path = DEFAULT_PREFIX_FULL_START_BRIDGE_REPORT,
) -> str:
    """Resolve a day-0 window index to its calendar start date for display.

    Falls back to the raw window id only when the index is missing or not in the
    loaded start bank (e.g. synthetic reports), so the primary view shows a
    recognizable date rather than an internal id.
    """

    idx: int | None = None
    if window_index is not None:
        try:
            idx = int(window_index)
        except (TypeError, ValueError):
            idx = None
    if idx is not None:
        entry = _as_dict(_bridge_window_metadata(str(bridge_report_path)).get(idx))
        as_of = str(_as_dict(entry.get("calendar")).get("calendar_end_date", "") or "")
        if as_of:
            return as_of
    return str(fallback_id or "")


@lru_cache(maxsize=2)
def full_start_date_choices(
    bridge_report_path: str = DEFAULT_PREFIX_FULL_START_BRIDGE_REPORT,
) -> list[tuple[str, int]]:
    """Searchable (date label -> window index) choices for the day-0 start picker.

    One entry per train-region window in the full start bank, chronological, so
    the dropdown is filterable by typing a year (e.g. "2008"). Degrades to a
    single placeholder when the bank has not been built yet.
    """

    meta = _bridge_window_metadata(str(bridge_report_path))
    choices: list[tuple[str, int]] = []
    for idx in sorted(meta.keys()):
        as_of = str(_as_dict(meta[idx].get("calendar")).get("calendar_end_date", "") or "")
        # Label is the day-0 date only (no internal window id); the integer index
        # remains the dropdown VALUE so the run wiring is unchanged.
        label = as_of if as_of else f"window {idx}"
        choices.append((label, int(idx)))
    if not choices:
        choices = [("start bank not built — run build_train_region_full_start_bridge.py", 0)]
    return choices


def _default_full_start_index(
    bridge_report_path: str = DEFAULT_PREFIX_FULL_START_BRIDGE_REPORT,
) -> int:
    """Default day-0 selection: the late-Oct-2008 GFC crash window when available.

    Selected on the DAY-0 date (calendar_end_date = last observed history day) so
    the shown date and the day-0 SPX level agree (~2008-10-24, SPX ~877).
    """

    meta = _bridge_window_metadata(str(bridge_report_path))
    keys = sorted(meta.keys())
    if not keys:
        return 0
    for idx in keys:
        day0 = str(_as_dict(meta[idx].get("calendar")).get("calendar_end_date", "") or "")
        if day0 >= "2008-10-24":  # late-Oct 2008 crash; ISO dates sort lexicographically
            return int(idx)
    return int(keys[len(keys) // 2])


def load_validation_gate_report(
    path: str | Path = DEFAULT_PREFIX_VALIDATION_GATE_REPORT,
) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


def load_boss_demo_pack(
    path: str | Path = DEFAULT_BOSS_DEMO_PACK_JSON,
) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


def boss_demo_status_strip(report: dict[str, Any]) -> str:
    if not report:
        return (
            "**Validation evidence:** not available. Optional casebook details "
            "below are empty until the boss demo pack is generated."
        )
    snapshot = _as_dict(report.get("validation_snapshot"))
    live_snapshot = _as_dict(report.get("live_casebook_snapshot"))
    caption_audit = _as_dict(report.get("fixed_start_caption_audit_snapshot"))
    run_count = snapshot.get("run_count", 0)
    case_count = live_snapshot.get("case_count", 0)
    audit_status = str(caption_audit.get("status", "") or "n/a")
    return (
        "**Validation evidence:** "
        f"offline CRPS `{snapshot.get('improved_crps_rows', 0)}/{run_count}`, "
        f"energy `{snapshot.get('improved_energy_rows', 0)}/{run_count}`, "
        f"live API casebook `{live_snapshot.get('pass_count', 0)}/{case_count}` pass, "
        f"fixed-start caption audit `{audit_status}`. "
        "These are demo-readiness checks; full details below are optional."
    )


def boss_demo_pack_markdown(report: dict[str, Any]) -> str:
    if not report:
        return "\n".join(
            [
                "## Demo readiness evidence",
                "",
                "- Status: `not available`",
                "- Generate the boss demo pack to populate this section.",
            ]
        )
    snapshot = _as_dict(report.get("validation_snapshot"))
    live_snapshot = _as_dict(report.get("live_casebook_snapshot"))
    caption_audit = _as_dict(report.get("fixed_start_caption_audit_snapshot"))
    artifact_paths = _as_dict(report.get("artifact_paths"))
    live_models = ", ".join(_as_list(live_snapshot.get("grounding_models"))) or "n/a"
    embedding_models = (
        ", ".join(_as_list(live_snapshot.get("embedding_models"))) or "n/a"
    )
    evidence_path = str(artifact_paths.get("summary_markdown", ""))
    evidence_label = Path(evidence_path).name if evidence_path else "n/a"
    lines = [
            "## Demo readiness evidence",
            "",
            f"- Evidence pack: `{evidence_label}`",
            f"- Offline validation: `{snapshot.get('run_count', 0)}` runs.",
            f"- CRPS improved `{snapshot.get('improved_crps_rows', 0)}/{snapshot.get('run_count', 0)}`.",
            f"- Energy improved `{snapshot.get('improved_energy_rows', 0)}/{snapshot.get('run_count', 0)}`.",
            f"- Offline mean CRPS improvement: `{_fmt_pct(snapshot.get('mean_crps_improvement_vs_persistence'))}`.",
            f"- Live API casebook: `{live_snapshot.get('pass_count', 0)}/{live_snapshot.get('case_count', 0)}` pass.",
            f"- Fixed-start caption audit: `{caption_audit.get('status', 'n/a')}`.",
            f"- OpenAI tokens `{live_snapshot.get('total_openai_tokens', 0)}`.",
            f"- Min support candidates `{live_snapshot.get('min_support_candidate_count', 0)}`.",
            f"- Grounding model: `{live_models}`.",
            f"- Embedding model: `{embedding_models}`.",
            "- Contract:",
            "- Current/recent implications are conditioning inputs.",
            "- Forward-risk language is warning-only.",
        ]
    if caption_audit:
        prof_vs_null = _as_dict(caption_audit.get("professional_minus_start_only"))
        prof_vs_simple = _as_dict(caption_audit.get("professional_minus_simple"))
        lines.extend(
            [
                "- Fixed-start evidence:",
                f"- Professional vs start-only factor KS delta `{_fmt_float(prof_vs_null.get('factor_terminal_ks'))}`.",
                f"- Professional vs start-only portfolio KS delta `{_fmt_float(prof_vs_null.get('portfolio_terminal_ks'))}`.",
                f"- Professional vs simple path-energy delta `{_fmt_float(prof_vs_simple.get('path_energy'))}`.",
            ]
        )
    return "\n".join(lines)


def boss_demo_live_casebook_table(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    live_snapshot = _as_dict(report.get("live_casebook_snapshot"))
    for item in _as_list(live_snapshot.get("case_rows")):
        if not isinstance(item, dict):
            continue
        summary_path = str(item.get("summary_path", ""))
        rows.append(
            {
                "Case": str(item.get("case_name", "")),
                "Start": str(item.get("expected_start_index", "")),
                "Status": str(item.get("overall_status", "")),
                "Condition": str(item.get("condition_only_validation_status", "")),
                "Warnings": str(item.get("forward_warning_count", "")),
                "Support": str(item.get("support_candidate_count", "")),
                "Summary": Path(summary_path).name if summary_path else "",
            }
        )
    return _frame(rows, BOSS_DEMO_CASEBOOK_COLUMNS)


def _count_items_text(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return ", ".join(f"{key}={value[key]}" for key in sorted(value))


def validation_gate_markdown(report: dict[str, Any]) -> str:
    gate = _as_dict(report.get("gate"))
    if not gate:
        return "\n".join(
            [
                "## Latent-prefix validation",
                "",
                "- Status: `not available`",
                "- Run the cached validation gate to populate this section.",
            ]
        )
    return "\n".join(
        [
            "## Latent-prefix validation",
            "",
            f"- Operational status: `{gate.get('operational_status', 'n/a')}`",
            f"- Stress status: `{gate.get('stress_status', 'n/a')}`",
            f"- Overall diagnostic status: `{gate.get('overall_status', 'n/a')}`",
            f"- Endpoint max error: `{_fmt_float(gate.get('endpoint_max_abs_error'), 6)}`",
            f"- Warnings: `{_count_items_text(gate.get('warning_counts'))}`",
            f"- Failures: `{_count_items_text(gate.get('fail_counts'))}`",
        ]
    )


def validation_gate_table(report: dict[str, Any]) -> pd.DataFrame:
    gate = _as_dict(report.get("gate"))
    rows: list[dict[str, Any]] = []
    for item in _as_list(gate.get("hard_cases")):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Variant": str(item.get("variant", "")),
                "Role": str(item.get("case_role", "")),
                "Query": str(item.get("query_window_index", "")),
                "Start": str(item.get("start_window_index", "")),
                "Status": str(item.get("status", "")),
                "Memory Cosine": _fmt_float(item.get("input_memory_cosine")),
                "Distance from start (σ)": _fmt_float(item.get("start_distance_z")),
                "Terminal Shift": _fmt_float(item.get("terminal_mean_abs_delta_z")),
                "Warnings": ", ".join(str(x) for x in _as_list(item.get("warnings"))),
                "Failures": ", ".join(str(x) for x in _as_list(item.get("failures"))),
            }
        )
    return _frame(rows, VALIDATION_GATE_COLUMNS)


def prefix_variant_table(report: dict[str, Any]) -> pd.DataFrame:
    return _prefix_variant_table_for_role(report, role=None)


def prefix_selected_start_table(report: dict[str, Any]) -> pd.DataFrame:
    row = _operational_variant_row(report)
    rows: list[dict[str, Any]] = []
    if row:
        rows.append(
            {
                "Starting Level": _start_window_calendar_label(
                    row.get("start_window_index"),
                    str(row.get("start_window_id", "")),
                ),
                "Narrative match (cosine)": _fmt_float(
                    row.get("memory_support_cosine")
                ),
            }
        )
    return _frame(rows, PREFIX_SELECTED_START_COLUMNS)


def prefix_diagnostic_start_table(report: dict[str, Any]) -> pd.DataFrame:
    return _prefix_variant_table_for_role(report, role="diagnostic")


def _prefix_variant_table_for_role(
    report: dict[str, Any],
    *,
    role: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in _as_list(report.get("variant_rows")):
        if not isinstance(item, dict):
            continue
        is_operational = bool(
            item.get(
                "is_operational",
                str(item.get("variant", "")) != "original",
            )
        )
        if role == "operational" and not is_operational:
            continue
        if role == "diagnostic" and is_operational:
            continue
        rows.append(
            {
                "Variant": str(item.get("variant", "")),
                "Query Window": str(item.get("query_window_id", "")),
                "Start Window": str(item.get("start_window_id", "")),
                "Distance from start (σ)": _fmt_float(item.get("start_distance_z")),
                "Memory Support": _fmt_float(item.get("memory_support_cosine")),
                "Selection": str(item.get("start_selection_method", "")),
                "Start Split": str(item.get("start_manifest_split", "")),
            }
        )
    return _frame(rows, PREFIX_VARIANT_COLUMNS)


def prefix_validation_table(report: dict[str, Any]) -> pd.DataFrame:
    gate = _as_dict(report.get("validation_gate"))
    rows: list[dict[str, Any]] = []
    for item in _as_list(gate.get("cases")):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Variant": str(item.get("variant", "")),
                "Role": str(item.get("case_role", "")),
                "Query": str(item.get("query_window_index", "")),
                "Start": str(item.get("start_window_index", "")),
                "Status": str(item.get("status", "")),
                "Memory Cosine": _fmt_float(item.get("input_memory_cosine")),
                "Distance from start (σ)": _fmt_float(item.get("start_distance_z")),
                "Terminal Shift": _fmt_float(item.get("terminal_mean_abs_delta_z")),
                "Warnings": ", ".join(str(x) for x in _as_list(item.get("warnings"))),
                "Failures": ", ".join(str(x) for x in _as_list(item.get("failures"))),
            }
        )
    return _frame(rows, VALIDATION_GATE_COLUMNS)


def _episode_label(item: dict[str, Any]) -> str:
    """Human-recognizable episode label: the history-end date when present,
    otherwise the internal window id with the ``joint39_`` prefix stripped."""

    date = str(item.get("history_end_date", "") or "").strip()
    if date:
        return date
    window = str(item.get("window_id") or item.get("window_index", "") or "").strip()
    return window.replace("joint39_", "").replace("_", " ").strip() or window


def prefix_start_candidates_table(report: dict[str, Any]) -> pd.DataFrame:
    query = _as_dict(report.get("cached_query"))
    memory_prior = _as_dict(query.get("memory_prior"))
    generation = _as_dict(report.get("generation"))
    posterior = _as_dict(generation.get("posterior_ensemble"))
    candidate_rows = _as_list(posterior.get("selected_support")) or _as_list(
        memory_prior.get("candidate_details")
    )
    rows: list[dict[str, Any]] = []
    def append_rows(items: list[Any], *, used_for: str) -> None:
        for rank, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            checked = item.get("recent_prefix_checked")
            mismatches = item.get("recent_prefix_mismatches")
            if checked is None:
                direction_check = _fmt_float(item.get("recent_prefix_alignment_score"))
            else:
                try:
                    mismatch_count = int(mismatches or 0)
                    checked_count = int(checked)
                except (TypeError, ValueError):
                    mismatch_count = 0
                    checked_count = 0
                matched = max(checked_count - mismatch_count, 0)
                # Green when every checked direction matched, amber otherwise.
                # gr.Dataframe cells cannot carry CSS classes like the HTML
                # scenario table, so the colour cue is a status dot in the value.
                cue = "🟢" if mismatch_count == 0 else "🟠"
                direction_check = f"{cue} {matched}/{checked_count} matched"
            rows.append(
                {
                    "Used For": used_for,
                    "Rank": int(item.get("rank", rank)),
                    "Episode date": _episode_label(item),
                    "Weight": _fmt_float(item.get("weight")),
                    "Narrative match (cosine)": _fmt_float(
                        item.get("memory_support_cosine")
                    ),
                    "Distance from start (σ)": _fmt_float(item.get("start_distance_z")),
                    "Narrative directions": direction_check,
                }
            )

    append_rows(candidate_rows, used_for="Narrative scenario")
    baseline = _as_dict(generation.get("start_only_baseline"))
    baseline_support = _as_list(
        _as_dict(baseline.get("posterior_ensemble")).get("selected_support")
    ) or _as_list(baseline.get("support_candidates"))
    append_rows(baseline_support, used_for="Start-only baseline")
    return _frame(rows, PREFIX_START_CANDIDATE_COLUMNS)


def prefix_user_start_table(report: dict[str, Any]) -> pd.DataFrame:
    user_start = _as_dict(report.get("user_start_state"))
    if not user_start:
        return _frame([], PREFIX_USER_START_COLUMNS)
    user_row = None
    for item in _as_list(report.get("variant_rows")):
        if isinstance(item, dict) and str(item.get("variant")) == "user_start_state":
            user_row = item
            break
    nearest = (
        ""
        if user_row is None
        else str(user_row.get("nearest_train_start_window_index", ""))
    )
    distance = "" if user_row is None else _fmt_float(user_row.get("start_distance_z"))
    max_abs_z = (
        "" if user_row is None else _fmt_float(user_row.get("max_abs_user_start_z"))
    )
    return _frame(
        [
            {
                "Label": str(user_start.get("label", "")),
                "Source": str(user_start.get("source_path", "")),
                "Format": str(user_start.get("source_format", "")),
                "Dimension": str(user_start.get("dimension", "")),
                "Nearest Train": nearest,
                "Distance from start (σ)": distance,
                "Max Abs Z": max_abs_z,
            }
        ],
        PREFIX_USER_START_COLUMNS,
    )


def preview_start_state_json(path: str | None) -> tuple[str, pd.DataFrame]:
    candidate = str(path or "").strip()
    if not candidate:
        return (
            "## Start-State JSON Preview\n\n- Status: `waiting`\n- Enter a JSON path.",
            _frame([], PREFIX_START_PREVIEW_COLUMNS),
        )
    try:
        payload = json.loads(Path(candidate).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("start-state JSON must contain an object")
        label = str(payload.get("label", ""))
        coordinate = str(payload.get("coordinate", "raw_state"))
        rows = [
            {"Field": "Path", "Value": candidate},
            {"Field": "Label", "Value": label},
            {"Field": "Coordinate", "Value": coordinate},
        ]
        values = payload.get("values_by_name")
        vector = payload.get("state_vector")
        if isinstance(values, dict):
            rows.append({"Field": "Format", "Value": "values_by_name"})
            rows.append({"Field": "Field count", "Value": str(len(values))})
            for label_name, key in START_PREVIEW_FIELDS:
                if key in values:
                    rows.append({"Field": label_name, "Value": _fmt_float(values[key])})
            missing_preview = [
                label_name
                for label_name, key in START_PREVIEW_FIELDS
                if key not in values
            ]
            if missing_preview:
                rows.append(
                    {
                        "Field": "Missing preview fields",
                        "Value": ", ".join(missing_preview),
                    }
                )
        elif isinstance(vector, list):
            rows.append({"Field": "Format", "Value": "state_vector"})
            rows.append({"Field": "Vector length", "Value": str(len(vector))})
            if vector:
                rows.append({"Field": "First value", "Value": _fmt_float(vector[0])})
        else:
            raise ValueError("JSON must contain values_by_name or state_vector")
    except Exception as error:
        return (
            "## Start-State JSON Preview\n\n"
            f"- Status: `error`\n"
            f"- Error type: `{type(error).__name__}`\n"
            f"- Message: `{str(error)}`",
            _frame([], PREFIX_START_PREVIEW_COLUMNS),
        )
    return (
        "## Start-State JSON Preview\n\n"
        f"- Status: `ok`\n"
        f"- Format: `{rows[3]['Value'] if len(rows) > 3 else 'unknown'}`",
        _frame(rows, PREFIX_START_PREVIEW_COLUMNS),
    )


def build_start_state_payload(
    *,
    label: str,
    spec_names: list[str],
    raw_state: Any,
) -> dict[str, Any]:
    values = [float(value) for value in list(raw_state)]
    if len(values) != len(spec_names):
        raise ValueError("raw_state length must match spec_names")
    return {
        "label": str(label),
        "coordinate": "raw_state",
        "values_by_name": {
            str(name): float(value)
            for name, value in zip(spec_names, values, strict=True)
        },
    }


def load_joint39_start_bank_for_app() -> dict[str, Any]:
    bridge_report = json.loads(Path(DEFAULT_PREFIX_BRIDGE_REPORT).read_text())
    selected_windows = selected_bridge_window_indices(bridge_report)
    args = SimpleNamespace(
        state_scope="joint38",
        test_start=4511,
        val_size=441,
        max_windows=441,
        eval_split="val",
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
    _model, payload = load_model(DEFAULT_PREFIX_CHECKPOINT, torch.device("cpu"))
    (
        _all_history_level,
        _all_history_norm,
        _all_center,
        _all_scale,
        _all_drift,
        all_history_raw,
        specs,
        _block,
    ) = build_val_block(args, payload)
    return {
        "history_raw": all_history_raw[selected_windows],
        "spec_names": _spec_names(specs),
        "metadata": window_metadata_by_bridge_local_index(bridge_report),
    }


def export_historical_start_json_for_app(
    candidate_choice: str | int | float | None,
    explicit_start_window_index: float | int | None = None,
    *,
    output_dir: str | Path = DEFAULT_PREFIX_APP_OUTPUT_DIR,
    bank_loader: Callable[[], dict[str, Any]] = load_joint39_start_bank_for_app,
) -> tuple[str, str, pd.DataFrame]:
    start_index = historical_start_candidate_to_index(candidate_choice)
    if start_index is None and explicit_start_window_index is not None:
        start_index = historical_start_candidate_to_index(explicit_start_window_index)
    if start_index is None:
        return (
            "## Export Start JSON\n\n- Status: `error`\n- Message: `Select a historical candidate or enter an index first.`",
            "",
            _frame([], PREFIX_START_PREVIEW_COLUMNS),
        )
    try:
        bank = bank_loader()
        history_raw = np.asarray(bank["history_raw"], dtype=np.float32)
        spec_names = list(bank["spec_names"])
        metadata = _as_dict(bank.get("metadata")).get(int(start_index), {})
        if start_index < 0 or start_index >= len(history_raw):
            raise IndexError(f"start index {start_index} outside {len(history_raw)}")
        window_id = str(
            _as_dict(metadata).get("window_id") or f"joint39_start_{start_index:04d}"
        )
        label = f"user_template_from_{window_id}"
        payload = build_start_state_payload(
            label=label,
            spec_names=spec_names,
            raw_state=history_raw[int(start_index), -1, :],
        )
        output_path = Path(output_dir) / f"user_start_template_{start_index:04d}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        preview_status, preview = preview_start_state_json(str(output_path))
    except Exception as error:
        return (
            "## Export Start JSON\n\n"
            f"- Status: `error`\n"
            f"- Error type: `{type(error).__name__}`\n"
            f"- Message: `{str(error)}`",
            "",
            _frame([], PREFIX_START_PREVIEW_COLUMNS),
        )
    status = (
        "## Export Start JSON\n\n"
        f"- Status: `ok`\n"
        f"- Start index: `{start_index}`\n"
        f"- Path: `{output_path}`\n\n"
        f"{preview_status}"
    )
    return status, str(output_path), preview


def historical_start_candidate_choices(report: dict[str, Any]) -> list[tuple[str, str]]:
    query = _as_dict(report.get("cached_query"))
    memory_prior = _as_dict(query.get("memory_prior"))
    choices: list[tuple[str, str]] = []
    for item in _as_list(memory_prior.get("candidate_details")):
        if not isinstance(item, dict):
            continue
        bridge_index = item.get("bridge_local_index", item.get("window_index"))
        if bridge_index is None:
            continue
        window = str(item.get("window_id") or f"window_{bridge_index}")
        label = (
            f"{window} | idx {bridge_index} | "
            f"w {_fmt_float(item.get('weight'))} | "
            f"start {_fmt_float(item.get('start_distance_z'))}z"
        )
        choices.append((label, str(int(bridge_index))))
    return choices


def historical_start_candidate_update(report: dict[str, Any]) -> Any:
    import gradio as gr

    choices = historical_start_candidate_choices(report)
    value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=value)


def historical_start_candidate_to_index(choice: str | int | float | None) -> int | None:
    if choice in (None, ""):
        return None
    try:
        return int(float(choice))
    except (TypeError, ValueError):
        return None


def prefix_warning_component_table(report: dict[str, Any]) -> pd.DataFrame:
    product_gate = _as_dict(report.get("condition_only_product_gate"))
    decompositions = _as_list(product_gate.get("decompositions"))
    selected = {}
    for item in decompositions:
        if isinstance(item, dict) and item.get("start_mode") in {
            "balanced_memory_start",
            "implication_aligned_start",
        }:
            selected = item
            break
    if not selected and decompositions and isinstance(decompositions[0], dict):
        selected = decompositions[0]
    components = _as_dict(selected.get("components")) if selected else {}
    rows: list[dict[str, Any]] = []
    for name, item in components.items():
        if not isinstance(item, dict):
            continue
        metric = ""
        value: Any = ""
        for candidate in [
            "input_memory_cosine",
            "start_distance_z",
            "endpoint_max_abs_error",
            "terminal_mean_abs_delta_z",
            "mismatch_count",
        ]:
            if candidate in item:
                metric = candidate
                value = item[candidate]
                break
        rows.append(
            {
                "Check": str(name),
                "Status": str(item.get("status", "")),
                "Metric": metric,
                "Value": _fmt_float(
                    value, 6 if metric == "endpoint_max_abs_error" else 3
                ),
            }
        )
    return _frame(rows, PREFIX_WARNING_COMPONENT_COLUMNS)


def prefix_shift_factor_table(report: dict[str, Any]) -> pd.DataFrame:
    product_gate = _as_dict(report.get("condition_only_product_gate"))
    rows: list[dict[str, Any]] = []
    for item in _as_list(product_gate.get("decompositions")):
        if not isinstance(item, dict):
            continue
        mode = str(item.get("start_mode", ""))
        for factor in _as_list(item.get("top_rollout_shift_factors"))[:8]:
            if not isinstance(factor, dict):
                continue
            rows.append(
                {
                    "Start Mode": mode,
                    "Factor": str(factor.get("factor", "")),
                    "Terminal Abs Z": _fmt_float(factor.get("terminal_abs_shift_z")),
                    "Signed Terminal Z": _fmt_float(
                        factor.get("signed_terminal_shift_z")
                    ),
                    "Path Abs Z": _fmt_float(factor.get("mean_path_abs_shift_z")),
                }
            )
    return _frame(rows, PREFIX_SHIFT_FACTOR_COLUMNS)


def _default_analogue_scope(report: dict[str, Any]) -> str:
    posterior = _as_dict(_as_dict(report.get("generation")).get("posterior_ensemble"))
    default_key = str(posterior.get("default_analogue_key") or "").strip()
    if default_key:
        return default_key
    generation = _as_dict(report.get("generation"))
    path_rows = _as_list(generation.get("path_quantiles"))
    for row in path_rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("analogue_label") or "")
        key = str(row.get("analogue_key") or "")
        if key and label.startswith("Selected start:"):
            return key
    variant_rows = _as_list(report.get("variant_rows"))
    for rank, item in enumerate(variant_rows, start=1):
        if not isinstance(item, dict) or not bool(item.get("is_operational")):
            continue
        key = f"RANK_{rank}"
        if any(
            isinstance(row, dict) and str(row.get("analogue_key")) == key
            for row in path_rows
        ):
            return key
    return "ALL"


def _all_scope_is_mixed_start_diagnostic(report: dict[str, Any]) -> bool:
    generation = _as_dict(report.get("generation"))
    for row in _as_list(generation.get("path_quantiles")):
        if not isinstance(row, dict):
            continue
        if str(row.get("analogue_key", "ALL")) != "ALL":
            continue
        label = str(row.get("analogue_label") or "").lower()
        if "start variants" in label:
            return True
    return False


def _resolve_analogue_scope(report: dict[str, Any], analogue_scope: str) -> str:
    requested = str(analogue_scope or "ALL")
    default_scope = _default_analogue_scope(report)
    if (
        requested == "ALL"
        and default_scope != "ALL"
        and _all_scope_is_mixed_start_diagnostic(report)
    ):
        return default_scope
    return requested


def analogue_scope_choices(report: dict[str, Any]) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    posterior = _as_dict(_as_dict(report.get("generation")).get("posterior_ensemble"))
    default_key = str(posterior.get("default_analogue_key") or "").strip()
    if default_key:
        choices.append(
            (
                str(posterior.get("label") or TOP3_90_ENSEMBLE_LABEL),
                default_key,
            )
        )
    default_scope = _default_analogue_scope(report)
    if default_key:
        pass
    elif not _all_scope_is_mixed_start_diagnostic(report) or default_scope == "ALL":
        if not any(value == "ALL" for _, value in choices):
            choices.append(("All retrieved analogues", "ALL"))
    else:
        if not any(value == default_scope for _, value in choices):
            choices.append(("Operational selected start", default_scope))
    added = False
    for rank, item in enumerate(_as_list(report.get("historical_analogues")), start=1):
        if not isinstance(item, dict):
            continue
        window_id = str(item.get("window_id", f"analogue_{rank}"))
        choices.append((f"Analogue {rank}: {window_id}", f"RANK_{rank}"))
        added = True
    if added:
        return choices
    seen: set[str] = {value for _, value in choices}
    for row in _as_list(_as_dict(report.get("generation")).get("path_quantiles")):
        if not isinstance(row, dict):
            continue
        key = str(row.get("analogue_key", "ALL"))
        if key in seen or key == "ALL":
            continue
        label = str(row.get("analogue_label") or key)
        if label.startswith("Diagnostic baseline:"):
            continue
        choices.append((label, key))
        seen.add(key)
    return choices


def analogue_scope_update(report: dict[str, Any]) -> Any:
    import gradio as gr

    return gr.update(
        choices=analogue_scope_choices(report),
        value=_default_analogue_scope(report),
    )


def _path_quantile_row(
    report: dict[str, Any],
    market: str,
    analogue_scope: str = "ALL",
) -> dict[str, Any]:
    generation = _as_dict(report.get("generation"))
    requested = str(market or "SPX")
    requested_scope = str(analogue_scope or "ALL")
    rows = _as_list(generation.get("path_quantiles"))
    for row in rows:
        if (
            isinstance(row, dict)
            and str(row.get("market")) == requested
            and str(row.get("analogue_key", "ALL")) == requested_scope
        ):
            return row
    for row in rows:
        if (
            isinstance(row, dict)
            and str(row.get("market")) == requested
            and str(row.get("analogue_key", "ALL")) == "ALL"
        ):
            return row
    for row in rows:
        if (
            isinstance(row, dict)
            and str(row.get("market")) == "SPX"
            and str(row.get("analogue_key", "ALL")) == requested_scope
        ):
            return row
    for row in rows:
        if (
            isinstance(row, dict)
            and str(row.get("market")) == "SPX"
            and str(row.get("analogue_key", "ALL")) == "ALL"
        ):
            return row
    return {}


def _series_plus_start(values: Any, start_level: float | None) -> list[float]:
    series = _float_series(values)
    if not _is_finite(start_level):
        return series
    return [float(start_level) + float(value) for value in series]


def _path_quantile_row_as_raw_level(
    report: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    """Return a raw-level fan-chart row, converting legacy delta rows if needed."""

    if not row:
        return {}
    converted = dict(row)
    market = str(converted.get("market", ""))
    start_level = converted.get("start_level")
    if not _is_finite(start_level):
        start_level = _market_start_level(report, market)
    if _is_finite(start_level):
        converted["start_level"] = float(start_level)
    if str(converted.get("value_kind", "")).lower() == "raw_level":
        converted["value_kind"] = "raw_level"
        return converted
    for key in ("p10", "p50", "p90", "mean", "realized_path"):
        converted[key] = _series_plus_start(converted.get(key), start_level)
    sample_paths: list[dict[str, Any]] = []
    for path in _as_list(converted.get("sample_paths")):
        if not isinstance(path, dict):
            continue
        updated_path = dict(path)
        updated_path["values"] = _series_plus_start(path.get("values"), start_level)
        sample_paths.append(updated_path)
    converted["sample_paths"] = sample_paths
    converted["value_kind"] = "raw_level"
    converted["converted_from"] = str(row.get("value_kind") or "legacy_delta")
    return converted


def _baseline_path_quantile_row_as_raw_level(
    report: dict[str, Any],
    market: str,
) -> dict[str, Any]:
    """Return the start-only baseline fan row for the given market, converted to raw level.

    The baseline is stored at generation["start_only_baseline"]["path_quantiles"]
    by attach_start_only_baseline_report().  Returns {} if not present.
    """
    generation = _as_dict(report.get("generation"))
    baseline = _as_dict(generation.get("start_only_baseline"))
    baseline_rows = _as_list(baseline.get("path_quantiles"))
    requested = str(market or "SPX")
    # Search for a matching market row (no analogue_scope filter for baseline)
    baseline_row: dict[str, Any] = {}
    for candidate in baseline_rows:
        if isinstance(candidate, dict) and str(candidate.get("market")) == requested:
            baseline_row = candidate
            break
    if not baseline_row:
        for candidate in baseline_rows:
            if isinstance(candidate, dict) and str(candidate.get("market")) == "SPX":
                baseline_row = candidate
                break
    if not baseline_row:
        return {}
    return _path_quantile_row_as_raw_level(report, baseline_row)


def _prepend_start_to_series(
    days: list[float],
    values: list[float],
    start_level: float | None,
) -> tuple[list[float], list[float]]:
    if not (_is_finite(start_level) and values):
        return days, values
    if days and abs(float(days[0])) < 1e-9:
        return days, values
    return [0.0] + list(days), [float(start_level)] + list(values)


def fan_chart_figure(
    report: dict[str, Any],
    market: str,
    analogue_scope: str = "ALL",
) -> go.Figure:
    row = _path_quantile_row_as_raw_level(
        report,
        _path_quantile_row(report, market, analogue_scope),
    )
    if not row:
        fig = go.Figure()
        fig.update_layout(
            title="No scenario fan data",
            xaxis_title="Forward day",
            yaxis_title="Raw market level",
            template="plotly_white",
        )
        return fig

    display_name = str(row.get("display_name") or row.get("market") or market)
    # Implied vol cannot be negative; floor the RENDERED IV series at 0 so the
    # band/lines never dip below zero (display-only; sampling/734a untouched).
    is_iv = str(row.get("market") or market or "").upper().startswith("IV")
    def _floor_iv(series: list[float]) -> list[float]:
        return [max(float(v), 0.0) for v in series] if is_iv else series

    days = _float_series(row.get("days"))
    start_level = row.get("start_level")
    p10 = _float_series(row.get("p10"))
    p50 = _float_series(row.get("p50"))
    p90 = _float_series(row.get("p90"))
    mean = _float_series(row.get("mean"))
    days, p10 = _prepend_start_to_series(days, p10, start_level)
    _, p50 = _prepend_start_to_series(_float_series(row.get("days")), p50, start_level)
    _, p90 = _prepend_start_to_series(_float_series(row.get("days")), p90, start_level)
    _, mean = _prepend_start_to_series(
        _float_series(row.get("days")), mean, start_level
    )
    p10, p50, p90, mean = _floor_iv(p10), _floor_iv(p50), _floor_iv(p90), _floor_iv(mean)
    band_x = days + list(reversed(days))
    band_y = p90 + list(reversed(p10))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=band_x,
            y=band_y,
            fill="toself",
            fillcolor="rgba(33, 150, 243, 0.18)",
            line={"color": "rgba(33, 150, 243, 0)"},
            hoverinfo="skip",
            name="P10-P90 band",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=p50,
            mode="lines",
            line={"color": "#1565C0", "width": 3},
            name="Median (conditioned)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=mean,
            mode="lines",
            line={"color": "#00897B", "width": 2, "dash": "dot"},
            name="Mean (conditioned)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=p90,
            mode="lines",
            line={"color": "rgba(21, 101, 192, 0.45)", "width": 1},
            name="P90",
            showlegend=False,
        )
    )
    path_colors = [
        "#2E7D32",
        "#EF6C00",
        "#6A1B9A",
        "#00838F",
        "#AD1457",
        "#5D4037",
    ]
    sample_path_count = 0
    for idx, path in enumerate(_as_list(row.get("sample_paths"))):
        if not isinstance(path, dict):
            continue
        values = _float_series(path.get("values"))
        _, values = _prepend_start_to_series(
            _float_series(row.get("days")),
            values,
            start_level,
        )
        values = _floor_iv(values)
        if len(values) != len(days):
            continue
        # Collapse the individual sample paths into a single legend row: only the
        # first path shows in the legend (as a representative "Generated sample
        # paths" entry); the rest share its legend group so the legend stays
        # short.  Per-trace names are preserved for hover/inspection.
        first_path = sample_path_count == 0
        sample_path_count += 1
        fig.add_trace(
            go.Scatter(
                x=days,
                y=values,
                mode="lines",
                line={
                    "color": path_colors[idx % len(path_colors)],
                    "width": 1.6,
                },
                opacity=0.78,
                legendgroup="generated_sample_paths",
                showlegend=first_path,
                name=(
                    "Generated sample paths"
                    if first_path
                    else str(path.get("label", f"Generated path {idx + 1}"))
                ),
            )
        )
    realized = _float_series(row.get("realized_path"))
    _, realized = _prepend_start_to_series(
        _float_series(row.get("days")),
        realized,
        start_level,
    )
    realized = _floor_iv(realized)
    if len(realized) == len(days):
        fig.add_trace(
            go.Scatter(
                x=days,
                y=realized,
                mode="lines",
                line={"color": "#616161", "width": 1.3, "dash": "dot"},
                name="Actual outcome (hindsight)",
            )
        )
    # --- Start-only baseline secondary fan (dashed / translucent) ---
    # The start-only baseline is stored at generation["start_only_baseline"]["path_quantiles"]
    # by attach_start_only_baseline_report().  It represents the unconditioned
    # (narrative-free) fan from the same day-0 start, enabling a visual comparison
    # of how much the narrative conditioning shifts the distribution.
    baseline_row = _baseline_path_quantile_row_as_raw_level(report, market)
    if baseline_row:
        b_days_raw = _float_series(baseline_row.get("days"))
        b_start_level = baseline_row.get("start_level")
        b_p10 = _float_series(baseline_row.get("p10"))
        b_p50 = _float_series(baseline_row.get("p50"))
        b_p90 = _float_series(baseline_row.get("p90"))
        b_days, b_p10 = _prepend_start_to_series(b_days_raw, b_p10, b_start_level)
        _, b_p50 = _prepend_start_to_series(b_days_raw, b_p50, b_start_level)
        _, b_p90 = _prepend_start_to_series(b_days_raw, b_p90, b_start_level)
        b_p10, b_p50, b_p90 = _floor_iv(b_p10), _floor_iv(b_p50), _floor_iv(b_p90)
        if b_days and b_p10 and len(b_p10) == len(b_days):
            b_band_x = b_days + list(reversed(b_days))
            b_band_y = b_p90 + list(reversed(b_p10))
            fig.add_trace(
                go.Scatter(
                    x=b_band_x,
                    y=b_band_y,
                    fill="toself",
                    fillcolor="rgba(158, 158, 158, 0.10)",
                    line={"color": "rgba(158, 158, 158, 0)"},
                    hoverinfo="skip",
                    name="Start-only P10-P90 (baseline)",
                    showlegend=True,
                )
            )
        if b_days and b_p50 and len(b_p50) == len(b_days):
            fig.add_trace(
                go.Scatter(
                    x=b_days,
                    y=b_p50,
                    mode="lines",
                    line={"color": "#9E9E9E", "width": 2, "dash": "dash"},
                    opacity=0.65,
                    name="Start-only median (baseline)",
                )
            )
    fig.update_layout(
        title=f"{display_name} 30-day scenario fan (raw level)",
        xaxis_title="Forward day",
        yaxis_title="Raw market level",
        template="plotly_white",
        # Legend below the plot so it never collides with the title/subtitle in
        # the top band; generous top/bottom margins give the subtitle and the
        # horizontal legend room to breathe.
        margin={"l": 55, "r": 25, "t": 90, "b": 120},
        legend={"orientation": "h", "yanchor": "top", "y": -0.18, "x": 0},
    )
    # Fold the former corner annotations (analogue label + IV cell coordinates)
    # into a single centered subtitle line under the title.  Kept as an
    # annotation (annotations[0]) rather than a plotly title.subtitle so the
    # main title text stays exactly "<factor> 30-day scenario fan (raw level)".
    subtitle_parts: list[str] = []
    analogue_label = str(row.get("analogue_label", ""))
    if analogue_label and str(row.get("analogue_key", "ALL")) != "ALL":
        subtitle_parts.append(analogue_label)
    cell = _as_dict(row.get("cell"))
    if cell:
        subtitle_parts.append(f"{cell.get('maturity')} / K={cell.get('moneyness')}")
    if subtitle_parts:
        fig.add_annotation(
            text="  ·  ".join(subtitle_parts),
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.045,
            showarrow=False,
            font={"size": 12, "color": "#455A64"},
            xanchor="center",
            yanchor="bottom",
        )
    return fig


def refresh_fan_chart(
    report: dict[str, Any] | None,
    fan_market: str,
    analogue_scope: str,
) -> go.Figure:
    report_dict = _as_dict(report)
    return fan_chart_figure(
        report_dict,
        fan_market,
        _resolve_analogue_scope(report_dict, analogue_scope),
    )


def refresh_validated_spine_panels(
    report: dict[str, Any] | None,
    explicit_start_window_index: float | int | None = None,
) -> tuple[str, str, str, pd.DataFrame]:
    """Refresh the Track D zero-risk product panels from the final report.

    Returns (calendar-start label HTML, support-ESS HTML, hull-support HTML,
    terminal Day-30 table).  Each underlying helper guards on missing fields,
    so this fires safely on the error/blank yield paths too.
    """

    report_dict = _as_dict(report)
    return (
        calendar_start_label(report_dict, explicit_start_window_index),
        support_ess_html(report_dict),
        support_hull_html(report_dict),
        terminal_day30_table(report_dict),
    )


def status_markdown(report: dict[str, Any]) -> str:
    grounding = _as_dict(report.get("grounding"))
    condition = _as_dict(report.get("condition_diagnostics"))
    relevance = _as_dict(report.get("relevance"))
    hard_case = _as_dict(report.get("hard_case_gate"))
    generation = _as_dict(report.get("generation"))
    artifacts = _as_dict(report.get("artifact_paths"))
    return "\n".join(
        [
            "## Run Status",
            "",
            f"- Narrative frame: `{grounding.get('narrative_frame', 'n/a')}`",
            f"- Relevance: `{relevance.get('status', 'n/a')}` - {relevance.get('reason', '')}",
            f"- Hard-case gate: `{hard_case.get('status', 'n/a')}` - {hard_case.get('reason', '')}",
            f"- Top analogue cosine: `{_fmt_float(condition.get('top_cosine'))}`",
            f"- Top analogue gap: `{_fmt_float(condition.get('top_gap'))}`",
            f"- Condition dimension: `{condition.get('condition_dim', 'n/a')}`",
            f"- Generated shape: `{generation.get('generated_state_shape', 'not run')}`",
            f"- Markdown report: `{artifacts.get('markdown', 'n/a')}`",
            f"- JSON report: `{artifacts.get('json', 'n/a')}`",
        ]
    )


def prefix_latent_status_markdown(report: dict[str, Any]) -> str:
    query = _as_dict(report.get("cached_query"))
    gate = _as_dict(report.get("validation_gate"))
    generation = _as_dict(report.get("generation"))
    metrics = _operational_score_metrics(report)
    artifacts = _as_dict(report.get("artifact_paths"))
    lines = [
        "## Scenario Workflow Diagnostics",
        "",
        f"- Cached query: `{query.get('window_id', 'n/a')}` / `{query.get('kind', 'n/a')}`",
        f"- Condition source: `{query.get('condition_source', 'n/a')}`",
        f"- Text memory dimension: `{query.get('text_memory_dim', 'n/a')}`",
        f"- Selected start status: `{gate.get('selected_start_status', gate.get('operational_status', 'n/a'))}`",
        f"- Diagnostic baseline: `{gate.get('diagnostic_baseline_status', 'n/a')}`",
        f"- Research overall: `{gate.get('overall_status', 'n/a')}`",
        f"- Stress: `{gate.get('stress_status', 'n/a')}`",
        f"- Endpoint max error: `{_fmt_float(gate.get('endpoint_max_abs_error'), 6)}`",
        f"- Rollout temperature: `{_fmt_float(generation.get('rollout_temperature'))}`",
        f"- Rollout fan scale: `{_fmt_float(generation.get('rollout_fan_scale'))}`",
        f"- Scenario CRPS vs persistence: `{_fmt_pct(metrics.get('ensemble_crps_z_improvement_vs_persistence'))}`",
        f"- Scenario energy vs persistence: `{_fmt_pct(metrics.get('energy_score_z_improvement_vs_persistence'))}`",
        f"- Operational interpretation: `{prefix_trust_interpretation(report)}`",
        f"- Generated shape: `{generation.get('generated_state_shape', 'not run')}`",
    ]
    product_gate = _as_dict(report.get("condition_only_product_gate"))
    decision = _as_dict(product_gate.get("production_decision"))
    if decision:
        contributors: list[str] = []
        for item in _as_list(product_gate.get("decompositions")):
            if not isinstance(item, dict):
                continue
            for factor in _as_list(item.get("top_rollout_shift_factors"))[:3]:
                if isinstance(factor, dict):
                    contributors.append(
                        f"{factor.get('factor')} ({_fmt_float(factor.get('terminal_abs_shift_z'))}z)"
                    )
            if contributors:
                break
        lines.extend(
            [
                f"- Decision code: `{decision.get('decision', 'n/a')}`",
                f"- Decision note: {decision.get('ui_guidance', decision.get('reason', ''))}",
                f"- Main note contributors: `{', '.join(contributors) or 'n/a'}`",
            ]
        )
    lines.extend(
        [
            f"- Markdown report: `{artifacts.get('markdown', 'n/a')}`",
            f"- JSON report: `{artifacts.get('report', 'n/a')}`",
            f"- Run record: `{artifacts.get('run_record', 'n/a')}`",
        ]
    )
    return "\n".join(lines)


def prefix_latent_product_status_markdown(report: dict[str, Any]) -> str:
    generation = _as_dict(report.get("generation"))
    generated_shape = generation.get("generated_state_shape")
    if generated_shape:
        heading = "Scenario ready"
        next_step = "Review the fan chart and baseline-vs-narrative summary below."
    else:
        heading = "Scenario inputs ready"
        next_step = "Generate scenarios after choosing the starting market state."
    return "\n".join(["## " + heading, "", next_step])


def report_json_text(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


def _elapsed_text(start_time: float) -> str:
    return f"{time.monotonic() - float(start_time):.1f}s"


def _progress_status_markdown(
    *,
    start_time: float,
    samples: int,
    top_k: int,
    skip_generator: bool,
) -> str:
    generator_note = (
        "generator sampling skipped"
        if bool(skip_generator)
        else f"sampling {int(samples)} paths per analogue"
    )
    return "\n".join(
        [
            "## Run Status",
            "",
            f"- Run started: `{_elapsed_text(start_time)} ago`",
            "- Current step: `OpenAI grounding, embedding, analogue retrieval, and scenario generation`",
            f"- Requested historical analogues: `{int(top_k)}`",
            f"- Generator work: `{generator_note}`",
            "- Outputs will fill in automatically when the run completes.",
        ]
    )


def _prefix_progress_status_markdown(
    *,
    start_time: float,
    start_mode: str,
    samples: int,
    temperature: float = DEFAULT_PREFIX_ROLLOUT_TEMPERATURE,
    fan_scale: float = DEFAULT_PREFIX_ROLLOUT_FAN_SCALE,
    live_story: bool = False,
    condition_only_story: bool = False,
    cached_condition_report: bool = False,
    skip_rollout: bool = False,
) -> str:
    start_label = (
        "user-selected historical start"
        if str(start_mode) == "explicit_start_window"
        else (
            "user-supplied start state"
            if str(start_mode) == "user_start_state"
            else "start selection"
        )
    )
    if bool(cached_condition_report):
        condition_step = (
            f"cached story report, {start_label}, selected support regimes, "
            "support-ensemble setup"
        )
    elif bool(condition_only_story):
        condition_step = (
            "OpenAI story check, text embedding, "
            f"{start_label}, selected support regimes, support-ensemble setup"
        )
    elif bool(live_story):
        condition_step = (
            f"OpenAI story check and embedding, {start_label}, selected support "
            "regimes, support-ensemble setup"
        )
    else:
        condition_step = (
            f"cached text memory, {start_label}, selected support regimes, "
            "support-ensemble setup"
        )
    if bool(skip_rollout):
        condition_step = f"{condition_step}; scenario rollout skipped"
        generator_note = "not run during validation"
    else:
        condition_step = f"{condition_step}, 30-day scenario generation"
        generator_note = f"{int(samples)} scenario paths per support regime"
    return "\n".join(
        [
            "## Scenario Workflow Status",
            "",
            f"- Run started: `{_elapsed_text(start_time)} ago`",
            f"- Current step: `{condition_step}`",
            f"- Scenario samples: `{generator_note}`",
            "- Outputs will fill in automatically when the run completes.",
        ]
    )


def _completed_status_markdown(report: dict[str, Any], start_time: float) -> str:
    return status_markdown(report) + f"\n- Completed in: `{_elapsed_text(start_time)}`"


def _completed_prefix_status_markdown(report: dict[str, Any], start_time: float) -> str:
    return prefix_latent_product_status_markdown(report)


def _error_status_markdown(error: BaseException, start_time: float) -> str:
    return "\n".join(
        [
            "## Run Status",
            "",
            "- Status: `error`",
            f"- Failed after: `{_elapsed_text(start_time)}`",
            f"- Error type: `{type(error).__name__}`",
            f"- Message: `{str(error)}`",
        ]
    )


def _friendly_error_message(error: BaseException) -> str:
    """Map known OpenAI grounding failures to plain risk-manager guidance.

    Returns an empty string for unknown errors so the caller falls back to the
    generic technical status; in all cases the raw error type/message is kept in
    the audit report JSON.
    """

    names = {cls.__name__ for cls in type(error).__mro__}
    if "AuthenticationError" in names:
        return (
            "The narrative grounding service rejected the API credentials. "
            "Check the OpenAI API key, then generate again."
        )
    if "RateLimitError" in names:
        return (
            "The narrative grounding service is rate-limited right now. "
            "Wait a few seconds and generate again."
        )
    if "APITimeoutError" in names:
        return (
            "The narrative grounding service did not respond in time. "
            "Generate again; if it keeps timing out, shorten the narrative."
        )
    if "APIConnectionError" in names:
        return (
            "Could not reach the narrative grounding service. "
            "Check the network connection and generate again."
        )
    if names & {"APIError", "APIStatusError", "BadRequestError"}:
        return (
            "The narrative grounding service returned an error. Generate again; "
            "technical details are in Audit details > Raw JSON."
        )
    return ""


def _blank_run_outputs(
    *,
    status: str,
    fan_market: str,
) -> tuple[
    str,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    str,
    pd.DataFrame,
    go.Figure,
    str,
    dict[str, Any],
    Any,
]:
    return (
        "Run in progress. Results will appear here when complete.",
        _frame([], IMPLICATION_COLUMNS),
        _frame([], WARNING_COLUMNS),
        _frame([], ANALOGUE_COLUMNS),
        status,
        _frame([], SCENARIO_COLUMNS),
        fan_chart_figure({}, fan_market, "ALL"),
        "{}",
        {},
        analogue_scope_update({}),
    )


def _blank_prefix_outputs(
    *,
    status: str,
    fan_market: str,
) -> tuple[
    str,
    str,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    str,
    go.Figure,
    str,
    dict[str, Any],
    Any,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Any,
]:
    return (
        "Prefix-latent run in progress. Results will appear here when complete.",
        status,
        _frame([], PREFIX_SELECTED_START_COLUMNS),
        _frame([], PREFIX_VARIANT_COLUMNS),
        _frame([], VALIDATION_GATE_COLUMNS),
        scenario_summary_html({}),
        fan_chart_figure({}, fan_market, "ALL"),
        "{}",
        {},
        analogue_scope_update({}),
        _frame([], PREFIX_CONDITION_COLUMNS),
        _frame([], PREFIX_WARNING_DISPLAY_COLUMNS),
        _frame([], PREFIX_WARNING_COMPONENT_COLUMNS),
        _frame([], PREFIX_SHIFT_FACTOR_COLUMNS),
        _frame([], PREFIX_START_CANDIDATE_COLUMNS),
        _frame([], PREFIX_USER_START_COLUMNS),
        historical_start_candidate_update({}),
    )


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_condition_only_case_for_app(
    *,
    story: str,
    output_dir: str | Path,
    model: str = "gpt-5.4-mini",
    dotenv: str | Path = ".env",
    max_output_tokens: int = 1800,
    grounder: Callable[..., Any] = ground_condition_only_story_with_openai,
) -> dict[str, Any]:
    grounding, metadata = grounder(
        str(story or DEFAULT_STORY),
        model=str(model),
        dotenv_path=dotenv,
        max_output_tokens=int(max_output_tokens),
    )
    validation = validate_condition_only_grounding_result(grounding)
    query_text = condition_query_text_from_grounding(
        str(story or DEFAULT_STORY), grounding
    )
    case_dir = Path(output_dir)
    case_payload = {
        "case_name": "gradio_live_condition_only_story",
        "story": str(story or DEFAULT_STORY),
        "story_split": split_story_for_conditioning(str(story or DEFAULT_STORY)),
        "condition_only_grounding": grounding.model_dump(),
        "condition_only_validation": validation,
        "candidate_query_text": query_text,
        "metadata": {
            **_as_dict(metadata),
            "prompt_version": CONDITION_ONLY_PROMPT_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
        "artifact_paths": {
            "case_json": str(case_dir / "condition_only_grounding_case.json"),
            "query_text": str(case_dir / "condition_only_query_text.txt"),
        },
    }
    _write_json(case_dir / "condition_only_grounding_case.json", case_payload)
    (case_dir / "condition_only_query_text.txt").write_text(
        query_text.rstrip() + "\n",
        encoding="utf-8",
    )
    return case_payload


def build_condition_only_report_for_app(
    *,
    story: str,
    output_dir: str | Path,
    grounder: Callable[..., Any] = ground_condition_only_story_with_openai,
    condition_report_runner: Callable[
        [SimpleNamespace],
        dict[str, Any],
    ] = run_condition_only_report,
) -> dict[str, Any]:
    output = Path(output_dir)
    case = build_condition_only_case_for_app(
        story=story,
        output_dir=output / "condition_only_grounding",
        grounder=grounder,
    )
    report = condition_report_runner(
        SimpleNamespace(
            case_json=case["artifact_paths"]["case_json"],
            summary_json=None,
            case_name=None,
            case_index=0,
            output_dir=str(output / "condition_only_report"),
            bridge_arrays=DEFAULT_PREFIX_BRIDGE_ARRAYS,
            bridge_adapter=DEFAULT_BRIDGE_ADAPTER,
            embedding_model="text-embedding-3-small",
            dotenv=".env",
        )
    )
    report["condition_only_case"] = case
    return report


def enrich_prefix_report_with_product_gate(report: dict[str, Any]) -> dict[str, Any]:
    report_path = Path(_as_dict(report.get("artifact_paths")).get("report", ""))
    if not report_path.exists():
        return report
    decomposition = decompose_report(report_path, top_factors=8)
    product_gate = {
        "production_decision": production_decision([decomposition]),
        "decompositions": [decomposition],
    }
    enriched = {**report, "condition_only_product_gate": product_gate}
    _write_json(report_path, enriched)
    return enriched


def _joint39_spec_names(channel_count: int) -> list[str]:
    if int(channel_count) == len(JOINT39_SPEC_NAMES):
        return list(JOINT39_SPEC_NAMES)
    return [f"channel:{idx:02d}" for idx in range(int(channel_count))]


def _terminal_delta_values_for_market(
    *,
    terminal_delta: np.ndarray,
    market: str,
    spec_names: list[str],
) -> np.ndarray | None:
    market_key = str(market or "").upper()
    if market_key == "IV_SURFACE":
        if terminal_delta.shape[-1] < 25:
            return None
        return np.nanmean(terminal_delta[:, :25], axis=1)
    spec_name = SCENARIO_MARKET_TO_START_SPEC.get(market_key)
    if not spec_name:
        return None
    index = {name: idx for idx, name in enumerate(spec_names)}
    if spec_name not in index:
        return None
    return terminal_delta[:, index[spec_name]]


def _terminal_summary_with_sign_metrics(
    rows: Any,
    *,
    generated_states: np.ndarray,
    start_raw: np.ndarray,
    spec_names: list[str],
) -> list[dict[str, Any]]:
    """Attach empirical terminal sign shares to terminal summary rows."""

    states = np.asarray(generated_states, dtype=np.float32)
    start = np.asarray(start_raw, dtype=np.float32)
    if states.ndim != 3 or start.ndim != 1 or states.shape[-1] != start.shape[0]:
        return [dict(row) for row in _as_list(rows) if isinstance(row, dict)]

    terminal_delta = states[:, -1, :] - start[None, :]
    enriched: list[dict[str, Any]] = []
    for row in _as_list(rows):
        if not isinstance(row, dict):
            continue
        item = dict(row)
        values = _terminal_delta_values_for_market(
            terminal_delta=terminal_delta,
            market=str(item.get("market", "")),
            spec_names=spec_names,
        )
        if values is not None:
            valid = np.asarray(values, dtype=np.float64)
            valid = valid[np.isfinite(valid)]
            if valid.size:
                item.update(
                    {
                        "terminal_sample_count": int(valid.size),
                        "terminal_probability_up": float(np.mean(valid > 0.0)),
                        "terminal_probability_down": float(np.mean(valid < 0.0)),
                        "terminal_probability_flat": float(np.mean(valid == 0.0)),
                    }
                )
        enriched.append(item)
    return enriched


def _operational_variant_index_for_live_calibration(
    report: dict[str, Any],
    *,
    variant_count: int,
) -> int:
    cached = _as_dict(report.get("cached_query"))
    for raw in (
        cached.get("operational_memory_prior_variant_index"),
        _as_dict(report.get("selected_start_state")).get("variant_index"),
    ):
        if raw is None:
            continue
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < int(variant_count):
            return idx
    for idx, row in enumerate(_as_list(report.get("variant_rows"))):
        if isinstance(row, dict) and bool(row.get("is_operational")):
            if idx < int(variant_count):
                return int(idx)
    return 0


def _operational_path_label(report: dict[str, Any], op_idx: int) -> dict[str, str]:
    target_key = f"RANK_{int(op_idx) + 1}"
    variant_rows = _as_list(report.get("variant_rows"))
    start_id = ""
    if int(op_idx) < len(variant_rows) and isinstance(variant_rows[int(op_idx)], dict):
        start_id = str(variant_rows[int(op_idx)].get("start_window_id", ""))
    label = (
        f"Selected start: {start_id}" if start_id else f"Selected start {op_idx + 1}"
    )
    for row in _as_list(_as_dict(report.get("generation")).get("path_quantiles")):
        if not isinstance(row, dict):
            continue
        if str(row.get("analogue_key")) == target_key:
            return {
                "analogue_key": target_key,
                "analogue_label": str(row.get("analogue_label") or label),
                "window_id": str(row.get("window_id") or start_id),
            }
    return {"analogue_key": target_key, "analogue_label": label, "window_id": start_id}


def _replace_operational_path_quantiles(
    report: dict[str, Any],
    *,
    op_idx: int,
    calibrated_states: np.ndarray,
    start_raw: np.ndarray,
    spec_names: list[str],
) -> list[dict[str, Any]]:
    generation = _as_dict(report.get("generation"))
    existing_rows = _as_list(generation.get("path_quantiles"))
    label = _operational_path_label(report, op_idx)
    replacement_rows = path_quantiles_for_generated_states(
        calibrated_states[None, :, :, :],
        start_raw[None, :],
        spec_names,
        analogues=[label],
        future_states=None,
        max_paths=6,
    )
    replacement_by_market = {
        str(row.get("market")): row
        for row in replacement_rows
        if isinstance(row, dict)
        and str(row.get("analogue_key")) == label["analogue_key"]
    }
    replaced = False
    updated_rows: list[dict[str, Any]] = []
    for row in existing_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("analogue_key")) != label["analogue_key"]:
            updated_rows.append(row)
            continue
        replacement = replacement_by_market.get(str(row.get("market")))
        if replacement is None:
            updated_rows.append(row)
            continue
        merged = dict(row)
        for key in (
            "start_level",
            "start_level_p10",
            "start_level_p50",
            "start_level_p90",
            "days",
            "p10",
            "p50",
            "p90",
            "mean",
            "sample_paths",
            "value_kind",
            "display_name",
            "cell",
        ):
            if key in replacement:
                merged[key] = replacement[key]
        updated_rows.append(merged)
        replaced = True
    if not replaced:
        updated_rows.extend(replacement_by_market.values())
    return updated_rows


def _write_live_calibration_markdown(
    report: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    markdown_text = str(
        _as_dict(report.get("artifact_paths")).get("markdown", "")
    ).strip()
    if not markdown_text:
        return
    markdown_path = Path(markdown_text)
    if not markdown_path.exists() or not markdown_path.is_file():
        return
    text = markdown_path.read_text(encoding="utf-8")
    marker = "## Live Demo Narrative Calibration"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip()
    if bool(metadata.get("applied")):
        body = (
            f"{marker}\n\n"
            "Diagnostic note: this older support-gated directional delta "
            "calibration was applied after the support-grounded SNI rollout. It "
            "is retained for audit/replay only; the current product-facing "
            "demo view is the nearest-similar top3/90 posterior ensemble.\n\n"
            f"- Mode: `{metadata.get('mode', '')}`\n"
            f"- Effective beta: `{_fmt_float(metadata.get('effective_beta'))}`\n"
            f"- Support gate: `{_fmt_float(metadata.get('support_gate'))}`\n"
            f"- Active directional claims: `{int(metadata.get('active_direction_count', 0) or 0)}`\n"
        )
    else:
        body = (
            f"{marker}\n\n"
            "The live demo did not apply narrative ensemble calibration for this "
            "run.\n\n"
            f"- Reason: `{metadata.get('skip_reason', 'not_applied')}`\n"
            f"- Support gate: `{_fmt_float(metadata.get('support_gate'))}`\n"
        )
    markdown_path.write_text(
        text.rstrip() + "\n\n" + body.rstrip() + "\n", encoding="utf-8"
    )


def _write_live_top3_90_markdown(
    report: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    markdown_text = str(
        _as_dict(report.get("artifact_paths")).get("markdown", "")
    ).strip()
    if not markdown_text:
        return
    markdown_path = Path(markdown_text)
    if not markdown_path.exists() or not markdown_path.is_file():
        return
    text = markdown_path.read_text(encoding="utf-8")
    marker = "## Live Demo Top3/90 Ensemble"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip()
    if bool(metadata.get("applied")):
        rows = [
            f"- Ensemble: `{metadata.get('label', TOP3_90_ENSEMBLE_LABEL)}`",
            f"- Selected support regimes: `{metadata.get('selected_component_count', 0)}`",
            f"- Displayed scenario paths: `{metadata.get('posterior_sample_count', 0)}`",
            f"- Support weight mass before renormalization: `{_fmt_float(metadata.get('base_weight_mass'))}`",
        ]
        body = (
            f"{marker}\n\n"
            "The live demo displays the current paper candidate: nearest-similar "
            "support regimes with a main-regime top3/90 posterior. The full "
            "support set remains available in the JSON as audit evidence, but the "
            "fan chart and terminal summary use the selected main-regime ensemble.\n\n"
            + "\n".join(rows)
        )
    else:
        body = (
            f"{marker}\n\n"
            "The live demo could not build the top3/90 posterior view for this "
            "run, so the original generated report was left unchanged.\n\n"
            f"- Reason: `{metadata.get('skip_reason', 'not_applied')}`\n"
        )
    markdown_path.write_text(
        text.rstrip() + "\n\n" + body.rstrip() + "\n", encoding="utf-8"
    )


def attach_start_only_baseline_report(
    report: dict[str, Any],
    baseline_report: dict[str, Any],
    *,
    memory_prior_mode: str = "soft_topk_start_only",
) -> dict[str, Any]:
    """Attach a same-start, no-narrative baseline used for directional summaries."""

    generation = _as_dict(report.setdefault("generation", {}))
    baseline_generation = _as_dict(baseline_report.get("generation"))
    baseline_query = _as_dict(baseline_report.get("cached_query"))
    baseline_prior = _as_dict(baseline_query.get("memory_prior"))
    generation["start_only_baseline"] = {
        "memory_prior_mode": str(baseline_prior.get("mode") or memory_prior_mode),
        "terminal_delta_summary": _as_list(
            baseline_generation.get("terminal_delta_summary")
        ),
        "path_quantiles": _as_list(baseline_generation.get("path_quantiles")),
        "posterior_ensemble": _as_dict(
            baseline_generation.get("posterior_ensemble")
        ),
        "support_candidates": _as_list(baseline_prior.get("candidate_details"))[:3],
    }
    return report


def apply_live_top3_90_posterior_ensemble(report: dict[str, Any]) -> dict[str, Any]:
    """Make the live demo display the verified nearest-similar top3/90 candidate."""

    generation = dict(_as_dict(report.get("generation")))
    metadata: dict[str, Any] = {
        "mode": "nearest_similar_main_regime_top3_90",
        "label": TOP3_90_ENSEMBLE_LABEL,
        "max_components": int(TOP3_90_MAX_COMPONENTS),
        "min_cumulative_weight": float(TOP3_90_MIN_WEIGHT_MASS),
        "default_analogue_key": TOP3_90_ANALOGUE_KEY,
        "applied": False,
    }
    artifact_paths = _as_dict(report.get("artifact_paths"))
    arrays_path = Path(str(artifact_paths.get("arrays", "")))
    if not arrays_path.exists():
        metadata["skip_reason"] = "arrays_missing"
        generation["posterior_ensemble"] = metadata
        updated = {**report, "generation": generation}
        _write_live_top3_90_markdown(updated, metadata)
        return updated

    required = {
        "generated_states",
        "requested_raw",
        "rollout_component_variant_index",
        "rollout_component_window_index",
        "rollout_component_weight",
        "rollout_component_sample_count",
    }
    with np.load(str(arrays_path), allow_pickle=True) as arrays:
        missing = sorted(required.difference(arrays.files))
        if missing:
            metadata["skip_reason"] = "arrays_missing_keys"
            metadata["missing_keys"] = missing
            generation["posterior_ensemble"] = metadata
            updated = {**report, "generation": generation}
            _write_live_top3_90_markdown(updated, metadata)
            return updated
        states = np.asarray(arrays["generated_states"], dtype=np.float32)
        requested_raw = np.asarray(arrays["requested_raw"], dtype=np.float32)
        component_variant_index = np.asarray(
            arrays["rollout_component_variant_index"], dtype=np.int64
        )
        component_window_index = np.asarray(
            arrays["rollout_component_window_index"], dtype=np.int64
        )
        component_weight = np.asarray(
            arrays["rollout_component_weight"], dtype=np.float64
        )
        component_sample_count = np.asarray(
            arrays["rollout_component_sample_count"], dtype=np.int64
        )
    if states.ndim != 4 or requested_raw.ndim != 2:
        metadata["skip_reason"] = "unexpected_array_shape"
        generation["posterior_ensemble"] = metadata
        updated = {**report, "generation": generation}
        _write_live_top3_90_markdown(updated, metadata)
        return updated

    op_idx = _operational_variant_index_for_live_calibration(
        report,
        variant_count=states.shape[0],
    )
    components = component_slices_for_variant(
        variant_index=int(op_idx),
        component_variant_index=component_variant_index,
        component_window_index=component_window_index,
        component_weight=component_weight,
        component_sample_count=component_sample_count,
        sample_count=int(states.shape[1]),
    )
    component_rows = [
        {**component, "component_no": int(pos)}
        for pos, component in enumerate(components)
    ]
    selected = select_sparse_components(
        component_rows,
        max_components=TOP3_90_MAX_COMPONENTS,
        min_cumulative_weight=TOP3_90_MIN_WEIGHT_MASS,
    )
    sample_indices: list[int] = []
    support_rows: list[dict[str, Any]] = []
    candidate_by_window = {
        int(item.get("window_index", item.get("bridge_local_index"))): item
        for item in _as_list(
            _as_dict(_as_dict(report.get("cached_query")).get("memory_prior")).get(
                "candidate_details"
            )
        )
        if isinstance(item, dict)
        and item.get("window_index", item.get("bridge_local_index")) is not None
    }
    for display_rank, component in enumerate(
        sorted(selected, key=lambda item: float(item.get("sparse_weight", 0.0)), reverse=True),
        start=1,
    ):
        start_slice, stop_slice = component["sample_slice"]
        sample_indices.extend(range(int(start_slice), int(stop_slice)))
        window_index = int(component["window_index"])
        candidate = dict(candidate_by_window.get(window_index, {}))
        base_weight = float(component.get("weight", 0.0))
        posterior_weight = float(component.get("sparse_weight", 0.0))
        candidate.update(
            {
                "rank": int(display_rank),
                "window_index": int(window_index),
                "bridge_local_index": int(
                    candidate.get("bridge_local_index", window_index)
                ),
                "window_id": str(
                    candidate.get("window_id") or f"joint39_train_{window_index:04d}"
                ),
                "base_support_weight": float(base_weight),
                "weight": float(posterior_weight),
                "posterior_weight": float(posterior_weight),
                "posterior_role": "top3_90_selected",
                "component_sample_count": int(component.get("sample_count", 0)),
            }
        )
        support_rows.append(candidate)
    if not sample_indices:
        metadata["skip_reason"] = "no_selected_component_samples"
        generation["posterior_ensemble"] = metadata
        updated = {**report, "generation": generation}
        _write_live_top3_90_markdown(updated, metadata)
        return updated

    selected_states = states[int(op_idx), np.asarray(sample_indices, dtype=np.int64)]
    start_raw = requested_raw[int(op_idx)]
    spec_names = _joint39_spec_names(states.shape[-1])
    selected_summary = summarize_retrieval_generated_states(
        selected_states[None, :, :, :],
        start_raw[None, :],
        spec_names,
    )
    path_rows = path_quantiles_for_generated_states(
        selected_states[None, :, :, :],
        start_raw[None, :],
        spec_names,
        analogues=None,
        future_states=None,
        max_paths=6,
    )
    for row in path_rows:
        if not isinstance(row, dict):
            continue
        row["analogue_key"] = TOP3_90_ANALOGUE_KEY
        row["analogue_label"] = TOP3_90_ENSEMBLE_LABEL
        row["posterior_mode"] = "top3_90"
        row["value_kind"] = "raw_level"
    base_weight_mass = sum(float(item.get("base_support_weight", 0.0)) for item in support_rows)
    metadata.update(
        {
            "applied": True,
            "operational_variant_index": int(op_idx),
            "selected_component_count": int(len(support_rows)),
            "posterior_sample_count": int(selected_states.shape[0]),
            "base_weight_mass": float(base_weight_mass),
            "selected_support": support_rows,
        }
    )
    generation["terminal_delta_summary"] = _terminal_summary_with_sign_metrics(
        selected_summary.get("terminal_delta_summary", []),
        generated_states=selected_states,
        start_raw=start_raw,
        spec_names=spec_names,
    )
    generation["path_quantiles"] = path_rows
    generation["posterior_sample_count"] = int(selected_states.shape[0])
    generation["posterior_ensemble"] = metadata
    updated = {**report, "generation": generation}
    _write_live_top3_90_markdown(updated, metadata)
    return updated


def apply_live_support_gated_ensemble_calibration(
    report: dict[str, Any],
    *,
    beta: float = DEFAULT_PREFIX_ENSEMBLE_CALIBRATION_BETA,
    alpha: float = DEFAULT_PREFIX_ENSEMBLE_CALIBRATION_ALPHA,
    beta_bound: float = DEFAULT_PREFIX_ENSEMBLE_CALIBRATION_BETA_BOUND,
    support_gate_mode: str = "direction_status",
) -> dict[str, Any]:
    """Apply the current support-gated narrative calibration to a live app report.

    The calibration is intentionally bounded and support-gated. If the support
    prior is start-only or rejected by the direction check, the report is marked
    as skipped and the generated scenario distribution is left unchanged.
    """

    generation = dict(_as_dict(report.get("generation")))
    metadata: dict[str, Any] = {
        "mode": "support_gated_directional_delta_calibration",
        "beta": float(beta),
        "alpha": float(alpha),
        "beta_bound": float(beta_bound),
        "support_gate_mode": str(support_gate_mode),
        "applied": False,
    }
    artifact_paths = _as_dict(report.get("artifact_paths"))
    arrays_path = Path(str(artifact_paths.get("arrays", "")))
    if not arrays_path.exists():
        metadata["skip_reason"] = "arrays_missing"
        generation["narrative_ensemble_calibration"] = metadata
        updated = {**report, "generation": generation}
        _write_live_calibration_markdown(updated, metadata)
        return updated

    arrays = np.load(str(arrays_path))
    required = {"generated_states", "requested_raw", "delta_scale"}
    missing = sorted(required.difference(arrays.files))
    if missing:
        metadata["skip_reason"] = "arrays_missing_keys"
        metadata["missing_keys"] = missing
        generation["narrative_ensemble_calibration"] = metadata
        updated = {**report, "generation": generation}
        _write_live_calibration_markdown(updated, metadata)
        return updated

    states = np.asarray(arrays["generated_states"], dtype=np.float32)
    requested_raw = np.asarray(arrays["requested_raw"], dtype=np.float32)
    delta_scale = np.asarray(arrays["delta_scale"], dtype=np.float32)
    if states.ndim != 4 or requested_raw.ndim != 2:
        metadata["skip_reason"] = "unexpected_array_shape"
        generation["narrative_ensemble_calibration"] = metadata
        updated = {**report, "generation": generation}
        _write_live_calibration_markdown(updated, metadata)
        return updated

    op_idx = _operational_variant_index_for_live_calibration(
        report,
        variant_count=states.shape[0],
    )
    grounding = _as_dict(_as_dict(report.get("cached_query")).get("grounding"))
    direction = direction_vector_from_grounding(
        grounding,
        factor_count=states.shape[-1],
        fallback_text=str(
            _as_dict(report.get("cached_query")).get("query_text")
            or _as_dict(report.get("cached_query")).get("narrative_text")
            or ""
        ),
    )
    support_gate = float(
        _support_evidence_gate_from_report(report, mode=str(support_gate_mode))
    )
    metadata.update(
        {
            "operational_variant_index": int(op_idx),
            "support_gate": float(support_gate),
            "active_direction_count": int(np.count_nonzero(direction)),
            "effective_beta": float(beta) * float(support_gate),
        }
    )
    if support_gate <= 0.0:
        metadata["skip_reason"] = "support_gate_blocked"
        generation["narrative_ensemble_calibration"] = metadata
        updated = {**report, "generation": generation}
        _write_live_calibration_markdown(updated, metadata)
        return updated
    if int(np.count_nonzero(direction)) == 0:
        metadata["skip_reason"] = "no_directional_claims"
        generation["narrative_ensemble_calibration"] = metadata
        updated = {**report, "generation": generation}
        _write_live_calibration_markdown(updated, metadata)
        return updated

    start_raw = requested_raw[int(op_idx)]
    samples = states[int(op_idx)] - start_raw[None, None, :]
    calibrated_delta = apply_directional_delta_calibration(
        samples,
        delta_scale=delta_scale,
        direction_vector=direction,
        beta=float(beta) * support_gate,
        alpha=float(alpha),
        beta_bound=float(beta_bound),
    )
    calibrated_states = (start_raw[None, None, :] + calibrated_delta).astype(np.float32)
    spec_names = _joint39_spec_names(states.shape[-1])
    selected_summary = summarize_retrieval_generated_states(
        calibrated_states[None, :, :, :],
        start_raw[None, :],
        spec_names,
    )
    generation["terminal_delta_summary"] = _terminal_summary_with_sign_metrics(
        selected_summary.get("terminal_delta_summary", []),
        generated_states=calibrated_states,
        start_raw=start_raw,
        spec_names=spec_names,
    )
    if generation.get("path_quantiles"):
        generation["path_quantiles"] = _replace_operational_path_quantiles(
            report,
            op_idx=int(op_idx),
            calibrated_states=calibrated_states,
            start_raw=start_raw,
            spec_names=spec_names,
        )
    generation["narrative_ensemble_calibration"] = {
        **metadata,
        "applied": True,
        "summary_scope": "operational_selected_start",
    }
    updated = {**report, "generation": generation}
    _write_live_calibration_markdown(
        updated,
        _as_dict(generation.get("narrative_ensemble_calibration")),
    )
    return updated


def mark_live_app_conditioning(
    report: dict[str, Any],
    condition_report_payload: dict[str, Any],
) -> dict[str, Any]:
    """Mark reports whose condition report was freshly created by the app."""

    case = _as_dict(condition_report_payload.get("condition_only_case"))
    metadata = _as_dict(case.get("metadata"))
    live_note = (
        "Live Gradio narrative workflow. OpenAI was called in the app to ground "
        "the typed story and build the text-memory condition; the downstream "
        "prefix rollout then reused that freshly generated condition report with "
        "the explicit historical start."
    )
    enriched = {
        **report,
        "scope_note": live_note,
        "live_app_openai_conditioning": {
            "status": "fresh_condition_report",
            "grounding_model": metadata.get(
                "model", metadata.get("grounding_model", "")
            ),
            "response_id": metadata.get("response_id", ""),
            "usage": metadata.get("usage", {}),
        },
    }
    report_path = Path(_as_dict(enriched.get("artifact_paths")).get("report", ""))
    if report_path.exists():
        _write_json(report_path, enriched)
    markdown_path = Path(_as_dict(enriched.get("artifact_paths")).get("markdown", ""))
    if markdown_path.exists():
        markdown = markdown_path.read_text(encoding="utf-8")
        if live_note not in markdown:
            markdown = markdown.replace(
                "## Product Contract",
                f"## Live App Conditioning Note\n\n{live_note}\n\n## Product Contract",
                1,
            )
            markdown_path.write_text(markdown, encoding="utf-8")
    return enriched


def build_run_args(
    *,
    story: str,
    samples: int,
    top_k: int,
    skip_generator: bool,
    output_dir: str = DEFAULT_APP_OUTPUT_DIR,
) -> SimpleNamespace:
    return SimpleNamespace(
        story=str(story or DEFAULT_STORY),
        grounding_json=None,
        grounding_model="gpt-5.4-mini",
        grounding_max_output_tokens=1200,
        embedding_model="text-embedding-3-small",
        dotenv=".env",
        pipeline_report=DEFAULT_PIPELINE_REPORT,
        pipeline_npz=DEFAULT_PIPELINE_NPZ,
        bridge_adapter=DEFAULT_BRIDGE_ADAPTER,
        casebook=DEFAULT_CASEBOOK,
        hard_case_manifest=DEFAULT_HARD_CASE_MANIFEST,
        checkpoint=None,
        top_k=int(top_k),
        ood_threshold=0.75,
        samples=int(samples),
        n_steps=30,
        chunk_size=max(8, min(32, int(samples))),
        temperature=1.0,
        device="cpu",
        skip_generator=bool(skip_generator),
        output_dir=str(output_dir),
    )


def build_prefix_latent_run_args(
    *,
    start_mode: str,
    samples: int,
    memory_prior_mode: str = "cohesive_topk_narrative_start_checked",
    live_story: bool = False,
    story: str = DEFAULT_STORY,
    condition_report: str | None = None,
    explicit_start_window_index: int | None = None,
    start_state_json: str | None = None,
    skip_rollout: bool = False,
    output_dir: str = DEFAULT_PREFIX_APP_OUTPUT_DIR,
) -> SimpleNamespace:
    return SimpleNamespace(
        # Full train-region start bridge (4010 windows; day-0 Feb 2000 - Jan 2016)
        # so the demo can start from any historical date incl. 2008. Start pool == 939a support
        # bank (below); the legacy oracle bridge stays the query projector.
        bridge_report=DEFAULT_PREFIX_FULL_START_BRIDGE_REPORT,
        bridge_arrays=DEFAULT_PREFIX_FULL_START_BRIDGE_ARRAYS,
        support_bank_report=(
            DEFAULT_PREFIX_SUPPORT_BANK_REPORT
            if Path(DEFAULT_PREFIX_SUPPORT_BANK_REPORT).exists()
            else None
        ),
        support_bank_arrays=(
            DEFAULT_PREFIX_SUPPORT_BANK_ARRAYS
            if Path(DEFAULT_PREFIX_SUPPORT_BANK_ARRAYS).exists()
            else None
        ),
        pipeline_report=DEFAULT_PIPELINE_REPORT,
        checkpoint=(
            "models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/"
            "best_model.pt"
        ),
        output_dir=str(output_dir),
        query_role="anchor",
        query_kind=None,
        query_window_id=None,
        query_index=0,
        condition_report=condition_report,
        live_story=bool(live_story),
        story=str(story or DEFAULT_STORY),
        grounding_json=None,
        grounding_model="gpt-5.4-mini",
        grounding_max_output_tokens=1200,
        # Legacy-oracle bridge adapter is 1536-d (text-embedding-3-small). Do NOT use
        # text-embedding-3-large (3072-d) — it would dim-mismatch the adapter. The 14x14 bridge
        # that used 3072-d is on contamination hold and failed its fit gate (992b ceiling).
        embedding_model="text-embedding-3-small",
        bridge_adapter=DEFAULT_PREFIX_BRIDGE_ADAPTER,
        start_reliability_manifest=(
            DEFAULT_PREFIX_START_RELIABILITY_MANIFEST
            if Path(DEFAULT_PREFIX_START_RELIABILITY_MANIFEST).exists()
            else None
        ),
        dotenv=".env",
        start_mode=str(start_mode),
        explicit_start_window_index=explicit_start_window_index,
        start_state_json=start_state_json,
        start_distance_threshold_z=15.0,
        start_distance_penalty=0.02,
        implication_alignment_weight=0.25,
        memory_prior_mode=str(memory_prior_mode),
        memory_prior_top_k=8,
        memory_prior_temperature=0.2,
        memory_prior_diverse_max_pairwise_cosine=0.95,
        memory_prior_diverse_min_index_gap=30,
        memory_prior_quality_guard_candidate_pool_size=12,
        memory_prior_quality_guard_mixture_size=3,
        memory_prior_quality_guard_max_mixtures=64,
        memory_prior_quality_guard_min_candidate_mixtures=4,
        prefix_prior_mode="decoder",
        rollout_mixture_mode="component_prefix_mixture",
        include_original_baseline=True,
        hidden_dim=256,
        steps=1000,
        batch_size=64,
        eval_batch_size=16,
        lr=1e-3,
        seed=791,
        device="cuda",
        skip_rollout=bool(skip_rollout),
        samples=int(samples),
        n_steps=30,
        chunk_size=max(4, min(16, int(samples))),
        temperature=DEFAULT_PREFIX_ROLLOUT_TEMPERATURE,
        rollout_fan_scale=DEFAULT_PREFIX_ROLLOUT_FAN_SCALE,
        score_scale_floor=1e-3,
        hard_case_count=8,
        max_paths=6,
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


def run_story_for_app(
    story: str,
    samples: int,
    top_k: int,
    fan_market: str,
    analogue_scope: str,
    skip_generator: bool,
    *,
    runner: Callable[[SimpleNamespace], dict[str, Any]] = run_story_smoke,
) -> Any:
    start_time = time.monotonic()
    running_status = _progress_status_markdown(
        start_time=start_time,
        samples=samples,
        top_k=top_k,
        skip_generator=skip_generator,
    )
    yield _blank_run_outputs(status=running_status, fan_market=fan_market)

    args = build_run_args(
        story=story,
        samples=samples,
        top_k=top_k,
        skip_generator=skip_generator,
    )
    try:
        report = runner(args)
    except Exception as error:  # pragma: no cover - defensive UI path
        error_report = {
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        yield (
            "The run failed before a report could be produced.",
            _frame([], IMPLICATION_COLUMNS),
            _frame([], WARNING_COLUMNS),
            _frame([], ANALOGUE_COLUMNS),
            _error_status_markdown(error, start_time),
            _frame([], SCENARIO_COLUMNS),
            fan_chart_figure({}, fan_market, "ALL"),
            report_json_text(error_report),
            error_report,
            analogue_scope_update({}),
        )
        return

    markdown_path = Path(_as_dict(report.get("artifact_paths")).get("markdown", ""))
    markdown = (
        markdown_path.read_text(encoding="utf-8")
        if str(markdown_path) and markdown_path.exists()
        else render_story_smoke_markdown(report)
    )
    yield (
        markdown,
        implications_table(report),
        warnings_table(report),
        analogues_table(report),
        _completed_status_markdown(report, start_time),
        scenario_table(report),
        fan_chart_figure(report, fan_market, _default_analogue_scope(report)),
        report_json_text(report),
        report,
        analogue_scope_update(report),
    )


def run_prefix_latent_for_app(
    start_mode: str,
    samples: int,
    fan_market: str,
    analogue_scope: str,
    live_story: bool = False,
    story: str = DEFAULT_STORY,
    cached_condition_report: str | None = None,
    condition_only_story: bool = False,
    use_explicit_start: bool = False,
    explicit_start_window_index: float | int | None = None,
    use_user_start_state: bool = False,
    start_state_json: str | None = DEFAULT_USER_START_STATE_JSON,
    approve_start: bool = True,
    skip_rollout: bool = False,
    include_start_only_baseline: bool = False,
    *,
    runner: Callable[[SimpleNamespace], dict[str, Any]] = run_prefix_latent_story_smoke,
    condition_grounder: Callable[..., Any] = ground_condition_only_story_with_openai,
    condition_report_runner: Callable[
        [SimpleNamespace],
        dict[str, Any],
    ] = run_condition_only_report,
) -> Any:
    start_time = time.monotonic()
    effective_start_mode = (
        "user_start_state"
        if bool(use_user_start_state)
        else "explicit_start_window" if bool(use_explicit_start) else str(start_mode)
    )
    explicit_start = (
        int(explicit_start_window_index)
        if bool(use_explicit_start)
        and not bool(use_user_start_state)
        and explicit_start_window_index is not None
        else None
    )
    user_start_path = str(start_state_json or "").strip() or None
    cached_report_path = str(cached_condition_report or "").strip()
    running_status = _prefix_progress_status_markdown(
        start_time=start_time,
        start_mode=effective_start_mode,
        samples=int(samples),
        temperature=DEFAULT_PREFIX_ROLLOUT_TEMPERATURE,
        fan_scale=DEFAULT_PREFIX_ROLLOUT_FAN_SCALE,
        live_story=bool(live_story),
        condition_only_story=bool(condition_only_story),
        cached_condition_report=bool(cached_report_path),
        skip_rollout=bool(skip_rollout),
    )
    yield _blank_prefix_outputs(status=running_status, fan_market=fan_market)

    try:
        condition_report_payload: dict[str, Any] | None = None
        condition_report_path: str | None = None
        output_dir = DEFAULT_PREFIX_APP_OUTPUT_DIR
        if cached_report_path:
            condition_report_path = cached_report_path
            if not Path(condition_report_path).exists():
                raise FileNotFoundError(
                    f"cached condition report not found: {condition_report_path}"
                )
            output_dir = str(
                Path(DEFAULT_PREFIX_APP_OUTPUT_DIR)
                / "cached_casebook_run"
                / _slug(Path(condition_report_path).parent.name)
            )
        elif bool(condition_only_story):
            condition_report_payload = build_condition_only_report_for_app(
                story=str(story or DEFAULT_STORY),
                output_dir=Path(DEFAULT_PREFIX_APP_OUTPUT_DIR) / "condition_only_live",
                grounder=condition_grounder,
                condition_report_runner=condition_report_runner,
            )
            condition_report_path = str(
                _as_dict(condition_report_payload.get("artifact_paths")).get("report")
            )
            output_dir = str(Path(DEFAULT_PREFIX_APP_OUTPUT_DIR) / "condition_only_run")
        args = build_prefix_latent_run_args(
            start_mode=effective_start_mode,
            samples=int(samples),
            live_story=(
                bool(live_story)
                and not bool(condition_only_story)
                and not bool(cached_report_path)
            ),
            story=str(story or DEFAULT_STORY),
            condition_report=condition_report_path,
            explicit_start_window_index=explicit_start,
            start_state_json=user_start_path if bool(use_user_start_state) else None,
            skip_rollout=bool(skip_rollout),
            output_dir=output_dir,
        )
        report = runner(args)
        if condition_report_payload is not None:
            report["condition_only_case"] = condition_report_payload.get(
                "condition_only_case"
            )
            report = mark_live_app_conditioning(report, condition_report_payload)
        report = enrich_prefix_report_with_product_gate(report)
        if not bool(skip_rollout):
            report = apply_live_top3_90_posterior_ensemble(report)
        if bool(include_start_only_baseline) and not bool(skip_rollout):
            baseline_args = build_prefix_latent_run_args(
                start_mode=effective_start_mode,
                samples=int(samples),
                memory_prior_mode="soft_topk_start_only",
                live_story=False,
                story=str(story or DEFAULT_STORY),
                condition_report=condition_report_path,
                explicit_start_window_index=explicit_start,
                start_state_json=user_start_path if bool(use_user_start_state) else None,
                skip_rollout=False,
                output_dir=str(Path(output_dir) / "start_only_baseline"),
            )
            baseline_report = runner(baseline_args)
            baseline_report = enrich_prefix_report_with_product_gate(baseline_report)
            baseline_report = apply_live_top3_90_posterior_ensemble(baseline_report)
            report = attach_start_only_baseline_report(
                report,
                baseline_report,
                memory_prior_mode="soft_topk_start_only",
            )
        if runner is run_prefix_latent_story_smoke:
            run_record_path = (
                Path(output_dir) / "run_record" / "prefix_latent_run_record.json"
            )
            _as_dict(report.setdefault("artifact_paths", {}))["run_record"] = str(
                run_record_path
            )
            report_path_text = str(
                _as_dict(report.get("artifact_paths")).get("report", "")
            )
            if report_path_text:
                _write_json(Path(report_path_text), report)
            run_record = write_prefix_run_record(
                report,
                output_dir=Path(output_dir) / "run_record",
            )
            report["run_record_summary"] = {
                "record_id": run_record.get("record_id", ""),
                "status": run_record.get("status", ""),
                "artifact_paths": run_record.get("artifact_paths", {}),
            }
    except Exception as error:  # pragma: no cover - defensive UI path
        error_report = {
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        friendly = _friendly_error_message(error)
        if friendly:
            headline = f"## Scenario generation could not complete\n\n{friendly}"
            status = "\n".join(
                [
                    "## Scenario Workflow Status",
                    "",
                    "- Status: `could not complete`",
                    f"- {friendly}",
                    "- Technical details are in Audit details > Raw JSON.",
                ]
            )
        else:
            headline = (
                "The scenario run failed before a report could be produced. "
                "Technical details are in Audit details > Raw JSON."
            )
            status = _error_status_markdown(error, start_time)
        yield (
            headline,
            status,
            _frame([], PREFIX_SELECTED_START_COLUMNS),
            _frame([], PREFIX_VARIANT_COLUMNS),
            _frame([], VALIDATION_GATE_COLUMNS),
            scenario_summary_html({}),
            fan_chart_figure({}, fan_market, "ALL"),
            report_json_text(error_report),
            error_report,
            analogue_scope_update({}),
            _frame([], PREFIX_CONDITION_COLUMNS),
            _frame([], PREFIX_WARNING_DISPLAY_COLUMNS),
            _frame([], PREFIX_WARNING_COMPONENT_COLUMNS),
            _frame([], PREFIX_SHIFT_FACTOR_COLUMNS),
            _frame([], PREFIX_START_CANDIDATE_COLUMNS),
            _frame([], PREFIX_USER_START_COLUMNS),
            historical_start_candidate_update({}),
        )
        return

    markdown_path = Path(_as_dict(report.get("artifact_paths")).get("markdown", ""))
    markdown = (
        markdown_path.read_text(encoding="utf-8")
        if str(markdown_path) and markdown_path.exists()
        else prefix_latent_status_markdown(report)
    )
    yield (
        markdown,
        _completed_prefix_status_markdown(report, start_time),
        prefix_selected_start_table(report),
        prefix_diagnostic_start_table(report),
        prefix_validation_table(report),
        scenario_summary_html(report),
        fan_chart_figure(report, fan_market, _default_analogue_scope(report)),
        report_json_text(report),
        report,
        analogue_scope_update(report),
        prefix_condition_implications_table(report),
        prefix_condition_warnings_display_table(report),
        prefix_warning_component_table(report),
        prefix_shift_factor_table(report),
        prefix_start_candidates_table(report),
        prefix_user_start_table(report),
        historical_start_candidate_update(report),
    )


def preview_prefix_start_for_app(
    start_mode: str,
    samples: int,
    fan_market: str,
    analogue_scope: str,
    live_story: bool = False,
    story: str = DEFAULT_STORY,
    cached_condition_report: str | None = None,
    condition_only_story: bool = False,
    use_explicit_start: bool = False,
    explicit_start_window_index: float | int | None = None,
    use_user_start_state: bool = False,
    start_state_json: str | None = DEFAULT_USER_START_STATE_JSON,
) -> Any:
    """Preview start/support diagnostics without running the final rollout."""

    yield from run_prefix_latent_for_app(
        start_mode=start_mode,
        samples=samples,
        fan_market=fan_market,
        analogue_scope=analogue_scope,
        live_story=live_story,
        story=story,
        cached_condition_report=cached_condition_report,
        condition_only_story=condition_only_story,
        use_explicit_start=use_explicit_start,
        explicit_start_window_index=explicit_start_window_index,
        use_user_start_state=use_user_start_state,
        start_state_json=start_state_json,
        approve_start=False,
        skip_rollout=True,
    )


def _manual_start_index(value: float | int | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _manual_start_required_outputs(*, fan_market: str) -> tuple[
    str,
    str,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    str,
    go.Figure,
    str,
    dict[str, Any],
    Any,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Any,
]:
    status = "\n".join(
        [
            "## Scenario Workflow Status",
            "",
            "- Historical start required.",
            "- Enter a bridge-local historical start window index before preview or generation.",
        ]
    )
    outputs = list(_blank_prefix_outputs(status=status, fan_market=fan_market))
    outputs[0] = (
        "## Historical Start Required\n\n"
        "This production workflow does not infer a starting level. Enter the "
        "historical start window index supplied by the risk manager."
    )
    return tuple(outputs)


def _prefix_guard_outputs(
    *,
    fan_market: str,
    headline: str,
    status_lines: list[str],
) -> tuple[Any, ...]:
    """Blank prefix-latent outputs carrying a plain guard message (no model run)."""

    outputs = list(
        _blank_prefix_outputs(
            status="\n".join(status_lines),
            fan_market=fan_market,
        )
    )
    outputs[0] = headline
    return tuple(outputs)


def _blank_narrative_guard_outputs(*, fan_market: str) -> tuple[Any, ...]:
    """Short-circuit panel for an empty narrative (never substitutes a default)."""

    return _prefix_guard_outputs(
        fan_market=fan_market,
        headline=(
            "## Enter a market narrative\n\n"
            "Describe the current/recent market state in the narrative box above, "
            "then generate scenarios."
        ),
        status_lines=[
            "## Scenario Workflow Status",
            "",
            "- Waiting for a market narrative.",
            "- Enter a narrative describing current/recent market conditions, then generate.",
        ],
    )


def _out_of_range_start_guard_outputs(
    *,
    fan_market: str,
    start_index: int,
    lo: int,
    hi: int,
) -> tuple[Any, ...]:
    """Short-circuit panel for an out-of-range day-0 index (before the paid call)."""

    return _prefix_guard_outputs(
        fan_market=fan_market,
        headline=(
            "## Starting market state out of range\n\n"
            f"The starting market state index {start_index} is outside the available "
            f"history ({lo}–{hi}). Pick a window in this range, then generate."
        ),
        status_lines=[
            "## Scenario Workflow Status",
            "",
            f"- Starting market state index {start_index} is out of range (valid {lo}–{hi}).",
            "- No grounding call was made. Adjust the starting market state and generate again.",
        ],
    )


def preview_live_openai_start_for_app(
    samples: int,
    fan_market: str,
    analogue_scope: str,
    story: str = DEFAULT_STORY,
    explicit_start_window_index: float | int | None = None,
    *,
    runner: Callable[[SimpleNamespace], dict[str, Any]] = run_prefix_latent_story_smoke,
    condition_grounder: Callable[..., Any] = ground_condition_only_story_with_openai,
    condition_report_runner: Callable[
        [SimpleNamespace],
        dict[str, Any],
    ] = run_condition_only_report,
) -> Any:
    """Preview a production-style live OpenAI narrative condition and start."""

    if not str(story or "").strip():
        yield _blank_narrative_guard_outputs(fan_market=fan_market)
        return
    explicit_start = _manual_start_index(explicit_start_window_index)
    if explicit_start is None:
        yield _manual_start_required_outputs(fan_market=fan_market)
        return
    lo, hi = _start_index_bounds()
    if hi > lo and (explicit_start < lo or explicit_start > hi):
        yield _out_of_range_start_guard_outputs(
            fan_market=fan_market, start_index=explicit_start, lo=lo, hi=hi
        )
        return
    yield from run_prefix_latent_for_app(
        start_mode="explicit_start_window",
        samples=samples,
        fan_market=fan_market,
        analogue_scope=analogue_scope,
        live_story=True,
        story=story,
        cached_condition_report="",
        condition_only_story=True,
        use_explicit_start=True,
        explicit_start_window_index=explicit_start,
        use_user_start_state=False,
        start_state_json=None,
        approve_start=False,
        skip_rollout=True,
        runner=runner,
        condition_grounder=condition_grounder,
        condition_report_runner=condition_report_runner,
    )


def run_live_openai_prefix_for_app(
    samples: int,
    fan_market: str,
    analogue_scope: str,
    story: str = DEFAULT_STORY,
    explicit_start_window_index: float | int | None = None,
    *,
    runner: Callable[[SimpleNamespace], dict[str, Any]] = run_prefix_latent_story_smoke,
    condition_grounder: Callable[..., Any] = ground_condition_only_story_with_openai,
    condition_report_runner: Callable[
        [SimpleNamespace],
        dict[str, Any],
    ] = run_condition_only_report,
) -> Any:
    """Run production-style live OpenAI narrative conditioning and rollout."""

    if not str(story or "").strip():
        yield _blank_narrative_guard_outputs(fan_market=fan_market)
        return
    explicit_start = _manual_start_index(explicit_start_window_index)
    if explicit_start is None:
        yield _manual_start_required_outputs(fan_market=fan_market)
        return
    lo, hi = _start_index_bounds()
    if hi > lo and (explicit_start < lo or explicit_start > hi):
        yield _out_of_range_start_guard_outputs(
            fan_market=fan_market, start_index=explicit_start, lo=lo, hi=hi
        )
        return
    yield from run_prefix_latent_for_app(
        start_mode="explicit_start_window",
        samples=samples,
        fan_market=fan_market,
        analogue_scope=analogue_scope,
        live_story=True,
        story=story,
        cached_condition_report="",
        condition_only_story=True,
        use_explicit_start=True,
        explicit_start_window_index=explicit_start,
        use_user_start_state=False,
        start_state_json=None,
        approve_start=True,
        skip_rollout=False,
        include_start_only_baseline=True,
        runner=runner,
        condition_grounder=condition_grounder,
        condition_report_runner=condition_report_runner,
    )


RunStoryForAppOutput = tuple[
    str,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    str,
    pd.DataFrame,
    go.Figure,
    str,
    dict[str, Any],
    Any,
]


def build_demo() -> Any:
    import gradio as gr

    with gr.Blocks(title="Narrative Conditioned Scenario Demo") as demo:
        prefix_report_state = gr.State({})
        prefix_samples = gr.State(48)
        gr.Markdown(
            "# Narrative-Conditioned Scenario Generator\n"
            "Describe the current market story and choose the day-0 market state; "
            "the demo builds a 30-day scenario distribution from the closest real "
            "historical episodes."
            + _HOW_IT_WORKS_NOTE,
            elem_classes=["demo-hero", "demo-shell"],
        )
        with gr.Accordion("About these scenarios", open=False):
            gr.Markdown(
                "These scenarios are built from the historical market episodes "
                "most similar to your narrative and starting state — a grounded "
                "what-if distribution over the next 30 days, not a point forecast. "
                "The directional read reflects those retrieved analogues and the "
                "day-0 state, not the wording of the narrative.",
                elem_classes=["demo-shell"],
            )
        story = gr.Textbox(
            label="Risk-manager narrative",
            value=APP_DEFAULT_STORY,
            lines=6,
            max_lines=10,
            placeholder="Describe the current/recent market state in risk-manager language.",
            elem_classes=["demo-shell"],
        )
        recommended_narrative = gr.Dropdown(
            choices=recommended_narrative_choices(),
            value="",
            label="Recommended narrative examples",
            info=(
                "Examples describe current/recent market conditions. "
                "Future-risk phrases are allowed, but are shown as warnings and "
                "excluded from conditioning."
            ),
            elem_classes=["demo-shell"],
        )
        with gr.Accordion("How to read this screen", open=False):
            gr.Markdown(
                "- A historical start is the day-0 market level. In production, "
                "the risk manager supplies this level from today's market or a "
                "chosen historical window.\n"
                "- The narrative selects historical analogues with similar "
                "current/recent market behavior. Future-looking language is shown "
                "as a warning, not treated as an input target.\n"
                "- The demo keeps the closest one-to-three historical regimes that "
                "together cover about 90% of the match weight, then rebalances their "
                "weights for the scenario fan.\n"
                "- Result notes are product guidance, not forecasts. The fan chart "
                "is the model's 30-day conditional distribution from the selected start."
            )
        start_date_choices = full_start_date_choices()
        start_default_index = _default_full_start_index()
        with gr.Row(equal_height=False, elem_classes=["demo-responsive-row"]):
            with gr.Column(scale=1, min_width=280):
                prefix_explicit_start_index = gr.Dropdown(
                    choices=start_date_choices,
                    value=start_default_index,
                    label="Starting market state (day-0)",
                    filterable=True,
                    info=(
                        "Pick the historical date the scenario starts from — type "
                        "a year to search. Spans Feb 2000 – Jan 2016, including the "
                        "2008 crisis. In production this would be today's market state."
                    ),
                )
            with gr.Column(scale=1, min_width=280):
                prefix_fan_market = gr.Dropdown(
                    choices=FAN_MARKET_CHOICES,
                    value="SPX",
                    label="Scenario factor",
                )
            with gr.Column(scale=1, min_width=280):
                prefix_run_button = gr.Button(
                    "Generate 30-Day Scenarios",
                    variant="primary",
                )
        prefix_start_date_hint = gr.HTML(
            value=start_index_date_hint(start_default_index),
            elem_classes=["demo-shell"],
        )
        prefix_status = gr.Markdown(
            "## Scenario Workflow Status\n\n"
            "- Sample narrative and an Oct-2008 start are loaded — click "
            "**Generate 30-Day Scenarios** (or edit the narrative/date first).",
            label="Prefix-latent status",
        )
        prefix_analogue_scope = gr.Dropdown(
            choices=[("All", "ALL")],
            value="ALL",
            visible=False,
            show_label=False,
        )
        prefix_explicit_start_candidate = gr.Dropdown(
            choices=[],
            value=None,
            visible=False,
            show_label=False,
        )
        # --- Lead with the scenario distribution (the product) -------------
        gr.Markdown("## Scenario Distribution")
        prefix_fan_plot = gr.Plot(
            value=fan_chart_figure({}, "SPX"),  # labeled empty state until first run
            label="30-day scenario fan chart",
        )
        # Thin badge chip-row directly under the chart (day-0 date, effective
        # analogues, historical-support honesty badge).
        with gr.Row(elem_classes=["demo-responsive-row"]):
            prefix_calendar_start = gr.HTML(value=calendar_start_label({}, None))
            prefix_support_ess = gr.HTML(value=support_ess_html({}))
            prefix_support_hull = gr.HTML(value=support_hull_html({}))
        gr.Markdown(
            "The colored fan is the narrative-conditioned scenario; the grey dashed "
            "median + translucent band is the start-only baseline (same start, no "
            "narrative). The gap between them reflects the historical analogues the "
            "narrative retrieved versus the start-only analogues. The selected "
            "starting level is the day-0 market state used before the historical "
            "analogues and rollout are built."
        )
        gr.Markdown(
            "### Terminal Day-30 levels (per factor)\n"
            "Day-30 P10 / P50 / P90 raw levels for the start-only baseline versus "
            "the narrative-conditioned scenario. These are the terminal slices of "
            "the fan above — the gap between baseline and conditioned columns "
            "reflects the analogues the narrative retrieved at the 30-day horizon."
        )
        prefix_terminal_day30 = gr.Dataframe(
            headers=TERMINAL_DAY30_COLUMNS,
            label="Terminal Day-30 quantiles",
            interactive=False,
            elem_classes=[DEMO_TABLE_CLASS],
        )
        prefix_selected_start = gr.Dataframe(
            headers=PREFIX_SELECTED_START_COLUMNS,
            label="Selected starting level",
            interactive=False,
            elem_classes=[DEMO_TABLE_CLASS],
        )
        gr.Markdown(
            "_The day-0 window the scenario starts from: its date and its "
            "narrative match (cosine) — how closely that historical market state "
            "matches your narrative._"
        )
        gr.Markdown(
            "This table compares the start-only baseline with the "
            "narrative-conditioned scenario. Views are the headline direction "
            "versus the starting level; path shares show what fraction of "
            "terminal paths point that way; mean moves show the average terminal "
            "change in market units and standardized size. "
            "_The change-vs-baseline tilt is baseline-relative directional context "
            "from the narrative-selected historical analogues — useful for monitoring, "
            "not a calibrated point forecast._"
        )
        gr.Markdown(
            "_How direction is set:_ your narrative selects which historical "
            "episodes ground the scenario; the distribution — including its "
            "direction — emerges from those analogues and the day-0 market state. "
            "It reflects what has historically followed conditions like these, so it "
            "can run opposite to the moves your narrative describes — testing your "
            "read against the historical record rather than echoing it."
        )
        prefix_scenario = gr.HTML(
            value=scenario_summary_html({}),
            elem_classes=["demo-scenario-summary"],
        )
        # --- Provenance & grounding (secondary; collapsed by default) ------
        with gr.Accordion("Provenance & grounding", open=False):
            gr.Markdown("### Story grounding")
            gr.Markdown(
                "The model first extracts current/recent market claims from the story. "
                "Forward-looking phrases are shown as warnings and excluded from conditioning."
            )
            with gr.Row(equal_height=False, elem_classes=["demo-responsive-row"]):
                with gr.Column(scale=2, min_width=320):
                    prefix_condition_implications = gr.Dataframe(
                        headers=PREFIX_CONDITION_COLUMNS,
                        label="Grounded current/recent market claims",
                        interactive=False,
                        elem_classes=[DEMO_TABLE_CLASS],
                    )
                with gr.Column(scale=1, min_width=280):
                    prefix_condition_warnings = gr.Dataframe(
                        headers=PREFIX_WARNING_DISPLAY_COLUMNS,
                        label="Warnings",
                        interactive=False,
                        elem_classes=[DEMO_TABLE_CLASS],
                    )
            gr.Markdown("### Historical analogues")
            gr.Markdown(
                "Both analogue sets: the narrative-conditioned episodes used for the "
                "displayed scenario fan, and the start-only baseline episodes chosen "
                "from the same starting market level without the narrative. "
                "_Narrative match (cosine)_ is how closely each analogue matches the "
                "narrative; _distance from start (σ)_ is how far its day-0 state sits "
                "from your starting market state."
            )
            prefix_start_candidates = gr.Dataframe(
                headers=PREFIX_START_CANDIDATE_COLUMNS,
                label="Selected historical analogues",
                interactive=False,
                elem_classes=[DEMO_TABLE_CLASS],
            )
        with gr.Accordion("Audit details", open=False):
            gr.Markdown(
                "Technical run details are kept here for traceability. The main "
                "demo should be read from the story claims, warnings, selected "
                "historical analogues, fan chart, and terminal level summary above."
            )
            prefix_validation = gr.Dataframe(
                headers=VALIDATION_GATE_COLUMNS,
                label="Technical gate data",
                interactive=False,
                visible=False,
                elem_classes=[DEMO_TABLE_CLASS],
            )
            prefix_diagnostic_start = gr.Dataframe(
                headers=PREFIX_VARIANT_COLUMNS,
                label="Baseline comparison",
                interactive=False,
                visible=False,
                elem_classes=[DEMO_TABLE_CLASS],
            )
            prefix_user_start = gr.Dataframe(
                headers=PREFIX_USER_START_COLUMNS,
                label="Start-state metadata",
                interactive=False,
                visible=False,
                elem_classes=[DEMO_TABLE_CLASS],
            )
            prefix_warning_components = gr.Dataframe(
                headers=PREFIX_WARNING_COMPONENT_COLUMNS,
                label="Narrative check metadata",
                interactive=False,
                visible=False,
                elem_classes=[DEMO_TABLE_CLASS],
            )
            prefix_shift_factors = gr.Dataframe(
                headers=PREFIX_SHIFT_FACTOR_COLUMNS,
                label="Sensitivity metadata",
                interactive=False,
                visible=False,
                elem_classes=[DEMO_TABLE_CLASS],
            )
            with gr.Accordion("Markdown report", open=False):
                prefix_report_markdown = gr.Markdown(label="Report")
            with gr.Accordion("Raw JSON", open=False):
                prefix_report_json = gr.Code(language="json", label="JSON")
        prefix_run_event = prefix_run_button.click(
            fn=run_live_openai_prefix_for_app,
            inputs=[
                prefix_samples,
                prefix_fan_market,
                prefix_analogue_scope,
                story,
                prefix_explicit_start_index,
            ],
            outputs=[
                prefix_report_markdown,
                prefix_status,
                prefix_selected_start,
                prefix_diagnostic_start,
                prefix_validation,
                prefix_scenario,
                prefix_fan_plot,
                prefix_report_json,
                prefix_report_state,
                prefix_analogue_scope,
                prefix_condition_implications,
                prefix_condition_warnings,
                prefix_warning_components,
                prefix_shift_factors,
                prefix_start_candidates,
                prefix_user_start,
                prefix_explicit_start_candidate,
            ],
            show_progress="full",
            show_progress_on=prefix_status,
            api_name="run_live_openai_prefix_for_app",
        )
        # Track D zero-risk product panels: refreshed from the final report once
        # the generator above has exhausted (prefix_report_state is populated).
        prefix_run_event.then(
            fn=refresh_validated_spine_panels,
            inputs=[prefix_report_state, prefix_explicit_start_index],
            outputs=[
                prefix_calendar_start,
                prefix_support_ess,
                prefix_support_hull,
                prefix_terminal_day30,
            ],
            show_progress="hidden",
            api_name="refresh_validated_spine_panels",
        )
        recommended_narrative.change(
            fn=recommended_narrative_text,
            inputs=[recommended_narrative, story],
            outputs=story,
            show_progress="hidden",
            api_name="recommended_narrative_text",
        )
        prefix_fan_market.change(
            fn=refresh_fan_chart,
            inputs=[prefix_report_state, prefix_fan_market, prefix_analogue_scope],
            outputs=prefix_fan_plot,
            show_progress="hidden",
            api_name="refresh_fan_chart",
        )
        prefix_explicit_start_index.change(
            fn=start_index_date_hint,
            inputs=[prefix_explicit_start_index],
            outputs=prefix_start_date_hint,
            show_progress="hidden",
            api_name="start_index_date_hint",
        )
    return demo


def main() -> None:
    import gradio as gr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--auth-user-env", default=DEFAULT_AUTH_USER_ENV)
    parser.add_argument("--auth-password-env", default=DEFAULT_AUTH_PASSWORD_ENV)
    parser.add_argument(
        "--require-auth",
        action="store_true",
        help="fail launch unless auth user/password env vars are present",
    )
    args = parser.parse_args()
    auth = resolve_launch_auth(
        user_env=str(args.auth_user_env),
        password_env=str(args.auth_password_env),
        require_auth=bool(args.require_auth),
    )
    demo = build_demo()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=int(args.server_port),
        share=bool(args.share),
        auth=auth,
        theme=gr.themes.Origin(),
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
