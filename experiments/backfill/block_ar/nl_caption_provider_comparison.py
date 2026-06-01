#!/usr/bin/env python
"""Compare API and Codex scenario-to-text captions.

This report is intentionally deterministic. It checks whether an optional
Codex/ChatGPT caption lane preserves the same structured data contract as the
OpenAI API caption lane, and whether it appears materially richer in the
risk-manager fields that matter for downstream conditioning.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_risk_manager_caption_v2 import (  # noqa: E402
    DEFAULT_PIPELINE_REPORT,
    RiskManagerCaptionV2,
    load_pipeline_report,
    validate_caption_v2,
)

DEFAULT_API_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_caption_v2_testflight_916a_openai/"
    "risk_manager_caption_v2_testflight_report.json"
)
DEFAULT_CODEX_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_caption_v2_codex_probe_916c"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_caption_v2_provider_comparison_916d"
)

REQUIRED_SECTIONS = (
    "scenario_title",
    "mechanical_summary",
    "current_market_state",
    "trigger",
    "transmission",
    "cross_asset_reaction",
    "sequence",
    "portfolio_vulnerability",
    "risk_manager_implication",
    "training_caption",
)

PROFESSIONAL_TERMS = (
    "risk appetite",
    "duration",
    "credit",
    "spread",
    "volatility",
    "carry",
    "haven",
    "liquidity",
    "portfolio",
    "exposure",
    "hedge",
    "dollar",
    "rates",
    "growth",
)


@dataclass(frozen=True)
class CaptionScore:
    provider: str
    window_id: str
    title: str
    archetype: str
    validation_error_count: int
    validation_warning_count: int
    section_completeness: float
    training_words: int
    evidence_count: int
    ambiguity_count: int
    leakage_exclusion_count: int
    contrastive_count: int
    professional_term_count: int
    source_market_coverage: float
    total_score: float


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _caption_by_window(report_path: Path) -> dict[str, RiskManagerCaptionV2]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    captions: dict[str, RiskManagerCaptionV2] = {}
    for row in report.get("captions", []):
        cap = RiskManagerCaptionV2.model_validate(row)
        captions[cap.window_id] = cap
    return captions


def _codex_caption_paths(codex_dir: Path) -> list[Path]:
    candidates = []
    for path in sorted(codex_dir.glob("codex_gpt55_caption*.json")):
        if "failed_validator" in path.name:
            continue
        if path.name.endswith("_pass.json") or "joint39_val_" in path.name:
            candidates.append(path)
    return candidates


def _load_codex_captions(codex_dir: Path) -> dict[str, RiskManagerCaptionV2]:
    captions: dict[str, RiskManagerCaptionV2] = {}
    for path in _codex_caption_paths(codex_dir):
        cap = RiskManagerCaptionV2.model_validate_json(path.read_text(encoding="utf-8"))
        captions[cap.window_id] = cap
    return captions


def _source_markets(pipeline_report: dict[str, Any], window_id: str) -> set[str]:
    bundles = pipeline_report.get("narrative_bundles", [])
    for bundle in bundles:
        if isinstance(bundle, dict) and bundle.get("window_id") == window_id:
            markets = set()
            for item in bundle.get("market_implications", []):
                if isinstance(item, dict) and item.get("market"):
                    markets.add(str(item["market"]).lower())
            return markets
    return set()


def _market_mentions(cap: RiskManagerCaptionV2) -> set[str]:
    text = " ".join(
        [
            cap.mechanical_summary,
            cap.current_market_state,
            cap.cross_asset_reaction,
            cap.training_caption,
            " ".join(cap.evidence_used),
        ]
    ).lower()
    aliases = {
        "spx": ("spx", "equities", "equity"),
        "vix": ("vix", "volatility"),
        "us2y": ("us2y", "2y", "2-year"),
        "us10y": ("us10y", "10y", "10-year"),
        "bbb_oas": ("bbb", "credit"),
        "aaa_oas": ("aaa", "credit"),
        "usdjpy": ("usdjpy", "yen"),
        "dxy": ("dxy", "dollar"),
        "gold": ("gold",),
        "crude_oil": ("crude", "oil"),
        "iv_surface": ("iv surface", "volatility surface"),
        "iv_skew": ("iv skew", "skew"),
    }
    found = set()
    for market, terms in aliases.items():
        if any(term in text for term in terms):
            found.add(market)
    return found


def score_caption(
    provider: str,
    cap: RiskManagerCaptionV2,
    source_markets: set[str],
) -> CaptionScore:
    issues = validate_caption_v2(cap)
    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity != "error"]
    complete = sum(1 for field in REQUIRED_SECTIONS if str(getattr(cap, field)).strip())
    section_completeness = complete / len(REQUIRED_SECTIONS)
    full_text = " ".join(
        [
            cap.mechanical_summary,
            cap.current_market_state,
            cap.trigger,
            cap.transmission,
            cap.cross_asset_reaction,
            cap.sequence,
            cap.portfolio_vulnerability,
            cap.risk_manager_implication,
            cap.training_caption,
        ]
    ).lower()
    professional_term_count = sum(1 for term in PROFESSIONAL_TERMS if term in full_text)
    mentioned = _market_mentions(cap)
    source_market_coverage = (
        len(source_markets & mentioned) / len(source_markets) if source_markets else 0.0
    )
    total_score = (
        20.0 * section_completeness
        + min(len(_words(cap.training_caption)), 160) / 8.0
        + min(len(cap.evidence_used), 10) * 1.0
        + min(len(cap.ambiguity_flags), 4) * 1.5
        + min(len(cap.contrastive_captions), 4) * 2.0
        + min(professional_term_count, 10) * 1.2
        + source_market_coverage * 12.0
        - len(errors) * 25.0
        - len(warnings) * 5.0
    )
    return CaptionScore(
        provider=provider,
        window_id=cap.window_id,
        title=cap.scenario_title,
        archetype=cap.archetype,
        validation_error_count=len(errors),
        validation_warning_count=len(warnings),
        section_completeness=section_completeness,
        training_words=len(cap.training_caption.split()),
        evidence_count=len(cap.evidence_used),
        ambiguity_count=len(cap.ambiguity_flags),
        leakage_exclusion_count=len(cap.leakage_exclusions),
        contrastive_count=len(cap.contrastive_captions),
        professional_term_count=professional_term_count,
        source_market_coverage=source_market_coverage,
        total_score=total_score,
    )


def _case_markdown(window_id: str, api: RiskManagerCaptionV2, codex: RiskManagerCaptionV2) -> str:
    return "\n".join(
        [
            f"### {window_id}",
            "",
            f"- API title/archetype: `{api.scenario_title}` / `{api.archetype}`",
            f"- Codex title/archetype: `{codex.scenario_title}` / `{codex.archetype}`",
            "",
            "**API Training Caption**",
            "",
            api.training_caption,
            "",
            "**Codex Training Caption**",
            "",
            codex.training_caption,
            "",
        ]
    )


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    api_captions = _caption_by_window(Path(args.api_report))
    codex_captions = _load_codex_captions(Path(args.codex_dir))
    pipeline_report = load_pipeline_report(args.pipeline_report)
    common = sorted(set(api_captions) & set(codex_captions))
    if not common:
        raise ValueError("No common API/Codex captions to compare")

    rows = []
    cases = []
    for window_id in common:
        markets = _source_markets(pipeline_report, window_id)
        api_score = score_caption("gpt-5.4-mini-api", api_captions[window_id], markets)
        codex_score = score_caption("codex-gpt-5.5-xhigh", codex_captions[window_id], markets)
        rows.append(
            {
                "window_id": window_id,
                "source_market_count": len(markets),
                "api": api_score.__dict__,
                "codex": codex_score.__dict__,
                "score_delta_codex_minus_api": codex_score.total_score
                - api_score.total_score,
                "same_archetype": api_captions[window_id].archetype
                == codex_captions[window_id].archetype,
            }
        )
        cases.append(_case_markdown(window_id, api_captions[window_id], codex_captions[window_id]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "pass",
        "scope_note": (
            "Matched caption-provider comparison. Scores are deterministic "
            "spot-check heuristics, not downstream scenario-quality proof."
        ),
        "api_report": str(args.api_report),
        "codex_dir": str(args.codex_dir),
        "case_count": len(rows),
        "mean_score_delta_codex_minus_api": sum(
            row["score_delta_codex_minus_api"] for row in rows
        )
        / len(rows),
        "api_validation_errors": sum(row["api"]["validation_error_count"] for row in rows),
        "codex_validation_errors": sum(
            row["codex"]["validation_error_count"] for row in rows
        ),
        "api_validation_warnings": sum(
            row["api"]["validation_warning_count"] for row in rows
        ),
        "codex_validation_warnings": sum(
            row["codex"]["validation_warning_count"] for row in rows
        ),
        "rows": rows,
        "artifact_paths": {
            "json": str(output_dir / "caption_provider_comparison.json"),
            "markdown": str(output_dir / "caption_provider_comparison.md"),
        },
    }
    Path(summary["artifact_paths"]["json"]).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Caption Provider Comparison",
        "",
        "This compares existing `gpt-5.4-mini` API captions against Codex `gpt-5.5` captions on matched windows.",
        "",
        f"- Matched cases: {len(rows)}",
        f"- Mean score delta, Codex minus API: {summary['mean_score_delta_codex_minus_api']:.3f}",
        f"- API validation errors/warnings: {summary['api_validation_errors']} / {summary['api_validation_warnings']}",
        f"- Codex validation errors/warnings: {summary['codex_validation_errors']} / {summary['codex_validation_warnings']}",
        "",
        "## Case Metrics",
        "",
        "| Window | API Title | Codex Title | API Score | Codex Score | Delta | API Words | Codex Words | API Evidence | Codex Evidence | API Contrastives | Codex Contrastives |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {window_id} | {api_title} | {codex_title} | {api_score:.2f} | {codex_score:.2f} | {delta:.2f} | {api_words} | {codex_words} | {api_evidence} | {codex_evidence} | {api_contrastive} | {codex_contrastive} |".format(
                window_id=row["window_id"],
                api_title=row["api"]["title"].replace("|", "/"),
                codex_title=row["codex"]["title"].replace("|", "/"),
                api_score=row["api"]["total_score"],
                codex_score=row["codex"]["total_score"],
                delta=row["score_delta_codex_minus_api"],
                api_words=row["api"]["training_words"],
                codex_words=row["codex"]["training_words"],
                api_evidence=row["api"]["evidence_count"],
                codex_evidence=row["codex"]["evidence_count"],
                api_contrastive=row["api"]["contrastive_count"],
                codex_contrastive=row["codex"]["contrastive_count"],
            )
        )
    lines.extend(["", "## Case Studies", "", "\n".join(cases)])
    Path(summary["artifact_paths"]["markdown"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-report", type=Path, default=DEFAULT_API_REPORT)
    parser.add_argument("--codex-dir", type=Path, default=DEFAULT_CODEX_DIR)
    parser.add_argument("--pipeline-report", type=Path, default=DEFAULT_PIPELINE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = run_comparison(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "case_count": summary["case_count"],
                "mean_score_delta_codex_minus_api": summary[
                    "mean_score_delta_codex_minus_api"
                ],
                "json": summary["artifact_paths"]["json"],
                "markdown": summary["artifact_paths"]["markdown"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
