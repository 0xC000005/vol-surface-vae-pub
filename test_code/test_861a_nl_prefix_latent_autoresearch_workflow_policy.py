import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOAL = ROOT / "docs/research_protocols/nl_prefix_latent_goal.json"
PLAN = ROOT / "docs/research_protocols/nl_prefix_latent_autoresearch_plan.md"
CURRENT_TRUTH = ROOT / "docs/research_protocols/nl_prefix_latent_current_truth.md"
MULTIAGENT_REVIEW = (
    ROOT / "docs/research_protocols/nl_prefix_autoresearch_multiagent_review.md"
)
CASE_SPEC = (
    ROOT
    / "docs/research_protocols/nl_prefix_latent_promoted_specs/"
    / "fixed_start_narrative_matrix_858b_cases.json"
)
VERIFIER_REPORT = (
    ROOT
    / "docs/research_protocols/nl_prefix_latent_verifier_reports/"
    / "2026-05-11_workflow_audit.md"
)


def test_tracked_goal_disallows_hidden_model_chosen_start() -> None:
    goal = json.loads(GOAL.read_text(encoding="utf-8"))

    assert "narrative_only_model_chosen_start" in goal["disallowed_default_modes"]
    assert "hidden_model_chosen_start" in goal["disallowed_default_modes"]
    assert goal["production_modes"] == [
        "narrative_plus_manual_historical_or_current_start",
        "narrative_plus_user_specified_joint39_start",
    ]
    assert "resolve_user_selected_or_user_supplied_start_before_support_mixture" in goal[
        "core_contract"
    ]


def test_protocol_requires_promotion_artifacts_and_null_controls() -> None:
    plan = PLAN.read_text(encoding="utf-8")

    assert "nl_prefix_latent_current_truth.md" in plan
    assert "nl_prefix_latent_verifier_reports" in plan
    assert "nl_prefix_latent_promoted_specs" in plan
    assert "nl_prefix_autoresearch_multiagent_review.md" in plan
    assert "same narrative, same fixed start, different rollout seeds" in plan
    assert "shuffled narratives assigned to the same fixed starts" in plan
    assert "per-start quality floors" in plan
    assert "Multi-Agent Sidecar Policy" in plan


def test_promoted_fixed_start_case_spec_is_tracked_and_complete() -> None:
    payload = json.loads(CASE_SPEC.read_text(encoding="utf-8"))
    cases = payload["cases"]

    assert len(cases) == 36
    starts = {case["candidate_index"] for case in cases}
    narratives = {
        case["case_name"].rsplit("_start", 1)[0]
        for case in cases
    }
    assert starts == {0, 18, 22, 40, 77, 178}
    assert narratives == {
        "fragile_risk_on",
        "defensive_risk_off",
        "rates_selloff",
        "commodity_inflation",
        "dollar_liquidity",
        "safe_haven_gold",
    }


def test_current_truth_and_verifier_report_capture_warning_limits() -> None:
    current_truth = CURRENT_TRUTH.read_text(encoding="utf-8")
    verifier = VERIFIER_REPORT.read_text(encoding="utf-8")

    assert "Not Yet Promoted" in current_truth
    assert "Arbitrary live user narrative plus arbitrary user-supplied joint39 start" in current_truth
    assert "starts `0` and `178` damp narrative influence" in current_truth
    assert "Partially working" in verifier
    assert "Direction pass can be inflated by the selection mechanism" in verifier


def test_multiagent_policy_keeps_sidecars_bounded() -> None:
    review = MULTIAGENT_REVIEW.read_text(encoding="utf-8")

    assert "centralized HEAD orchestrator" in review
    assert "Literature Scout" in review
    assert "Experiment Critic" in review
    assert "Artifact Verifier" in review
    assert "Report Auditor" in review
    assert "Implementation Worker" in review
    assert "narrow sequential debugging" in review
    assert "metric truth by debate" in review
