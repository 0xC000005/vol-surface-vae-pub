import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.part1_jepa_latent.analyze_additive_exact_state_guardrail import (  # noqa: E402
    summarize_exact_state_guardrail,
)


def test_summarize_exact_state_guardrail_checks_raw_plus_against_raw_surface():
    probe_metrics = {
        "raw_surface": {"targets": {"iv_surface": {"mse": 0.010}}},
        "scale_barlow": {"targets": {"iv_surface": {"mse": 0.030}}},
        "raw_surface_plus_scale_barlow": {
            "targets": {"iv_surface": {"mse": 0.009}}
        },
        "raw_geometry_upper": {"targets": {"iv_surface": {"mse": 0.008}}},
    }

    summary = summarize_exact_state_guardrail(probe_metrics)

    assert summary["raw_only_iv_mse"] == 0.010
    assert summary["learned_only_iv_mse"] == 0.030
    assert summary["raw_plus_learned_iv_mse"] == 0.009
    assert summary["raw_plus_delta_vs_raw"] < 0.0
    assert summary["status"] == "PASS"
