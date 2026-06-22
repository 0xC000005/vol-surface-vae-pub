#!/usr/bin/env python
"""Regression tests for the 994a Gate [D] CPU tilt-space null band.

Asserts the emitted JSON has per-factor entries with finite p10<=mean<=p90,
that it is built only from the verified-clean 994a arrays (no GPU/model import),
and that the builder module never imports a model/checkpoint/torch dependency.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_994a_null_band_tilt_space import (  # noqa: E402
    SCHEMA_VERSION,
    build_null_band,
    compute_tilt_matrix,
)
from experiments.backfill.block_ar.nl_joint39_anchor_map import (  # noqa: E402
    joint39_anchor_columns,
)

OUTPUT_DIR = (
    ROOT
    / "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "val_frame_eval_994a_start_only"
)
ARRAYS_PATH = OUTPUT_DIR / "scenario_level_eval_arrays.npz"
JSON_PATH = OUTPUT_DIR / "nl_994a_null_band_tilt_space_v1.json"
BUILDER_PATH = (
    ROOT / "experiments/backfill/block_ar/nl_994a_null_band_tilt_space.py"
)

# Models/GPU that the CPU-only builder must never pull in.
FORBIDDEN_IMPORT_SUBSTRINGS = (
    "torch",
    "single_pass_ar",
    "generic_state_aware_normalized_innovation_flow_matching",
    "narrative_conditioning_adapter",
    "evaluate_662a",
    "evaluate_220h",
)


@pytest.fixture(scope="module")
def payload() -> dict:
    assert JSON_PATH.exists(), f"missing emitted null-band JSON: {JSON_PATH}"
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


def test_arrays_input_exists() -> None:
    assert ARRAYS_PATH.exists(), f"clean 994a arrays missing: {ARRAYS_PATH}"


def test_schema_and_block_structure(payload: dict) -> None:
    assert payload["schema_version"] == SCHEMA_VERSION
    assert "scope_note" in payload and "reference band" in payload["scope_note"].lower()
    bs = payload["block_structure"]
    # block length = 30 trading days = 30 // stride query windows (6 at stride 5)
    assert bs["block_len_days"] == 30
    assert bs["block_len_windows"] == bs["block_len_days"] // bs["query_stride"]
    assert bs["block_len_windows"] == 6
    assert bs["n_nonoverlapping_30d_blocks"] >= 1
    assert payload["n_windows"] == len(payload["evaluated_window_indices"])
    assert payload["n_windows"] > 0


def test_per_factor_band_finite_and_ordered(payload: dict) -> None:
    factors = payload["factors"]
    assert factors, "no per-factor entries"
    # All 14 named anchors must be present in the default (anchor) scope.
    anchors = set(joint39_anchor_columns().keys())
    assert anchors.issubset(set(factors)), anchors - set(factors)
    for name, entry in factors.items():
        band = entry["band"]
        vals = [band["p10"], band["mean"], band["std"], band["p90"]]
        assert all(np.isfinite(v) for v in vals), f"{name}: non-finite band {band}"
        assert band["p10"] <= band["mean"] <= band["p90"], (
            f"{name}: p10<=mean<=p90 violated: {band}"
        )
        assert entry["p10_le_mean_le_p90"] is True
        assert band["std"] >= 0.0
        # bootstrap CIs present and finite for each statistic
        for stat in ("p10", "mean", "std", "p90"):
            ci = entry["band_block_bootstrap_ci"][stat]
            assert np.isfinite(ci["ci_low"]) and np.isfinite(ci["ci_high"])
            assert ci["ci_low"] <= ci["ci_high"]


def test_tilt_recomputation_matches_arrays(payload: dict) -> None:
    """The JSON band is reproducible directly from the clean 994a arrays."""

    anchor_cols = joint39_anchor_columns()
    names = list(anchor_cols.keys())
    cols = [anchor_cols[n] for n in names]
    tilt, evaluated = compute_tilt_matrix(ARRAYS_PATH, factor_cols=cols)
    assert evaluated == payload["evaluated_window_indices"]
    for j, name in enumerate(names):
        col = tilt[:, j]
        band = payload["factors"][name]["band"]
        assert band["mean"] == pytest.approx(float(col.mean()), rel=1e-9, abs=1e-9)
        assert band["p10"] == pytest.approx(
            float(np.quantile(col, 0.10)), rel=1e-9, abs=1e-9
        )
        assert band["p90"] == pytest.approx(
            float(np.quantile(col, 0.90)), rel=1e-9, abs=1e-9
        )


def test_provenance_documents_clean_source_and_deferred_gpu(payload: dict) -> None:
    assert "clean" in payload["provenance"].lower()
    assert "no gpu" in payload["provenance"].lower()
    deferred = payload["deferred_gpu_piece"].lower()
    assert "gpu" in deferred and "placebo" in deferred


def test_builder_is_cpu_only_no_model_or_gpu_import() -> None:
    """Static check: the builder module imports no model/torch/GPU dependency."""

    tree = ast.parse(BUILDER_PATH.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    for mod in imported:
        for forbidden in FORBIDDEN_IMPORT_SUBSTRINGS:
            assert forbidden not in mod, (
                f"builder imports forbidden (model/GPU) module {mod!r}"
            )


def test_builder_reproduces_emitted_band(payload: dict) -> None:
    """Re-running the builder deterministically reproduces the point band."""

    rebuilt = build_null_band(arrays_path=ARRAYS_PATH, n_boot=200)
    for name, entry in payload["factors"].items():
        assert rebuilt["factors"][name]["band"] == pytest.approx(
            entry["band"], rel=1e-9, abs=1e-9
        ), name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
