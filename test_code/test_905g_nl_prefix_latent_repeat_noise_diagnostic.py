import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_repeat_noise_diagnostic import (
    build_repeat_noise_report,
)


VARIANT = "decoder_component_topk_narrative_start_checked_gen_temp_0p50"


def _write_case(path: Path, *, prefix_value: float, path_shift: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "prefix_latent_story_smoke_report.json").write_text(
        json.dumps({"variant_rows": [{"is_operational": True}]}),
        encoding="utf-8",
    )
    states = np.zeros((2, 2, 39), dtype=np.float32)
    states[0, 0, :] = path_shift
    states[0, 1, :] = path_shift + 0.1
    states[1, 0, :] = path_shift + 0.2
    states[1, 1, :] = path_shift + 0.3
    np.savez_compressed(
        path / "prefix_latent_story_smoke_arrays.npz",
        generated_states=states[None, ...],
        requested_raw=np.zeros((1, 39), dtype=np.float32),
        decoded_history_level=np.full((1, 2, 39), prefix_value, dtype=np.float32),
        rollout_component_variant_index=np.asarray([0, 0], dtype=np.int64),
        rollout_component_window_index=np.asarray([10, 11], dtype=np.int64),
        rollout_component_weight=np.asarray([0.5, 0.5], dtype=np.float32),
    )


def test_repeat_noise_diagnostic_detects_prefix_instability(tmp_path: Path) -> None:
    component = tmp_path / "component"
    control = tmp_path / "control"
    _write_case(
        component / "case_a_start18" / "fixed_start_18" / VARIANT,
        prefix_value=0.0,
        path_shift=0.0,
    )
    _write_case(
        component / "case_b_start18" / "fixed_start_18" / VARIANT,
        prefix_value=4.0,
        path_shift=2.0,
    )
    _write_case(
        control / "repeat_controls" / "case_a_start18" / "seed_1",
        prefix_value=0.0,
        path_shift=0.0,
    )
    _write_case(
        control / "repeat_controls" / "case_a_start18" / "seed_2",
        prefix_value=3.0,
        path_shift=0.1,
    )

    report = build_repeat_noise_report(
        component_root=component,
        control_root=control,
        variant_dir=VARIANT,
    )

    assert report["status"] == "warning"
    assert report["diagnosis"] == "prefix_decoder_or_support_instability"
    assert report["ratios"]["repeat_to_observed_decoded_prefix_l2"] > 0.5
