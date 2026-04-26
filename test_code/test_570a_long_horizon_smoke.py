import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.generate_568a_risk_scenario_deck import (
    BlockwiseLongHorizonModel,
    build_manifest,
    scenario_diagnostics,
)


def test_scenario_diagnostics_summarize_long_horizon_stability() -> None:
    scenarios = np.array(
        [
            [[[0.10]], [[0.20]], [[0.30]]],
            [[[0.15]], [[0.25]], [[0.35]]],
        ],
        dtype=np.float32,
    )

    diagnostics = scenario_diagnostics(scenarios)

    assert diagnostics["finite_rate"] == 1.0
    assert diagnostics["min_iv"] == 0.1
    assert diagnostics["max_iv"] == 0.35
    assert diagnostics["terminal_mean_iv"] == 0.325


def test_build_manifest_can_carry_252_day_diagnostics() -> None:
    diagnostics = {
        "finite_rate": 1.0,
        "min_iv": 0.01,
        "max_iv": 0.8,
        "terminal_mean_iv": 0.2,
    }

    manifest = build_manifest(
        model_type="340c",
        checkpoint="checkpoint.pt",
        data_path="data.npz",
        history_start_index=100,
        history_end_index=130,
        history_len=30,
        future_len=252,
        samples=3,
        candidate_count=9,
        seed=570,
        scenario_shape=(3, 252, 5, 5),
        path_mean_iv=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        scenario_diagnostics=diagnostics,
    )

    assert manifest["future_len"] == 252
    assert manifest["scenario_diagnostics"] == diagnostics


def test_blockwise_long_horizon_model_rolls_past_base_limit() -> None:
    class FakeConfig:
        future_len = 2

    class FakeBase(torch.nn.Module):
        cfg = FakeConfig()

        def sample_batched(
            self,
            history: torch.Tensor,
            n_samples: int,
            n_steps: int,
            chunk_size: int,
            history_is_normalized: bool,
            **_: object,
        ) -> torch.Tensor:
            if n_steps > self.cfg.future_len:
                raise ValueError("base limit exceeded")
            last = history[:, -1].unsqueeze(1).unsqueeze(1)
            offsets = torch.arange(1, n_steps + 1, dtype=history.dtype).view(1, 1, n_steps, 1, 1)
            return last + offsets.expand(history.shape[0], n_samples, n_steps, 5, 5)

    wrapper = BlockwiseLongHorizonModel(FakeBase(), max_block_steps=2)
    history = torch.zeros((1, 3, 5, 5), dtype=torch.float32)

    samples = wrapper.sample_batched(
        history,
        n_samples=2,
        n_steps=5,
        chunk_size=1,
        history_is_normalized=False,
    )

    assert samples.shape == (1, 2, 5, 5, 5)
    assert torch.allclose(samples[0, 0, :, 0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
