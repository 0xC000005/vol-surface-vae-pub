import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.train_593a_ar_common_latent_wrapper import (  # noqa: E402
    ARCommonLatentWrapper,
)


class TinyBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = type(
            "Cfg",
            (),
            {
                "history_len": 3,
                "future_len": 2,
                "n_cells": 4,
                "memory_dim": 6,
                "flow_steps": 2,
                "sample_temperature": 1.0,
            },
        )()
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def history_scores(self, history_norm: torch.Tensor) -> torch.Tensor:
        return history_norm

    def _encode_prefix_scores(self, prefix: torch.Tensor) -> torch.Tensor:
        summed = prefix.sum(dim=-1, keepdim=True).expand(prefix.shape[0], prefix.shape[1], self.cfg.memory_dim)
        return summed / 10.0

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        current_score: torch.Tensor,
        memory_state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        del t
        return 0.1 * x_t + 0.01 * current_score + 0.01 * memory_state[:, : self.cfg.n_cells]

    def _scores_to_values(self, scores: torch.Tensor) -> torch.Tensor:
        return scores


def test_ar_common_latent_wrapper_samples_scores_shape() -> None:
    wrapper = ARCommonLatentWrapper(TinyBase(), latent_dim=3)
    history = torch.randn(2, 3, 4)

    scores = wrapper._sample_scores(history, n_samples=5, n_steps=2, flow_steps=2)

    assert scores.shape == (2, 5, 2, 4)
    assert torch.isfinite(scores).all()
    assert not any(param.requires_grad for param in wrapper.base_model.parameters())
