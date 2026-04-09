#!/usr/bin/env python
"""
211a: H=1 pure decoder-only token transformer.

Minimal decoder-only follow-up after the 210e/210f/210g latent failures:
  - no prior/posterior latent
  - no codebook
  - no teacher-guided event token
  - no severity-sidecar prefix
  - one Hugging Face GPT-style causal decoder

Context:
  - history input uses transformed levels + transformed deltas
  - target output is the next-day transformed delta
  - target is generated autoregressively across the 25 cells
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers.cache_utils import DynamicCache
from transformers import GPT2Config, GPT2LMHeadModel

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    build_one_step_windows,
    iv_to_unconstrained,
    make_serializable,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_208a_h1_teacher_guided_local_family_student_t import (
    compute_h1_shape_stats,
)


def fit_quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    q = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    edges = np.quantile(values.astype(np.float64), q).astype(np.float32)
    for i in range(1, edges.shape[0]):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges


@dataclass
class SurfaceTokenQuantizer:
    level_edges: np.ndarray
    delta_edges: np.ndarray

    pad_id: int = 0
    start_target_id: int = 1

    @property
    def n_level_bins(self) -> int:
        return int(self.level_edges.shape[0] - 1)

    @property
    def n_delta_bins(self) -> int:
        return int(self.delta_edges.shape[0] - 1)

    @property
    def level_offset(self) -> int:
        return 2

    @property
    def delta_offset(self) -> int:
        return self.level_offset + self.n_level_bins

    @property
    def vocab_size(self) -> int:
        return self.delta_offset + self.n_delta_bins

    @property
    def level_centers(self) -> np.ndarray:
        return 0.5 * (self.level_edges[:-1] + self.level_edges[1:])

    @property
    def delta_centers(self) -> np.ndarray:
        return 0.5 * (self.delta_edges[:-1] + self.delta_edges[1:])

    @classmethod
    def fit(
        cls,
        level_values: np.ndarray,
        delta_values: np.ndarray,
        n_level_bins: int,
        n_delta_bins: int,
    ) -> "SurfaceTokenQuantizer":
        return cls(
            level_edges=fit_quantile_edges(level_values, n_level_bins),
            delta_edges=fit_quantile_edges(delta_values, n_delta_bins),
        )

    def encode_level(self, values: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.level_edges[1:-1], values, side="right").astype(np.int64)
        return idx + self.level_offset

    def encode_delta(self, values: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.delta_edges[1:-1], values, side="right").astype(np.int64)
        return idx + self.delta_offset

    def decode_delta_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        centers = torch.as_tensor(self.delta_centers, device=token_ids.device, dtype=torch.float32)
        idx = (token_ids - self.delta_offset).clamp_(0, self.n_delta_bins - 1)
        return centers[idx]


def prepare_quantizer_inputs(
    history_01: torch.Tensor,
    target_01: torch.Tensor,
    support_lo: float,
    support_hi: float,
    support_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    hist_u = iv_to_unconstrained(history_01.reshape(history_01.shape[0], history_01.shape[1], -1), lo=support_lo, hi=support_hi, eps=support_eps)
    target_u = iv_to_unconstrained(target_01, lo=support_lo, hi=support_hi, eps=support_eps)

    hist_delta_u = torch.zeros_like(hist_u)
    hist_delta_u[:, 1:] = hist_u[:, 1:] - hist_u[:, :-1]
    target_delta_u = target_u - hist_u[:, -1]

    level_values = hist_u.detach().cpu().numpy().reshape(-1)
    delta_values = np.concatenate(
        [
            hist_delta_u.detach().cpu().numpy().reshape(-1),
            target_delta_u.detach().cpu().numpy().reshape(-1),
        ],
        axis=0,
    )
    return level_values, delta_values


def build_tokenized_sequences(
    history_01: torch.Tensor,
    target_01: torch.Tensor,
    quantizer: SurfaceTokenQuantizer,
    support_lo: float,
    support_hi: float,
    support_eps: float,
) -> dict[str, torch.Tensor]:
    hist_u = iv_to_unconstrained(
        history_01.reshape(history_01.shape[0], history_01.shape[1], -1),
        lo=support_lo,
        hi=support_hi,
        eps=support_eps,
    )
    target_u = iv_to_unconstrained(target_01, lo=support_lo, hi=support_hi, eps=support_eps)

    hist_delta_u = torch.zeros_like(hist_u)
    hist_delta_u[:, 1:] = hist_u[:, 1:] - hist_u[:, :-1]
    target_delta_u = target_u - hist_u[:, -1]

    level_ids = torch.from_numpy(quantizer.encode_level(hist_u.detach().cpu().numpy())).long()
    delta_hist_ids = torch.from_numpy(quantizer.encode_delta(hist_delta_u.detach().cpu().numpy())).long()
    target_delta_ids = torch.from_numpy(quantizer.encode_delta(target_delta_u.detach().cpu().numpy())).long()

    prefix_tokens = torch.cat([level_ids, delta_hist_ids], dim=-1).reshape(history_01.shape[0], -1)
    start_tokens = torch.full((history_01.shape[0], 1), quantizer.start_target_id, dtype=torch.long)
    input_ids = torch.cat([prefix_tokens, start_tokens, target_delta_ids], dim=1)
    labels = input_ids.clone()
    labels[:, : prefix_tokens.shape[1] + 1] = -100

    return {
        "input_ids": input_ids,
        "labels": labels,
        "prefix_ids": torch.cat([prefix_tokens, start_tokens], dim=1),
        "target_delta_ids": target_delta_ids,
    }


class H1DecoderOnlyTokenTransformer(nn.Module):
    def __init__(
        self,
        quantizer: SurfaceTokenQuantizer,
        history_len: int,
        n_cells: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        support_lo: float = 0.01,
        support_hi: float = 1.0,
        support_eps: float = 1e-5,
    ):
        super().__init__()
        self.quantizer = quantizer
        self.history_len = history_len
        self.n_cells = n_cells
        self.support_lo = support_lo
        self.support_hi = support_hi
        self.support_eps = support_eps
        self.prefix_len = history_len * n_cells * 2 + 1
        self.total_seq_len = self.prefix_len + n_cells

        cfg = GPT2Config(
            vocab_size=quantizer.vocab_size,
            n_positions=self.total_seq_len + 8,
            n_ctx=self.total_seq_len + 8,
            n_embd=d_model,
            n_layer=n_layers,
            n_head=n_heads,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            bos_token_id=quantizer.start_target_id,
            eos_token_id=None,
            pad_token_id=quantizer.pad_id,
            use_cache=True,
        )
        self.backbone = GPT2LMHeadModel(cfg)

    def _repeat_past_key_values(
        self,
        past_key_values: DynamicCache,
        repeats: int,
    ) -> DynamicCache:
        cache = DynamicCache(list(past_key_values), config=self.backbone.config)
        cache.batch_repeat_interleave(repeats)
        return cache

    def forward_loss(self, input_ids: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = self.backbone(input_ids=input_ids, labels=labels)
        logits = out.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        if mask.any():
            pred = shift_logits.argmax(dim=-1)
            token_top1 = (pred[mask] == shift_labels[mask]).float().mean()
            probs = torch.softmax(shift_logits[mask], dim=-1)
            token_entropy = (-(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)).mean()
        else:
            token_top1 = logits.new_tensor(0.0)
            token_entropy = logits.new_tensor(0.0)
        metrics = {
            "token_ce": out.loss.detach(),
            "token_top1": token_top1.detach(),
            "token_entropy": token_entropy.detach(),
        }
        return out.loss, metrics

    @torch.no_grad()
    def prefix_cache(
        self,
        prefix_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, DynamicCache]:
        out = self.backbone(input_ids=prefix_ids, use_cache=True)
        return out.logits[:, -1, :], out.past_key_values

    def _sample_next_token(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        logits = logits / max(temperature, 1e-6)
        allowed = torch.zeros_like(logits, dtype=torch.bool)
        allowed[:, self.quantizer.delta_offset : self.quantizer.delta_offset + self.quantizer.n_delta_bins] = True
        logits = logits.masked_fill(~allowed, -float("inf"))

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        cutoff = cumulative > top_p
        cutoff[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, -float("inf"))
        filtered = torch.full_like(logits, -float("inf"))
        filtered.scatter_(1, sorted_indices, sorted_logits)
        probs = torch.softmax(filtered, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def sample_target_delta_ids(
        self,
        prefix_ids: torch.Tensor,
        n_samples: int,
        temperature: float = 0.9,
        top_p: float = 0.95,
        sample_batch_size: int = 16,
        prefix_cache: tuple[torch.Tensor, DynamicCache] | None = None,
    ) -> torch.Tensor:
        self.eval()
        batch = prefix_ids.shape[0]
        outputs = []
        device = prefix_ids.device
        if prefix_cache is None:
            base_logits, base_past = self.prefix_cache(prefix_ids)
        else:
            base_logits, base_past = prefix_cache

        for start in range(0, n_samples, sample_batch_size):
            per_hist = min(sample_batch_size, n_samples - start)
            logits = base_logits.repeat_interleave(per_hist, dim=0)
            past = self._repeat_past_key_values(base_past, per_hist)
            generated = []
            for _ in range(self.n_cells):
                next_tok = self._sample_next_token(logits, temperature=temperature, top_p=top_p)
                generated.append(next_tok)
                out = self.backbone(input_ids=next_tok.unsqueeze(1), past_key_values=past, use_cache=True)
                logits = out.logits[:, -1, :]
                past = out.past_key_values
            chunk = torch.stack(generated, dim=1).view(batch, per_hist, self.n_cells)
            outputs.append(chunk)
        return torch.cat(outputs, dim=1).to(device)

    @torch.no_grad()
    def decode_next_iv_from_delta_token_ids(
        self,
        token_ids: torch.Tensor,
        prev_01: torch.Tensor,
    ) -> torch.Tensor:
        delta_u = self.quantizer.decode_delta_token_ids(token_ids)
        prev_u = iv_to_unconstrained(prev_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)
        next_u = prev_u.unsqueeze(1) + delta_u
        return unconstrained_to_iv(next_u, lo=self.support_lo, hi=self.support_hi)

    @torch.no_grad()
    def sample_next_iv(
        self,
        prefix_ids: torch.Tensor,
        prev_01: torch.Tensor,
        n_samples: int,
        temperature: float = 0.9,
        top_p: float = 0.95,
        sample_batch_size: int = 16,
        prefix_cache: tuple[torch.Tensor, DynamicCache] | None = None,
    ) -> torch.Tensor:
        token_ids = self.sample_target_delta_ids(
            prefix_ids=prefix_ids,
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
            sample_batch_size=sample_batch_size,
            prefix_cache=prefix_cache,
        )
        return self.decode_next_iv_from_delta_token_ids(token_ids, prev_01=prev_01)


@torch.no_grad()
def evaluate_h1(
    model: H1DecoderOnlyTokenTransformer,
    loader: DataLoader,
    q95_threshold: float,
    q99_threshold: float,
    eval_samples: int,
    sample_batch_size: int,
    temperature: float,
    top_p: float,
) -> dict[str, float]:
    model.eval()
    totals = {
        "val_token_ce": 0.0,
        "val_token_top1": 0.0,
        "val_token_entropy": 0.0,
        "val_mae": 0.0,
        "val_coverage_90": 0.0,
        "val_width_90": 0.0,
    }
    q95_cover_sum = 0.0
    q99_cover_sum = 0.0
    q95_count = 0
    q99_count = 0
    total_count = 0
    gt_delta_all = []
    sample_delta_all = []
    unique_seq_sum = 0.0

    for input_ids, labels, prefix_ids, history_01, target_01 in loader:
        input_ids = input_ids.to(next(model.parameters()).device)
        labels = labels.to(next(model.parameters()).device)
        prefix_ids = prefix_ids.to(next(model.parameters()).device)
        history_01 = history_01.to(next(model.parameters()).device)
        target_01 = target_01.to(next(model.parameters()).device)

        ce_loss, token_metrics = model.forward_loss(input_ids, labels)
        prev = history_01[:, -1].reshape(history_01.shape[0], -1)
        prefix_state = model.prefix_cache(prefix_ids)
        sample_tokens = model.sample_target_delta_ids(
            prefix_ids=prefix_ids,
            n_samples=eval_samples,
            temperature=temperature,
            top_p=top_p,
            sample_batch_size=sample_batch_size,
            prefix_cache=prefix_state,
        )
        samples = model.decode_next_iv_from_delta_token_ids(sample_tokens, prev_01=prev)

        q05 = samples.quantile(0.05, dim=1)
        q95 = samples.quantile(0.95, dim=1)
        mean_pred = samples.mean(dim=1)

        target_abs = (target_01 - prev).abs()
        q95_mask = target_abs >= q95_threshold
        q99_mask = target_abs >= q99_threshold
        coverage = ((target_01 >= q05) & (target_01 <= q95)).float().mean()
        mae = (mean_pred - target_01).abs().mean()
        width = (q95 - q05).mean()
        q95_cov = (
            ((target_01[q95_mask] >= q05[q95_mask]) & (target_01[q95_mask] <= q95[q95_mask])).float().mean()
            if q95_mask.any()
            else target_01.new_tensor(0.0)
        )
        q99_cov = (
            ((target_01[q99_mask] >= q05[q99_mask]) & (target_01[q99_mask] <= q95[q99_mask])).float().mean()
            if q99_mask.any()
            else target_01.new_tensor(0.0)
        )

        batch_size = input_ids.shape[0]
        totals["val_token_ce"] += float(ce_loss.item()) * batch_size
        totals["val_token_top1"] += float(token_metrics["token_top1"].item()) * batch_size
        totals["val_token_entropy"] += float(token_metrics["token_entropy"].item()) * batch_size
        totals["val_mae"] += float(mae.item()) * batch_size
        totals["val_coverage_90"] += float(coverage.item()) * batch_size
        totals["val_width_90"] += float(width.item()) * batch_size

        if q95_mask.any():
            q95_cover_sum += float(q95_cov.item()) * int(q95_mask.sum().item())
            q95_count += int(q95_mask.sum().item())
        if q99_mask.any():
            q99_cover_sum += float(q99_cov.item()) * int(q99_mask.sum().item())
            q99_count += int(q99_mask.sum().item())

        gt_delta_all.append((target_01 - prev).detach().cpu().numpy())
        sample_delta_all.append((samples - prev.unsqueeze(1)).detach().cpu().numpy())
        unique_tokens = sample_tokens[:, : min(64, sample_tokens.shape[1])].detach().cpu().numpy()
        unique_seq_sum += float(
            np.mean([np.unique(unique_tokens[i], axis=0).shape[0] / max(unique_tokens.shape[1], 1) for i in range(unique_tokens.shape[0])])
        ) * batch_size
        total_count += batch_size

    gt_delta = np.concatenate(gt_delta_all, axis=0)
    sample_delta = np.concatenate(sample_delta_all, axis=0)
    shape = compute_h1_shape_stats(gt_delta, sample_delta)

    metrics = {k: v / max(total_count, 1) for k, v in totals.items()}
    metrics.update(
        {
            "val_realized_q95_coverage_90": q95_cover_sum / max(q95_count, 1),
            "val_realized_q99_coverage_90": q99_cover_sum / max(q99_count, 1),
            "val_q95_cell_count": q95_count,
            "val_q99_cell_count": q99_count,
            "val_h1_quiet_ratio": shape["quiet_ratio"],
            "val_h1_shoulder_ratio": shape["shoulder_ratio"],
            "val_h1_extreme_ratio": shape["extreme_ratio"],
            "val_h1_kurtosis_ratio": shape["kurtosis_ratio"],
            "val_unique_sequence_ratio": unique_seq_sum / max(total_count, 1),
        }
    )
    for key, value in shape.items():
        if key in {"quiet_ratio", "shoulder_ratio", "extreme_ratio", "kurtosis_ratio"}:
            continue
        metrics[f"val_h1_{key}"] = float(value)
    return metrics


def load_model(checkpoint_path: str, device: torch.device) -> tuple[H1DecoderOnlyTokenTransformer, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = payload["config"]
    if raw_config["type"] != "transformer_h1_decoder_only_token_211a":
        raise ValueError(f"Expected 211a checkpoint, got {raw_config['type']}")
    quantizer = SurfaceTokenQuantizer(
        level_edges=np.asarray(raw_config["level_edges"], dtype=np.float32),
        delta_edges=np.asarray(raw_config["delta_edges"], dtype=np.float32),
    )
    model = H1DecoderOnlyTokenTransformer(
        quantizer=quantizer,
        history_len=raw_config["history_len"],
        n_cells=raw_config["n_cells"],
        d_model=raw_config["d_model"],
        n_heads=raw_config["n_heads"],
        n_layers=raw_config["n_layers"],
        dropout=raw_config["dropout"],
        support_lo=raw_config.get("support_lo", 0.01),
        support_hi=raw_config.get("support_hi", 1.0),
        support_eps=raw_config.get("support_eps", 1e-5),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="211a H=1 pure decoder-only token transformer")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=512)
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--eval_samples", type=int, default=256)
    parser.add_argument("--sample_batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--n_level_bins", type=int, default=256)
    parser.add_argument("--n_delta_bins", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    train_abs_delta = np.abs(np.diff(surfaces[: args.test_start], axis=0).reshape(-1))
    q95_threshold = float(np.quantile(train_abs_delta, 0.95))
    q99_threshold = float(np.quantile(train_abs_delta, 0.99))

    max_train_idx = args.test_start - args.history_len - 30
    train_indices = np.arange(0, max_train_idx - args.val_size)
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    if args.max_train_windows is not None:
        train_indices = train_indices[: args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]

    surf_tensor = torch.from_numpy(surfaces)
    train_hist, train_target = build_one_step_windows(train_indices, surf_tensor, args.history_len)
    val_hist, val_target = build_one_step_windows(val_indices, surf_tensor, args.history_len)

    level_values, delta_values = prepare_quantizer_inputs(
        history_01=train_hist,
        target_01=train_target,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
    )
    quantizer = SurfaceTokenQuantizer.fit(
        level_values=level_values,
        delta_values=delta_values,
        n_level_bins=args.n_level_bins,
        n_delta_bins=args.n_delta_bins,
    )

    train_tok = build_tokenized_sequences(
        history_01=train_hist,
        target_01=train_target,
        quantizer=quantizer,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
    )
    val_tok = build_tokenized_sequences(
        history_01=val_hist,
        target_01=val_target,
        quantizer=quantizer,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
    )

    train_loader = DataLoader(
        TensorDataset(train_tok["input_ids"], train_tok["labels"]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            val_tok["input_ids"],
            val_tok["labels"],
            val_tok["prefix_ids"],
            val_hist,
            val_target,
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = H1DecoderOnlyTokenTransformer(
        quantizer=quantizer,
        history_len=args.history_len,
        n_cells=25,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1)

    n_params = sum(p.numel() for p in model.parameters())
    print("211a H=1 pure decoder-only token transformer")
    print(f"  Train windows: {train_hist.shape[0]}")
    print(f"  Val windows:   {val_hist.shape[0]}")
    print(f"  Params:        {n_params:,}")
    print(f"  q95={q95_threshold:.5f} q99={q99_threshold:.5f}")
    print(f"  Vocab size:    {quantizer.vocab_size}")
    print(f"  Seq len:       {model.total_seq_len}")
    print("  Backend:       transformers.GPT2LMHeadModel")

    best_score = float("inf")
    best_metrics: dict[str, Any] | None = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep = {
            "train_loss": 0.0,
            "train_token_ce": 0.0,
            "train_token_top1": 0.0,
            "train_token_entropy": 0.0,
        }
        nb = 0

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss, metrics = model.forward_loss(input_ids, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            ep["train_loss"] += float(loss.item())
            ep["train_token_ce"] += float(metrics["token_ce"].item())
            ep["train_token_top1"] += float(metrics["token_top1"].item())
            ep["train_token_entropy"] += float(metrics["token_entropy"].item())
            nb += 1

        scheduler.step()
        train_metrics = {k: v / max(nb, 1) for k, v in ep.items()}
        val_metrics = evaluate_h1(
            model=model,
            loader=val_loader,
            q95_threshold=q95_threshold,
            q99_threshold=q99_threshold,
            eval_samples=args.eval_samples,
            sample_batch_size=args.sample_batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        gap = (
            max(0.0, 0.85 - val_metrics["val_coverage_90"])
            + max(0.0, val_metrics["val_coverage_90"] - 0.93)
            + max(0.0, 0.58 - val_metrics["val_realized_q99_coverage_90"])
            + max(0.0, 0.85 - val_metrics["val_h1_quiet_ratio"])
            + max(0.0, val_metrics["val_h1_shoulder_ratio"] - 1.10)
            + max(0.0, 0.65 - val_metrics["val_h1_kurtosis_ratio"])
        )
        selection_score = gap + 0.01 * val_metrics["val_token_ce"]
        val_metrics["selection_gap"] = gap
        val_metrics["selection_score"] = selection_score

        row = {"epoch": epoch, **train_metrics, **val_metrics, "elapsed_sec": time.time() - t0}
        history.append(make_serializable(row))

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": {
                "type": "transformer_h1_decoder_only_token_211a",
                "history_len": args.history_len,
                "n_cells": 25,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
                "support_lo": args.support_lo,
                "support_hi": args.support_hi,
                "support_eps": args.support_eps,
                "n_level_bins": args.n_level_bins,
                "n_delta_bins": args.n_delta_bins,
                "level_edges": quantizer.level_edges.tolist(),
                "delta_edges": quantizer.delta_edges.tolist(),
            },
            "metrics": make_serializable(row),
        }
        torch.save(ckpt, output_dir / "final_model.pt")
        if selection_score < best_score:
            best_score = selection_score
            best_metrics = dict(row)
            torch.save(ckpt, output_dir / "best_model.pt")

        with open(output_dir / "training_history.json", "w") as f:
            json.dump(make_serializable(history), f, indent=2)

        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"loss={train_metrics['train_loss']:.4f}  "
            f"top1={train_metrics['train_token_top1']:.3f}  "
            f"valCE={val_metrics['val_token_ce']:.4f}  "
            f"cov90={val_metrics['val_coverage_90']:.3f}  "
            f"q99cov={val_metrics['val_realized_q99_coverage_90']:.3f}  "
            f"quiet={val_metrics['val_h1_quiet_ratio']:.3f}  "
            f"shoulder={val_metrics['val_h1_shoulder_ratio']:.3f}  "
            f"kurt={val_metrics['val_h1_kurtosis_ratio']:.3f}  "
            f"uniq={val_metrics['val_unique_sequence_ratio']:.3f}  "
            f"score={selection_score:.4f}"
        )

    summary = {
        "best_score": best_score,
        "best_metrics": make_serializable(best_metrics),
        "thresholds": {"q95": q95_threshold, "q99": q99_threshold},
        "vocab_size": quantizer.vocab_size,
        "seq_len": model.total_seq_len,
        "backend": "transformers.GPT2LMHeadModel",
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(make_serializable(summary), f, indent=2)

    print("Done.")
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
