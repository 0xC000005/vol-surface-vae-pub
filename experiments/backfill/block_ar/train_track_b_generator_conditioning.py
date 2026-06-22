#!/usr/bin/env python
"""Track B Task 4 -- train the narrative-conditioning probe (B-direction).

This trains ONLY the ``narrative_adapter`` wired into the frozen SNI generator
(``generic_state_aware_normalized_innovation_flow_matching``) on a SEPARATE checkpoint.
The 734a backbone is loaded read-only, frozen, and never written.

Objective (kept deliberately simple -- the directional / real-vs-shuffled / metamorphic
GATES live in Task 6, not here): the model's EXISTING velocity-matching
``training_loss(..., narrative_emb=emb, narrative_present=present)`` over (window, its
realized future) pairs WITH the narrative channel active. Minimizing that loss makes the
adapter learn a narrative -> additive memory_states context that helps reconstruct the
window's realized future -- i.e. the narrative steers the rollout. A fraction of examples
are passed with ``narrative_present=False`` (zero context) so the no-op/baseline behaviour
is anchored.

Data path (canonical, reused -- NOT reinvented):
- Windows + SNI tensors come from the SAME construction the support bank used:
  ``build_val_block`` (evaluate_662a_state_aware_normalized_innovation_flow.py:48) for the
  start arrays + ``block``, and ``select_normalized_innovation_scope``
  (train_662a_state_aware_normalized_innovation_flow.py:47) on that same ``block`` to
  recover ``future_level`` / ``future_norm`` (which ``build_val_block`` computes then
  discards as ``_future_level`` / ``_future_norm``). The bank builder
  (nl_prefix_latent_support_bank.py:98 ``build_support_bank``) used the identical path, so
  our windows line up with the bank row-for-row -- asserted to atol 1e-4 against
  ``support_bank_arrays.npz``. This guarantees the narrative text -> window mapping (by bank
  local index) is correct, and that ``future_norm`` is canonical for free.
- Narrative text per window: positive examples from the clean self-supervised manifest
  ``stride5_self_supervised_training_manifest_clean_20260615/training_examples.jsonl``,
  keyed by ``label_window_index`` (== bank local window index, range 0..4009).
- Embeddings: ``--embed-mode openai`` (text-embedding-3-small, cached) for real training, or
  ``--embed-mode hash`` (deterministic, NO API calls) for the CPU smoke. A window without a
  cached/derivable embedding is treated as ``present=False``.

Split: windows whose forecast region overlaps the held-out validation frame (the same val
ranges + purge gap used by the calibration) are EXCLUDED from training.

Outputs: ``models/backfill/generator_conditioning_probe_b1/`` (best_model.pt +
training_history.json). 734a is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    GenericStateAwareNormalizedInnovationFMConfig,
    GenericStateAwareNormalizedInnovationFlowMatching,
    load_model,
    save_checkpoint,
)
from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    build_val_block,
)
from experiments.backfill.block_ar.train_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    select_normalized_innovation_scope,
)

# --- fixed defaults that reproduce the support bank's window set (from its report) -------
DEFAULT_CHECKPOINT = (
    "models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/"
    "best_model.pt"
)
DEFAULT_SUPPORT_BANK = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
)
DEFAULT_MANIFEST_EXAMPLES = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "stride5_self_supervised_training_manifest_clean_20260615/training_examples.jsonl"
)
DEFAULT_OUTPUT_DIR = "models/backfill/generator_conditioning_probe_b1"

# Bank-build config (support_bank_report.json -> "config") + checkpoint scope args.
BANK_BUILD = {
    "eval_split": "train",
    "test_start": 4511,
    "val_size": 441,
    "max_windows": 0,
    "decoder_test_windows": 66,
    "state_scope": "joint38",
    "iv_count": 25,
}
# Held-out validation frame + purge gap, COPIED VERBATIM from the calibration's
# build_masks (nl_locality_soft_fit_gate_calibration_t2.py:62-63). These are WINDOW
# indices over [0, bank_size); for this bank source_index == window index (0..4009,
# stride 1, verified), so the masks apply directly on the window row.
VAL_RANGES = ((610, 730), (1490, 1610), (2370, 2490), (3250, 3370), (3915, 4010))
PURGE_GAP = 30
EMBED_DIM = 1536  # cfg.narrative_dim default; text-embedding-3-small is 1536-d.


def _bank_build_namespace(checkpoint: str) -> Namespace:
    """Namespace replicating the support-bank build args (so build_val_block reproduces it)."""
    return Namespace(
        checkpoint=checkpoint,
        clean_nonpositive_log_levels=True,
        iv_count=int(BANK_BUILD["iv_count"]),
        test_start=int(BANK_BUILD["test_start"]),
        val_size=int(BANK_BUILD["val_size"]),
        max_windows=int(BANK_BUILD["max_windows"]),
        eval_split=str(BANK_BUILD["eval_split"]),
        state_scope=str(BANK_BUILD["state_scope"]),
        # norm args are read from payload["normalization"] first; provide fallbacks.
        scale_half_life=0.0,
        scale_floor=1e-4,
        center_mode="zero",
        drift_feature_mode="none",
        iv_transform="bounded_logit",
        iv_lower_bound=1e-4,
        iv_upper_bound=1.0,
        positive_level_policy="reference_based",
    )


def build_windows(checkpoint: str, payload: dict[str, Any]) -> dict[str, np.ndarray]:
    """Canonical SNI windows for the bank's train split, WITH future_level/future_norm.

    Mirrors build_val_block but additionally recovers future_level/future_norm by calling
    select_normalized_innovation_scope on the same block with the checkpoint's norm config.
    """
    args = _bank_build_namespace(checkpoint)
    (
        history_level,
        history_norm,
        center,
        scale,
        drift_feature,
        history_raw,
        _specs,
        block,
    ) = build_val_block(args, payload)

    norm_cfg = payload.get("normalization", {})
    scale_half_life = norm_cfg.get("scale_half_life", args.scale_half_life)
    if scale_half_life is not None and float(scale_half_life) <= 0.0:
        scale_half_life = None
    scale_floor = float(norm_cfg.get("scale_floor", args.scale_floor))
    center_mode = norm_cfg.get("center_mode", args.center_mode)
    drift_feature_mode = norm_cfg.get("drift_feature_mode", args.drift_feature_mode)

    (
        history_level2,
        history_norm2,
        future_level,
        future_norm,
        center2,
        scale2,
        drift_feature2,
        _history_raw2,
        _specs2,
    ) = select_normalized_innovation_scope(
        block,
        payload.get("state_scope", args.state_scope),
        int(args.iv_count),
        scale_half_life=scale_half_life,
        scale_floor=scale_floor,
        center_mode=center_mode,
        drift_feature_mode=drift_feature_mode,
    )
    # Internal consistency: the re-derivation must reproduce build_val_block exactly.
    assert np.allclose(history_level, history_level2, atol=1e-5), "history_level re-derivation drift"
    assert np.allclose(history_norm, history_norm2, atol=1e-5), "history_norm re-derivation drift"

    # Units sanity on future_norm (rebuts the future_delta/scale failure mode whose std was
    # ~9740, absmax ~5e5). Normalized innovations are ~O(1) with a thin extreme tail: assert
    # finite, near-zero mean, std in a sane band, and that the bulk (q99) stays O(10).
    if not np.isfinite(future_norm).all():
        raise AssertionError("future_norm contains non-finite values -> normalization broke")
    fn_std = float(future_norm.std())
    fn_mean = float(np.abs(future_norm.mean()))
    fn_q99 = float(np.quantile(np.abs(future_norm), 0.99))
    if not (0.3 < fn_std < 10.0 and fn_mean < 0.5 and fn_q99 < 50.0):
        raise AssertionError(
            f"future_norm units look wrong: std={fn_std:.3f} |mean|={fn_mean:.3f} "
            f"q99={fn_q99:.3f} (expected O(1) normalized innovations) -> STOP"
        )

    source_index = np.asarray(block.indices, dtype=np.int64)
    return {
        "history_level": history_level.astype(np.float32),
        "history_norm": history_norm.astype(np.float32),
        "future_level": future_level.astype(np.float32),
        "future_norm": future_norm.astype(np.float32),
        "center": center.astype(np.float32),
        "scale": scale.astype(np.float32),
        "drift_feature": drift_feature.astype(np.float32),
        "source_index": source_index,
    }


def assert_bank_alignment(windows: dict[str, np.ndarray], bank_path: str) -> int:
    """Assert our windows match the support bank row-for-row (proves text<->window join)."""
    bank = np.load(bank_path)
    n = min(int(windows["history_level"].shape[0]), int(bank["history_level"].shape[0]))
    for key in ("history_level", "history_norm", "center", "scale", "drift_feature"):
        diff = float(np.abs(windows[key][:n] - bank[key][:n]).max())
        if diff > 1e-4:
            raise AssertionError(
                f"window/bank mismatch on {key!r}: max|diff|={diff} (bank built with "
                "different params -> STOP and report)"
            )
    return n


def load_window_texts(manifest_path: str, n_windows: int) -> dict[int, str]:
    """Map bank local window index -> a grounded narrative (first positive example seen)."""
    texts: dict[int, str] = {}
    with open(manifest_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("role") != "positive":
                continue
            idx = int(row["label_window_index"])
            if idx < 0 or idx >= n_windows:
                continue
            text = str(row.get("text", "")).strip()
            if text and idx not in texts:
                texts[idx] = text
    return texts


def _hash_embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic unit-norm pseudo-embedding (NO API). For the offline smoke only."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def build_embeddings(
    texts: dict[int, str], embed_mode: str, dotenv_path: str
) -> dict[int, np.ndarray]:
    """Embed each window's narrative. hash => offline deterministic; openai => API+cache."""
    if not texts:
        return {}
    if embed_mode == "hash":
        return {idx: _hash_embed(text) for idx, text in texts.items()}
    if embed_mode == "openai":
        from experiments.backfill.block_ar.nl_text_conditioning import (
            embed_texts_with_openai,
        )

        ordered = sorted(texts.items())
        vectors = embed_texts_with_openai(
            [text for _idx, text in ordered], dotenv_path=dotenv_path
        )
        return {idx: vectors[row] for row, (idx, _text) in enumerate(ordered)}
    raise ValueError(f"unknown embed_mode {embed_mode!r}")


def training_window_mask(source_index: np.ndarray) -> np.ndarray:
    """Train mask = ~val & ~purge, mirroring the calibration's build_masks EXACTLY.

    nl_locality_soft_fit_gate_calibration_t2.py:87-100 builds, over window indices
    [0, bank_size): val[lo:hi]=True; purge[max(0,lo-30):min(N,hi+30)]=True then
    purge &= ~val; train = ~val & ~purge. source_index == window index here (verified
    0..4009 stride 1), so apply directly on the window row -- do NOT reinvent the rule.
    """
    if not np.array_equal(source_index, np.arange(source_index.shape[0])):
        raise AssertionError(
            "source_index is not 0..N-1 stride-1; the calibration mask assumes window "
            "index == source_index -> STOP and reconcile index spaces"
        )
    n = int(source_index.shape[0])
    val = np.zeros(n, dtype=bool)
    for lo, hi in VAL_RANGES:
        val[int(lo) : int(hi)] = True
    purge = np.zeros(n, dtype=bool)
    for lo, hi in VAL_RANGES:
        purge[max(0, int(lo) - PURGE_GAP) : min(n, int(hi) + PURGE_GAP)] = True
    purge = purge & ~val
    return ~val & ~purge


# --- B2 unfreeze modes -------------------------------------------------------------------
# B1 (default, "none"): only the narrative_adapter trains; the entire 734a backbone is frozen
#   and asserted bit-identical after training. This path is UNCHANGED from B1.
#
# B2 gives the backbone LIMITED adaptation capacity. Two labeled variants (see RESEARCH_LOG
# 2026-06-18 B2 entry + advisor note):
#   "velocity_readout" (INTENT-FAITHFUL B2): unfreeze the velocity readout that actually
#       CONSUMES memory_state (which carries the additive narrative_context). The narrative
#       reaches the velocity ONLY via velocity.memory_proj -> mixer -> out
#       (causal_future_memory_transition_flow_matching.py:68-88). Unfreezing
#       velocity.memory_proj + velocity.mixer.layers[-1] + velocity.out gives the model
#       capacity to LISTEN to the injected narrative -- the hypothesis the task states.
#   "encoder_last_block" (LITERAL CONTROL): unfreeze model.memory.layers[-1] (the last
#       backbone TransformerEncoder block). NOTE: this block runs in _encode_prefix BEFORE
#       narrative_context is added (line 1026), so it is UPSTREAM of the injection and cannot
#       raise narrative sensitivity by construction. Kept as a labeled control / the literal
#       reading of "last backbone block"; a KILL here is near-guaranteed and is NOT evidence
#       about the directional ceiling. Do not report it as the intent-faithful B2.
B2_UNFREEZE_MODES = ("none", "velocity_readout", "encoder_last_block")


def _unfreeze_param_names(
    model: GenericStateAwareNormalizedInnovationFlowMatching, unfreeze_mode: str
) -> list[str]:
    """Return the backbone parameter-name PREFIXES to unfreeze for a B2 mode (besides the
    adapter). Empty for B1 ('none')."""
    if unfreeze_mode == "none":
        return []
    if unfreeze_mode == "velocity_readout":
        # The narrative-consuming readout path: input projection that ingests memory_state,
        # the LAST token-mixer block, and the final output head.
        n_mix = len(model.velocity.mixer.layers)
        return [
            "velocity.memory_proj",
            f"velocity.mixer.layers.{n_mix - 1}.",
            "velocity.out",
        ]
    if unfreeze_mode == "encoder_last_block":
        n_enc = len(model.memory.layers)
        return [f"memory.layers.{n_enc - 1}."]
    raise ValueError(f"unknown unfreeze_mode {unfreeze_mode!r}")


def freeze_backbone(
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    unfreeze_mode: str = "none",
) -> dict[str, Any]:
    """Freeze everything except narrative_adapter (B1) plus, for B2, the selected backbone
    sub-block. Returns a dict with trainable/adapter/unfrozen-backbone param counts and the
    list of unfrozen backbone parameter names (for the param-group LR split + the guards).

    Safety property preserved in BOTH paths: trainable == adapter + unfrozen-backbone, asserted
    exactly, so nothing trains that we did not deliberately unfreeze (734a lineage protection).
    """
    if unfreeze_mode not in B2_UNFREEZE_MODES:
        raise ValueError(f"unfreeze_mode must be one of {B2_UNFREEZE_MODES}")
    prefixes = _unfreeze_param_names(model, unfreeze_mode)

    def _is_unfrozen_backbone(name: str) -> bool:
        return any(name.startswith(p) or (p in name) for p in prefixes)

    unfrozen_backbone_names: list[str] = []
    for name, param in model.named_parameters():
        is_adapter = name.startswith("narrative_adapter")
        is_unfrozen_backbone = (not is_adapter) and _is_unfrozen_backbone(name)
        param.requires_grad = bool(is_adapter or is_unfrozen_backbone)
        if is_unfrozen_backbone:
            unfrozen_backbone_names.append(name)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter_params = sum(p.numel() for p in model.narrative_adapter.parameters())
    unfrozen_backbone_params = sum(
        p.numel() for n, p in model.named_parameters() if n in set(unfrozen_backbone_names)
    )
    if trainable != adapter_params + unfrozen_backbone_params:
        raise AssertionError(
            f"trainable {trainable} != adapter {adapter_params} + unfrozen-backbone "
            f"{unfrozen_backbone_params} (an unintended tensor trains -> 734a lineage at "
            "risk -> STOP)"
        )
    if unfreeze_mode == "none" and unfrozen_backbone_params != 0:
        raise AssertionError("B1 path must unfreeze ZERO backbone params")
    if unfreeze_mode != "none" and unfrozen_backbone_params == 0:
        raise AssertionError(
            f"B2 mode {unfreeze_mode!r} unfroze no backbone params (prefix match failed) -> STOP"
        )
    return {
        "trainable": int(trainable),
        "adapter_params": int(adapter_params),
        "unfrozen_backbone_params": int(unfrozen_backbone_params),
        "unfrozen_backbone_names": unfrozen_backbone_names,
        "unfreeze_mode": str(unfreeze_mode),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--support-bank", default=DEFAULT_SUPPORT_BANK)
    parser.add_argument("--manifest-examples", default=DEFAULT_MANIFEST_EXAMPLES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embed-mode", choices=["openai", "hash"], default="openai")
    parser.add_argument("--dotenv-path", default=".env")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="LR for the (zero-init) narrative_adapter; must move")
    parser.add_argument(
        "--unfreeze-mode",
        choices=list(B2_UNFREEZE_MODES),
        default="none",
        help="B1='none' (adapter only); B2='velocity_readout' (intent-faithful: unfreeze the "
             "narrative-consuming velocity readout) or 'encoder_last_block' (literal control, "
             "upstream of the narrative add -> confounded null).",
    )
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=1e-4,
        help="LR for the unfrozen PRETRAINED backbone block (B2 only). Lower than --lr to "
             "avoid catastrophic forgetting / blowing the fidelity guardrail.",
    )
    parser.add_argument("--null-fraction", type=float, default=0.25,
                        help="fraction of batch examples passed with narrative_present=False")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="if >0, stop after this many optimizer steps (smoke mode)")
    parser.add_argument("--max-windows", type=int, default=0,
                        help="if >0, subset training windows (smoke mode)")
    parser.add_argument("--seed", type=int, default=7344)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(
        args.device if (torch.cuda.is_available() or str(args.device) == "cpu") else "cpu"
    )

    # --- load 734a (read-only) and rebuild WITH the narrative channel on -----------------
    model, payload = load_model(
        args.checkpoint, device, cfg_overrides={"narrative_conditioning": True}
    )
    assert model.narrative_adapter is not None, "adapter must be instantiated"
    history_len = int(model.cfg.history_len)
    future_len = int(model.cfg.future_len)

    # --- freeze backbone (B1) or backbone-except-selected-block (B2); guard trainable count
    freeze_info = freeze_backbone(model, unfreeze_mode=str(args.unfreeze_mode))
    trainable = int(freeze_info["trainable"])
    unfrozen_backbone_names = list(freeze_info["unfrozen_backbone_names"])
    print(
        f"[freeze] mode={args.unfreeze_mode} trainable={trainable} "
        f"adapter={freeze_info['adapter_params']} "
        f"unfrozen_backbone={freeze_info['unfrozen_backbone_params']} "
        f"({len(unfrozen_backbone_names)} tensors)"
    )

    # --- canonical windows + bank alignment proof ----------------------------------------
    windows = build_windows(args.checkpoint, payload)
    n_aligned = assert_bank_alignment(windows, args.support_bank)
    n_windows = int(windows["history_level"].shape[0])

    # --- narrative texts + embeddings ----------------------------------------------------
    texts = load_window_texts(args.manifest_examples, n_windows)
    embeddings = build_embeddings(texts, args.embed_mode, args.dotenv_path)

    # --- training window mask (exclude val-frame-overlapping windows) --------------------
    keep = training_window_mask(windows["source_index"])
    train_rows = np.nonzero(keep)[0]
    if int(args.max_windows) > 0:
        train_rows = train_rows[: int(args.max_windows)]
    if train_rows.size == 0:
        raise RuntimeError("no training windows after held-out exclusion -> STOP")

    present_rows = np.array([r for r in train_rows if int(r) in embeddings], dtype=np.int64)
    present_frac = float(present_rows.size) / float(train_rows.size)
    print(
        f"[data] windows={n_windows} bank_aligned={n_aligned} train_windows={train_rows.size} "
        f"with_narrative={present_rows.size} ({present_frac:.1%}) embed_mode={args.embed_mode}"
    )
    if present_rows.size == 0:
        raise RuntimeError("no training windows carry a narrative embedding -> STOP")

    def stacked(key: str, rows: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(windows[key][rows]).to(device)

    def emb_for(rows: np.ndarray) -> np.ndarray:
        # rows without an embedding get a zero vector (and present=False below).
        out = np.zeros((rows.size, EMBED_DIM), dtype=np.float32)
        for i, r in enumerate(rows):
            if int(r) in embeddings:
                out[i] = embeddings[int(r)]
        return out

    # --- snapshot a KNOWN-FROZEN backbone tensor for the real bit-unchanged safety guard --
    # Must be a tensor that is NOT unfrozen in B2 (otherwise the guard would falsely pass on a
    # tensor we deliberately train). Pick the first backbone param that is neither the adapter
    # nor in the B2-unfrozen set. feature_proj.* always qualifies (always upstream + frozen).
    unfrozen_set = set(unfrozen_backbone_names)
    frozen_name = next(
        n for n, p in model.named_parameters()
        if (not n.startswith("narrative_adapter")) and (n not in unfrozen_set) and p.numel() > 0
    )
    frozen_ref = dict(model.named_parameters())[frozen_name].detach().clone()
    # In B2, also snapshot ONE unfrozen tensor so we can prove it actually changed (nonzero)
    # -- that nonzero diff is the task's reported backbone_max_diff for B2.
    unfrozen_probe_name = unfrozen_backbone_names[0] if unfrozen_backbone_names else None
    unfrozen_probe_ref = (
        dict(model.named_parameters())[unfrozen_probe_name].detach().clone()
        if unfrozen_probe_name is not None
        else None
    )

    # Two-group optimizer: zero-init adapter at --lr (must move); unfrozen PRETRAINED backbone
    # at the lower --lr-backbone (avoid catastrophic forgetting / blowing the fidelity gate).
    adapter_group = [p for n, p in model.named_parameters()
                     if p.requires_grad and n.startswith("narrative_adapter")]
    backbone_group = [p for n, p in model.named_parameters()
                      if p.requires_grad and n in unfrozen_set]
    param_groups = [{"params": adapter_group, "lr": float(args.lr)}]
    if backbone_group:
        param_groups.append({"params": backbone_group, "lr": float(args.lr_backbone)})
    optimizer = torch.optim.Adam(param_groups)
    print(
        f"[optim] adapter_group_lr={args.lr} backbone_group_lr="
        f"{args.lr_backbone if backbone_group else 'n/a'} "
        f"(adapter_tensors={len(adapter_group)} backbone_tensors={len(backbone_group)})"
    )
    rng = np.random.default_rng(int(args.seed))

    model.train()
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    stop = False
    for epoch in range(int(args.epochs)):
        order = rng.permutation(train_rows)
        epoch_losses: list[float] = []
        for start in range(0, order.size, int(args.batch_size)):
            batch_rows = order[start : start + int(args.batch_size)]
            emb_np = emb_for(batch_rows)
            # present = has embedding AND not chosen as a null-anchor example.
            present = np.array(
                [int(r) in embeddings for r in batch_rows], dtype=bool
            )
            null_draw = rng.random(batch_rows.size) < float(args.null_fraction)
            present = present & ~null_draw
            emb_t = torch.from_numpy(emb_np).to(device)
            present_t = torch.from_numpy(present).to(device)

            loss, _metrics = model.training_loss(
                stacked("history_level", batch_rows),
                stacked("history_norm", batch_rows),
                stacked("future_level", batch_rows),
                stacked("future_norm", batch_rows),
                stacked("center", batch_rows),
                stacked("scale", batch_rows),
                drift_feature=stacked("drift_feature", batch_rows),
                narrative_emb=emb_t,
                narrative_present=present_t,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            step += 1
            if int(args.max_steps) > 0 and step >= int(args.max_steps):
                stop = True
                break
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.append({"epoch": epoch, "mean_loss": mean_loss, "steps": step})
        print(f"[epoch {epoch}] mean_loss={mean_loss:.6f} steps={step}")
        if epoch_losses and mean_loss < best_loss:
            best_loss = mean_loss
            save_checkpoint(
                str(output_dir / "best_model.pt"),
                model,
                model.cfg,
                epoch=epoch,
                best_val=best_loss,
                extra={
                    "track": "B-direction",
                    "unfreeze_mode": str(args.unfreeze_mode),
                    "trainable_params": int(trainable),
                    "unfrozen_backbone_params": int(freeze_info["unfrozen_backbone_params"]),
                    "unfrozen_backbone_names": unfrozen_backbone_names,
                    "lr_adapter": float(args.lr),
                    "lr_backbone": float(args.lr_backbone) if backbone_group else None,
                    "embed_mode": str(args.embed_mode),
                    "train_window_count": int(train_rows.size),
                    "narrative_window_count": int(present_rows.size),
                    "base_checkpoint": str(args.checkpoint),
                },
            )
        if stop:
            break

    # --- HARD guard: a KNOWN-FROZEN backbone tensor must be bit-unchanged ----------------
    # This is the real 734a-lineage safety check: the frozen probe must NOT have moved.
    frozen_now = dict(model.named_parameters())[frozen_name].detach()
    frozen_max_diff = float((frozen_now - frozen_ref).abs().max())
    if frozen_max_diff != 0.0:
        raise AssertionError(
            f"FROZEN backbone tensor {frozen_name!r} changed (max|diff|={frozen_max_diff}) "
            "-> 734a lineage broken -> STOP"
        )
    # In B2, report the unfrozen probe's diff (EXPECTED nonzero; that's the whole point) and
    # assert it actually moved (a zero diff would mean the unfreeze silently did nothing).
    unfrozen_probe_max_diff: float | None = None
    if unfrozen_probe_ref is not None:
        probe_now = dict(model.named_parameters())[unfrozen_probe_name].detach()
        unfrozen_probe_max_diff = float((probe_now - unfrozen_probe_ref).abs().max())
        if unfrozen_probe_max_diff == 0.0:
            raise AssertionError(
                f"B2 unfrozen tensor {unfrozen_probe_name!r} did NOT change -> the unfreeze "
                "was a no-op -> STOP"
            )
    # The task's reported "backbone_max_diff vs 734a": 0.0 for B1, the unfrozen probe diff for B2.
    backbone_max_diff = (
        unfrozen_probe_max_diff if unfrozen_probe_max_diff is not None else frozen_max_diff
    )

    (output_dir / "training_history.json").write_text(
        json.dumps(
            {
                "history": history,
                "unfreeze_mode": str(args.unfreeze_mode),
                "trainable_params": int(trainable),
                "unfrozen_backbone_params": int(freeze_info["unfrozen_backbone_params"]),
                "unfrozen_backbone_names": unfrozen_backbone_names,
                "lr_adapter": float(args.lr),
                "lr_backbone": float(args.lr_backbone) if backbone_group else None,
                "frozen_probe_tensor": frozen_name,
                "frozen_probe_max_diff": frozen_max_diff,
                "unfrozen_probe_tensor": unfrozen_probe_name,
                "unfrozen_probe_max_diff": unfrozen_probe_max_diff,
                "backbone_max_diff": backbone_max_diff,
                "embed_mode": str(args.embed_mode),
                "train_window_count": int(train_rows.size),
                "narrative_window_count": int(present_rows.size),
                "bank_aligned_windows": int(n_aligned),
                "steps": int(step),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[done] mode={args.unfreeze_mode} steps={step} best_loss={best_loss:.6f} "
        f"frozen_probe_max_diff={frozen_max_diff} (={frozen_name}) "
        f"backbone_max_diff={backbone_max_diff} trainable_params={trainable}"
    )


if __name__ == "__main__":
    main()
