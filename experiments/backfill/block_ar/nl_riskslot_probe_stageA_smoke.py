"""Stage-A mechanics smoke for the risk_context oracle-injection probe.

Confirms (no GPU, no training):
  1. The frozen 734a loads into a risk_state_dim>0 model (strict=False; risk heads new).
  2. With the slot present but untrained, risk_context_proj is zero-init -> risk_context=0 ->
     sample() output MATCHES the original 734a (slot is benign).
  3. Oracle-injecting a non-zero risk_context (monkeypatch _risk_state_from_history) CHANGES
     the sampled output (the conditioning pathway is live).

Does NOT prove the trained slot gives channel-specific/realistic shifts past the null band
(that is Stage B fine-tune + Stage C eval). This only validates the mechanics + injection hook.
"""
from __future__ import annotations

import numpy as np
import torch

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (
    GenericStateAwareNormalizedInnovationFlowMatching as Model,
    GenericStateAwareNormalizedInnovationFMConfig as Cfg,
    load_model,
)

CKPT = "models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/best_model.pt"
ARRAYS = "experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
DEVICE = torch.device("cpu")
N_WIN, N_SAMP, SEED = 4, 8, 7


def _inputs():
    a = np.load(ARRAYS, allow_pickle=True)
    sl = slice(0, N_WIN)
    t = lambda x: torch.tensor(np.asarray(a[x][sl], dtype=np.float32), device=DEVICE)
    return t("history_level"), t("history_norm"), t("center"), t("scale"), t("drift_feature")


def _sample(model, hlv, hni, center, scale, drift):
    torch.manual_seed(SEED)
    with torch.no_grad():
        try:
            return model.sample_batched(hlv, hni, center, scale, n_samples=N_SAMP, n_steps=30,
                                        drift_feature=drift)
        except TypeError:
            return model.sample_batched(hlv, hni, center, scale, n_samples=N_SAMP, n_steps=30)


def main() -> None:
    hlv, hni, center, scale, drift = _inputs()

    # 1. original 734a
    orig, payload = load_model(CKPT, DEVICE)
    orig.eval()
    out_orig = _sample(orig, hlv, hni, center, scale, drift)

    # 2. slot-on model: risk_state_dim>0 + conditional_base_noise_scale, load 734a strict=False
    cfg = Cfg(**{**payload["config"], "risk_state_dim": 16, "conditional_base_noise_scale": True})
    slot = Model(cfg).to(DEVICE)
    incompat = slot.load_state_dict(payload["model_state_dict"], strict=False)
    missing = [k for k in incompat.missing_keys]
    unexpected = list(incompat.unexpected_keys)
    slot.eval()
    out_slot = _sample(slot, hlv, hni, center, scale, drift)

    benign_diff = float(torch.mean(torch.abs(out_slot - out_orig)))

    # 3. oracle injection: force an extreme risk_context (memory_dim) added to memory_states
    mdim = int(cfg.memory_dim)
    direction = torch.zeros(1, mdim, device=DEVICE)
    direction[0, : mdim // 2] = 1.0
    direction[0, mdim // 2:] = -1.0
    inj = 6.0 * direction / direction.norm()

    def _injected(*_a, **_k):
        return None, inj  # (predicted unused in sample; context broadcasts over batch)

    slot._risk_state_from_history = _injected  # type: ignore[assignment]
    out_inj = _sample(slot, hlv, hni, center, scale, drift)
    inject_diff = float(torch.mean(torch.abs(out_inj - out_slot)))

    print("=== Stage-A risk_context mechanics smoke ===")
    print(f"missing keys (expect only risk_state_head/risk_context_proj[/cond noise]): {sorted(missing)}")
    print(f"unexpected keys (expect none): {unexpected}")
    print(f"out shapes: orig={tuple(out_orig.shape)} slot={tuple(out_slot.shape)} inj={tuple(out_inj.shape)}")
    print(f"[2] benign-at-zero mean|slot-orig| = {benign_diff:.3e}  (want ~0)")
    print(f"[3] injection    mean|inj-slot|  = {inject_diff:.3e}  (want >> benign)")
    only_risk = all(("risk_state_head" in k or "risk_context_proj" in k or "base_noise_scale" in k.lower()) for k in missing)
    verdict = (not unexpected) and only_risk and benign_diff < 1e-5 and inject_diff > 100 * max(benign_diff, 1e-9)
    print(f"VERDICT: {'PASS' if verdict else 'CHECK'} — "
          f"{'mechanics OK (benign at zero, injection steers output)' if verdict else 'inspect numbers above'}")


if __name__ == "__main__":
    main()
