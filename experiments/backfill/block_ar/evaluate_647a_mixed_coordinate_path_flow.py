#!/usr/bin/env python
"""647a: score direct mixed-coordinate path-flow checkpoints on the IV 11-suite."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")

from diffusion.block_ar.generic_mixed_coordinate_path_flow_matching import (
    load_model as load_singlehead_model,
)  # noqa: E402
from diffusion.block_ar.generic_multihead_mixed_coordinate_path_flow_matching import (
    load_model as load_multihead_model,
)  # noqa: E402
from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    build_rollout_windows,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    FixedDeployableSampler,
    history_key,
    run_suite,
    set_seed,
)
from experiments.backfill.block_ar.evaluate_629a_state_conditioned_increment_flow import (  # noqa: E402
    alignment_diagnostics,
    build_val_block,
    generate_iv_samples,
    summarize_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument(
        "--state_scope", choices=["iv_only", "joint38"], default="joint38"
    )
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument(
        "--clean_nonpositive_log_levels", action="store_true", default=True
    )
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=647)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    probe = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if probe.get("model_architecture") == "multihead_mixed_coordinate_path":
        model, payload = load_multihead_model(args.checkpoint, device)
    else:
        model, payload = load_singlehead_model(args.checkpoint, device)
    history_level, history_increment, history_raw, specs, block = build_val_block(
        args, payload
    )
    cfg = payload["config"]
    n_windows = min(int(args.max_windows), int(history_level.shape[0]))
    history_level = history_level[:n_windows]
    history_increment = history_increment[:n_windows]
    history_raw = history_raw[:n_windows]
    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=int(cfg["history_len"]),
        future_len=int(cfg["future_len"]),
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        max_windows=int(n_windows),
        device=device,
        split="val",
    )
    alignment = alignment_diagnostics(block, batch, n_windows)
    if (
        alignment["history_max_abs_error"] > 1e-6
        or alignment["future_max_abs_error"] > 1e-6
    ):
        raise RuntimeError(f"647a/full-suite alignment failed: {alignment}")

    print(f"Generating {args.samples} samples for {n_windows} windows")
    t0 = time.time()
    cond_samples = generate_iv_samples(
        model,
        history_level,
        history_increment,
        history_raw,
        specs,
        samples=int(args.samples),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        chunk_size=int(args.chunk_size),
        device=device,
        iv_count=int(args.iv_count),
        sample_temperature=float(args.sample_temperature),
    )
    generation_time = time.time() - t0
    hist_norm_np = batch.history_norm.detach().cpu().numpy()[:n_windows]
    samples_by_key = {
        history_key(hist_norm_np[i]): cond_samples[i] for i in range(n_windows)
    }
    fixed_model = FixedDeployableSampler(samples_by_key).eval()
    results = run_suite(
        cond_samples=cond_samples,
        batch=batch,
        model=fixed_model,
        data_path=args.data_path,
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        history_len=int(cfg["history_len"]),
        future_len=int(cfg["future_len"]),
        batch_size=int(args.batch_size),
        conditionality_samples=int(args.conditionality_samples),
        conditionality_max_batches=int(args.conditionality_max_batches),
        device=device,
    )
    results["config"] = {
        "mode": "647a_mixed_coordinate_path_flow",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_best_val": float(payload.get("best_val", float("nan"))),
        "checkpoint_config": cfg,
        "state_scope": payload.get("state_scope", args.state_scope),
        "model_coordinate": payload.get("model_coordinate", "mixed_coordinate_path"),
        "generated_coordinate_policy": payload.get("generated_coordinate_policy", {}),
        "n_windows": int(n_windows),
        "samples": int(args.samples),
        "n_steps": int(args.n_steps),
        "sample_temperature": float(args.sample_temperature),
        "generation_time_s": float(generation_time),
        "device": str(device),
        "seed": int(args.seed),
        "alignment": alignment,
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(make_serializable(results), indent=2), encoding="utf-8"
    )
    write_markdown_summary(
        out_md,
        "647a Mixed-Coordinate Path Flow Official IV 11-Suite",
        summarize_results(results, alignment),
    )
    print(json.dumps(make_serializable(results["summary"]), indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
