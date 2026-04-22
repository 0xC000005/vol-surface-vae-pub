#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys

sys.path.insert(0, ".")


def main() -> None:
    parser = argparse.ArgumentParser(description="279a-v0 residual hierarchical retrieval scenario generator")
    parser.add_argument("--base_checkpoint", type=str, default="models/backfill/277d_v0_s42/best_model.pt")
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--sample_temperature", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    out_payload = {
        "base_config": payload["config"],
        "model_state_dict": payload["model_state_dict"],
        "library_future_embeddings": payload["library_future_embeddings"],
        "library_last_level_01": payload["library_last_level_01"],
        "library_future_01": payload["library_future_01"],
        "top_k": args.top_k,
        "sample_temperature": args.sample_temperature,
        "source_checkpoint": args.base_checkpoint,
    }
    torch.save(out_payload, out_dir / "best_model.pt")
    torch.save(out_payload, out_dir / "final_model.pt")
    summary = {
        "source_checkpoint": args.base_checkpoint,
        "top_k": args.top_k,
        "sample_temperature": args.sample_temperature,
        "library_size": int(payload["library_future_embeddings"].shape[0]),
    }
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
