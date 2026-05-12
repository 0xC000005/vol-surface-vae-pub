"""Test whether explicit hard-negative wording improves bridge separation.

This is a label-quality TestFlight. It does not relabel market narratives. It
only rewrites selected artificial contrastive examples so the embedding model
sees them as controls rather than plausible descriptions of the same window.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_condition_bridge_evaluation import (  # noqa: E402
    build_bridge_examples,
    evaluate_condition_bridge,
    select_train_test_windows_from_report,
    summarize_bridge_metrics,
)
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    hash_text_embeddings,
    train_narrative_adapter,
)
from experiments.backfill.block_ar.nl_text_conditioning import (  # noqa: E402
    embed_texts_with_openai,
    normalize_rows,
)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def repair_negative_text(*, original_text: str, observed_fact_tokens: str) -> str:
    """Wrap an artificial negative so its contrastive role is explicit."""

    return (
        "HARD_NEGATIVE_CONTROL: this is intentionally not the observed market "
        "window. It is a contrastive opposite or near-opposite control used "
        "only for representation learning.\n"
        f"OBSERVED_WINDOW_FACTS_TO_AVOID: {observed_fact_tokens}\n"
        f"CONTRASTIVE_CONTROL_TEXT: {original_text}"
    )


def select_repair_indices(
    examples: list[dict[str, Any]],
    repair_window_ids: set[str],
    *,
    repair_all_negatives: bool = False,
) -> list[int]:
    selected: list[int] = []
    for idx, example in enumerate(examples):
        if str(example.get("role")) != "negative":
            continue
        if repair_all_negatives or str(example.get("window_id")) in repair_window_ids:
            selected.append(idx)
    return selected


def _observed_fact_tokens_by_window(
    examples: list[dict[str, Any]],
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for example in examples:
        if str(example.get("role")) != "anchor":
            continue
        window_id = str(example.get("window_id", ""))
        text = str(example.get("text", ""))
        marker = "MARKET_IMPLICATIONS:"
        if marker in text:
            observed[window_id] = text.split(marker, 1)[1].split("\n", 1)[0].strip()
        else:
            observed[window_id] = text
    return observed


def _load_artifacts(
    report_path: str | Path, arrays_path: str | Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    with np.load(arrays_path) as data:
        arrays = {name: data[name] for name in data.files}
    return report, arrays


def _embed_repaired_texts(
    texts: list[str],
    *,
    embedding_backend: str,
    embedding_model: str,
    dotenv_path: str,
    batch_size: int,
    hash_dim: int,
) -> np.ndarray:
    if embedding_backend == "openai":
        return embed_texts_with_openai(
            texts,
            model=embedding_model,
            dotenv_path=dotenv_path,
            batch_size=batch_size,
        )
    if embedding_backend == "hash":
        return hash_text_embeddings(texts, dim=hash_dim)
    raise ValueError(f"unknown embedding backend: {embedding_backend}")


def run_repair_testflight(args: argparse.Namespace) -> dict[str, Any]:
    report, arrays = _load_artifacts(args.input_report, args.input_npz)
    examples = build_bridge_examples(report)
    embeddings = np.asarray(arrays["text_embeddings"], dtype=np.float32).copy()
    memory_targets = np.asarray(arrays["memory_targets"], dtype=np.float32)
    repair_ids = {
        item.strip() for item in str(args.repair_window_ids).split(",") if item.strip()
    }
    repair_indices = select_repair_indices(
        examples,
        repair_ids,
        repair_all_negatives=bool(args.repair_all_negatives),
    )
    observed_by_window = _observed_fact_tokens_by_window(examples)
    repaired_texts = [
        repair_negative_text(
            original_text=str(examples[idx].get("text", "")),
            observed_fact_tokens=observed_by_window.get(
                str(examples[idx].get("window_id", "")), ""
            ),
        )
        for idx in repair_indices
    ]
    if repaired_texts:
        repaired_embeddings = _embed_repaired_texts(
            repaired_texts,
            embedding_backend=str(args.embedding_backend),
            embedding_model=str(args.embedding_model),
            dotenv_path=str(args.dotenv),
            batch_size=int(args.embedding_batch_size),
            hash_dim=int(embeddings.shape[1]),
        )
        embeddings[np.asarray(repair_indices, dtype=np.int64)] = normalize_rows(
            repaired_embeddings
        )

    split = select_train_test_windows_from_report(
        report,
        int(memory_targets.shape[0]),
        train_windows=int(args.train_windows),
        test_windows=int(args.test_windows),
    )
    train_set = set(split["train_indices"])
    train_example_indices = [
        idx
        for idx, example in enumerate(examples)
        if int(example["window_index"]) in train_set
    ]
    train_examples = [examples[idx] for idx in train_example_indices]
    target_indices = np.asarray(
        [
            -1 if example["target_index"] is None else int(example["target_index"])
            for example in train_examples
        ],
        dtype=np.int64,
    )
    train_result = train_narrative_adapter(
        embeddings[np.asarray(train_example_indices, dtype=np.int64)],
        memory_targets,
        target_indices,
        [str(example["role"]) for example in train_examples],
        [str(example["window_id"]) for example in train_examples],
        condition_dim=int(memory_targets.shape[1]),
        hidden_dim=args.hidden_dim,
        steps=int(args.adapter_steps),
        lr=float(args.adapter_lr),
        contrastive_weight=float(args.contrastive_weight),
        contrastive_margin=float(args.contrastive_margin),
        seed=int(args.seed),
        device=str(args.device),
    )
    adapter = train_result["adapter"]
    adapter.eval()
    with torch.no_grad():
        condition_vectors = (
            adapter(
                torch.from_numpy(normalize_rows(embeddings))
                .float()
                .to(torch.device(str(args.device)))
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    evaluation = evaluate_condition_bridge(
        examples,
        condition_vectors,
        memory_targets,
        train_indices=split["train_indices"],
        test_indices=split["test_indices"],
        top_k=int(args.top_k),
    )
    summary = summarize_bridge_metrics(evaluation)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "hard_negative_repair_arrays.npz",
        text_embeddings=normalize_rows(embeddings),
        condition_vectors=condition_vectors,
        memory_targets=memory_targets,
        repaired_embedding_indices=np.asarray(repair_indices, dtype=np.int64),
    )
    output = {
        "status": "ok",
        "scope_note": (
            "Hard-negative wording repair TestFlight. Market narratives and "
            "memory targets are unchanged; selected artificial negative texts "
            "are re-embedded to test whether contrastive wording is the bottleneck."
        ),
        "input_report": str(args.input_report),
        "input_npz": str(args.input_npz),
        "embedding_backend": str(args.embedding_backend),
        "embedding_model": (
            str(args.embedding_model) if args.embedding_backend == "openai" else None
        ),
        "repair_all_negatives": bool(args.repair_all_negatives),
        "repair_window_ids": sorted(repair_ids),
        "repaired_negative_count": len(repair_indices),
        "split": split,
        "adapter_training": {
            "loss_first": float(train_result["loss_first"]),
            "loss_last": float(train_result["loss_last"]),
            "steps": int(args.adapter_steps),
            "contrastive_weight": float(args.contrastive_weight),
            "contrastive_margin": float(args.contrastive_margin),
        },
        "summary": summary,
        "evaluation": evaluation,
        "artifact_paths": {
            "arrays": str(output_dir / "hard_negative_repair_arrays.npz"),
            "report": str(output_dir / "hard_negative_repair_report.json"),
        },
    }
    _write_json(output_dir / "hard_negative_repair_report.json", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-report", required=True)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repair-window-ids", default="")
    parser.add_argument("--repair-all-negatives", action="store_true")
    parser.add_argument(
        "--embedding-backend", choices=["openai", "hash"], default="openai"
    )
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--train-windows", type=int, default=40)
    parser.add_argument("--test-windows", type=int, default=10)
    parser.add_argument("--adapter-steps", type=int, default=700)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=0.25)
    parser.add_argument("--contrastive-margin", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=878)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    report = run_repair_testflight(args)
    print(
        json.dumps(
            {
                "report": report["artifact_paths"]["report"],
                "repaired_negative_count": report["repaired_negative_count"],
                "summary": report["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
