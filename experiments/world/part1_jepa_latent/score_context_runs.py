from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


DEFAULT_COMPOSITE_WEIGHTS = {
    "frame_mrr": 1.0,
    "rank_fraction": 0.5,
    "decorrelation": 0.25,
    "ridge_mrr": 1.0,
    "ridge_mse_improvement": 0.5,
}


def _float_at(obj: Mapping[str, object], path: tuple[str, ...]) -> float:
    current: object = obj
    for key in path:
        if not isinstance(current, Mapping):
            raise KeyError(".".join(path))
        current = current[key]
    return float(current)


def composite_part1_score(
    train_result: Mapping[str, object],
    audit_result: Mapping[str, object],
    *,
    weights: Mapping[str, float] | None = None,
) -> dict[str, float | dict[str, float]]:
    w = dict(DEFAULT_COMPOSITE_WEIGHTS if weights is None else weights)
    context_dim = _float_at(audit_result, ("config", "context_dim"))
    frame_mrr = _float_at(
        train_result,
        ("best_val_metrics", "overall_retrieval", "mrr_mean"),
    )
    effective_rank = _float_at(
        audit_result,
        ("val_context_health", "effective_rank"),
    )
    offdiag_abs_mean = _float_at(
        audit_result,
        ("val_context_health", "offdiag_abs_mean"),
    )
    ridge_mrr = _float_at(
        audit_result,
        ("ridge_probe_target_metrics", "overall_retrieval", "mrr_mean"),
    )
    ridge_mse = _float_at(
        audit_result,
        ("ridge_probe_target_metrics", "overall_prediction", "mse"),
    )
    zero_mse = _float_at(
        audit_result,
        ("zero_delta_target_baseline", "overall_prediction", "mse"),
    )
    rank_fraction = effective_rank / max(context_dim, 1.0)
    decorrelation = max(0.0, 1.0 - offdiag_abs_mean)
    ridge_mse_improvement = max(0.0, (zero_mse - ridge_mse) / max(zero_mse, 1e-12))
    components = {
        "frame_mrr": frame_mrr,
        "rank_fraction": rank_fraction,
        "decorrelation": decorrelation,
        "ridge_mrr": ridge_mrr,
        "ridge_mse_improvement": ridge_mse_improvement,
    }
    score = float(sum(float(w.get(name, 0.0)) * value for name, value in components.items()))
    return {
        "score": score,
        "components": components,
        "weights": {name: float(value) for name, value in w.items()},
    }


def score_entry(label: str, train_json: str | Path, audit_json: str | Path) -> dict[str, object]:
    train_result = json.loads(Path(train_json).read_text(encoding="utf-8"))
    audit_result = json.loads(Path(audit_json).read_text(encoding="utf-8"))
    score = composite_part1_score(train_result, audit_result)
    return {
        "label": label,
        "train_json": str(train_json),
        "audit_json": str(audit_json),
        **score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score saved Part 1 context-probe runs")
    parser.add_argument(
        "--entry",
        nargs=3,
        action="append",
        metavar=("LABEL", "TRAIN_JSON", "AUDIT_JSON"),
        required=True,
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/world/part1_context_composite_scores.json",
    )
    args = parser.parse_args()

    scored = [score_entry(label, train_json, audit_json) for label, train_json, audit_json in args.entry]
    scored.sort(key=lambda row: float(row["score"]), reverse=True)
    result = {
        "weights": DEFAULT_COMPOSITE_WEIGHTS,
        "ranked": scored,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
