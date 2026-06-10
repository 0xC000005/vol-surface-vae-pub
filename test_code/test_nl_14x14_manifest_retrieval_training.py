import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar import nl_14x14_manifest_retrieval_training as train


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _tiny_manifest(tmp_path: Path) -> tuple[Path, Path, Path]:
    examples = [
        {
            "example_id": "joint39_train_0000__a__positive",
            "role": "positive",
            "view_name": "a",
            "text": "risk-off dollar squeeze with equities lower",
            "label_window_id": "joint39_train_0000",
            "label_window_index": 0,
            "target_window_id": "joint39_train_0000",
            "target_window_index": 0,
            "paired_negative_window_id": "joint39_train_0001",
        },
        {
            "example_id": "joint39_train_0000__a__hard_negative",
            "role": "hard_negative",
            "view_name": "a",
            "text": "risk-on commodity relief with equities higher",
            "label_window_id": "joint39_train_0001",
            "label_window_index": 1,
            "target_window_id": "joint39_train_0000",
            "target_window_index": 0,
            "paired_positive_window_id": "joint39_train_0000",
        },
        {
            "example_id": "joint39_train_0000__b__positive",
            "role": "positive",
            "view_name": "b",
            "text": "equity stress and dollar funding pressure",
            "label_window_id": "joint39_train_0000",
            "label_window_index": 0,
            "target_window_id": "joint39_train_0000",
            "target_window_index": 0,
            "paired_negative_window_id": "joint39_train_0001",
        },
        {
            "example_id": "joint39_train_0000__b__hard_negative",
            "role": "hard_negative",
            "view_name": "b",
            "text": "equity relief and commodity beta returning",
            "label_window_id": "joint39_train_0001",
            "label_window_index": 1,
            "target_window_id": "joint39_train_0000",
            "target_window_index": 0,
            "paired_positive_window_id": "joint39_train_0000",
        },
        {
            "example_id": "joint39_train_0002__a__positive",
            "role": "positive",
            "view_name": "a",
            "text": "rates selloff with duration pressure",
            "label_window_id": "joint39_train_0002",
            "label_window_index": 2,
            "target_window_id": "joint39_train_0002",
            "target_window_index": 2,
            "paired_negative_window_id": "joint39_train_0001",
        },
        {
            "example_id": "joint39_train_0002__a__hard_negative",
            "role": "hard_negative",
            "view_name": "a",
            "text": "duration bid with yields falling",
            "label_window_id": "joint39_train_0001",
            "label_window_index": 1,
            "target_window_id": "joint39_train_0002",
            "target_window_index": 2,
            "paired_positive_window_id": "joint39_train_0002",
        },
        {
            "example_id": "joint39_train_0002__b__positive",
            "role": "positive",
            "view_name": "b",
            "text": "higher yields press risk appetite",
            "label_window_id": "joint39_train_0002",
            "label_window_index": 2,
            "target_window_id": "joint39_train_0002",
            "target_window_index": 2,
            "paired_negative_window_id": "joint39_train_0001",
        },
        {
            "example_id": "joint39_train_0002__b__hard_negative",
            "role": "hard_negative",
            "view_name": "b",
            "text": "lower yields cushion equities",
            "label_window_id": "joint39_train_0001",
            "label_window_index": 1,
            "target_window_id": "joint39_train_0002",
            "target_window_index": 2,
            "paired_positive_window_id": "joint39_train_0002",
        },
    ]
    pairs = [
        {
            "pair_id": "joint39_train_0000__a",
            "target_window_id": "joint39_train_0000",
            "target_window_index": 0,
            "view_name": "a",
            "positive_example_id": "joint39_train_0000__a__positive",
            "negative_example_id": "joint39_train_0000__a__hard_negative",
            "negative_window_id": "joint39_train_0001",
            "negative_window_index": 1,
        },
        {
            "pair_id": "joint39_train_0002__a",
            "target_window_id": "joint39_train_0002",
            "target_window_index": 2,
            "view_name": "a",
            "positive_example_id": "joint39_train_0002__a__positive",
            "negative_example_id": "joint39_train_0002__a__hard_negative",
            "negative_window_id": "joint39_train_0001",
            "negative_window_index": 1,
        },
    ]
    examples_path = tmp_path / "training_examples.jsonl"
    pairs_path = tmp_path / "training_pairs.jsonl"
    arrays_path = tmp_path / "support_arrays.npz"
    _write_jsonl(examples_path, examples)
    _write_jsonl(pairs_path, pairs)
    memory_targets = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(arrays_path, memory_targets=memory_targets)
    return examples_path, pairs_path, arrays_path


def test_text_space_trainer_consumes_manifest_pairs(tmp_path: Path) -> None:
    examples_path, pairs_path, _arrays_path = _tiny_manifest(tmp_path)

    report = train.train_text_space_from_manifest(
        examples_jsonl=examples_path,
        pairs_jsonl=pairs_path,
        output_dir=tmp_path / "text_space",
        embedding_backend="hash",
        embedding_model="unit-hash",
        hash_dim=32,
        steps=4,
        batch_size=8,
        device="cpu",
    )

    assert report["status"] == "pass"
    assert report["method"] == "text_space_contrastive_14x14"
    assert report["training"]["pair_rows_consumed"] == 2
    assert report["training"]["example_count"] == 8
    assert "mean_pair_margin" in report["evaluation"]


def test_projected_memory_trainer_consumes_pair_rows_and_reports_margins(
    tmp_path: Path,
) -> None:
    examples_path, pairs_path, arrays_path = _tiny_manifest(tmp_path)

    report = train.train_projected_memory_from_manifest(
        examples_jsonl=examples_path,
        pairs_jsonl=pairs_path,
        support_arrays_path=arrays_path,
        output_dir=tmp_path / "projected",
        embedding_backend="hash",
        embedding_model="unit-hash",
        hash_dim=32,
        steps=5,
        batch_size=8,
        device="cpu",
    )

    assert report["status"] == "pass"
    assert report["method"] == "projected_memory_14x14"
    assert report["training"]["pair_rows_consumed"] == 2
    assert report["training"]["example_count"] == 8
    assert "target_cosine_mean" in report["evaluation"]
    assert "source_hard_negative_margin_mean" in report["evaluation"]
    assert "reciprocal_hard_negative_margin_mean" in report["evaluation"]


def test_combined_run_reuses_one_embedding_cache_for_both_methods(
    tmp_path: Path,
) -> None:
    examples_path, pairs_path, arrays_path = _tiny_manifest(tmp_path)

    report = train.run_training(
        Namespace(
            examples_jsonl=examples_path,
            pairs_jsonl=pairs_path,
            support_arrays=arrays_path,
            output_dir=tmp_path / "combined",
            method="both",
            embedding_backend="hash",
            embedding_model="unit-hash",
            dotenv_path=".env",
            embedding_batch_size=16,
            hash_dim=32,
            max_targets=None,
            steps=2,
            batch_size=8,
            adapter_dim=16,
            hidden_dim=16,
            lr=1e-3,
            pair_margin=0.15,
            seed=0,
            device="cpu",
        )
    )

    assert report["status"] == "pass"
    assert (
        report["text_space"]["embedding"]["cache_path"]
        == report["projected_memory"]["embedding"]["cache_path"]
    )


# --- 991a holdout / checkpoint / early-stopping upgrades (plan §0) ---

N_SPLIT_WINDOWS = 40
SPLIT_VAL_RANGES = "10:20"
SPLIT_PURGE_GAP = 5


def _split_zone(val_ranges: str = SPLIT_VAL_RANGES, purge_gap: int = SPLIT_PURGE_GAP):
    """Independent re-implementation of the split rule for expected counts."""
    val: set[int] = set()
    purge: set[int] = set()
    for chunk in val_ranges.split(","):
        lo, hi = (int(part) for part in chunk.split(":"))
        val.update(range(lo, hi))
        purge.update(range(max(0, lo - purge_gap), lo))
        purge.update(range(hi, hi + purge_gap))
    purge -= val
    return val, purge


def _split_manifest(tmp_path: Path) -> tuple[Path, Path, Path]:
    """40-window manifest: targets = even windows, views a/b, neg = (w+17)%40."""
    examples: list[dict] = []
    pairs: list[dict] = []
    for w in range(0, N_SPLIT_WINDOWS, 2):
        neg_w = (w + 17) % N_SPLIT_WINDOWS
        for view in ("a", "b"):
            pos_id = f"joint39_train_{w:04d}__{view}__positive"
            neg_id = f"joint39_train_{w:04d}__{view}__hard_negative"
            examples.append(
                {
                    "example_id": pos_id,
                    "role": "positive",
                    "view_name": view,
                    "text": f"positive window {w} view {view} regime text",
                    "label_window_id": f"joint39_train_{w:04d}",
                    "label_window_index": w,
                    "target_window_id": f"joint39_train_{w:04d}",
                    "target_window_index": w,
                }
            )
            examples.append(
                {
                    "example_id": neg_id,
                    "role": "hard_negative",
                    "view_name": view,
                    "text": f"negative window {neg_w} view {view} opposite regime",
                    "label_window_id": f"joint39_train_{neg_w:04d}",
                    "label_window_index": neg_w,
                    "target_window_id": f"joint39_train_{w:04d}",
                    "target_window_index": w,
                }
            )
            pairs.append(
                {
                    "pair_id": f"joint39_train_{w:04d}__{view}",
                    "target_window_id": f"joint39_train_{w:04d}",
                    "target_window_index": w,
                    "view_name": view,
                    "positive_example_id": pos_id,
                    "negative_example_id": neg_id,
                    "negative_window_id": f"joint39_train_{neg_w:04d}",
                    "negative_window_index": neg_w,
                }
            )
    examples_path = tmp_path / "training_examples.jsonl"
    pairs_path = tmp_path / "training_pairs.jsonl"
    arrays_path = tmp_path / "support_arrays.npz"
    _write_jsonl(examples_path, examples)
    _write_jsonl(pairs_path, pairs)
    memory_targets = np.eye(N_SPLIT_WINDOWS, 8, dtype=np.float32) + 0.01 * np.random.default_rng(
        0
    ).standard_normal((N_SPLIT_WINDOWS, 8)).astype(np.float32)
    np.savez_compressed(arrays_path, memory_targets=memory_targets)
    return examples_path, pairs_path, arrays_path


def test_window_holdout_excludes_val_and_purge_from_training(tmp_path: Path) -> None:
    examples_path, pairs_path, _ = _split_manifest(tmp_path)
    val, purge = _split_zone()
    excluded = val | purge

    report = train.train_text_space_from_manifest(
        examples_jsonl=examples_path,
        pairs_jsonl=pairs_path,
        output_dir=tmp_path / "text_space",
        embedding_backend="hash",
        embedding_model="unit-hash",
        hash_dim=32,
        steps=4,
        batch_size=8,
        device="cpu",
        val_window_ranges=SPLIT_VAL_RANGES,
        purge_gap=SPLIT_PURGE_GAP,
    )

    split_path = tmp_path / "text_space" / "holdout_split.json"
    assert split_path.exists()
    split = json.loads(split_path.read_text())

    rows = [json.loads(line) for line in examples_path.read_text().splitlines()]
    expected_train = sum(1 for r in rows if r["label_window_index"] not in excluded)
    expected_val = sum(1 for r in rows if r["label_window_index"] in val)
    assert split["train_example_count"] == expected_train
    assert split["val_example_count"] == expected_val
    assert split["purged_example_count"] == len(rows) - expected_train - expected_val
    # training consumed only the train slice
    assert report["training"]["example_count"] == expected_train
    # held-out metrics present
    assert "heldout_same_label_recall_at_10" in report["evaluation"]


def test_pairs_excluded_if_either_endpoint_in_val_or_purge(tmp_path: Path) -> None:
    examples_path, pairs_path, arrays_path = _split_manifest(tmp_path)
    val, purge = _split_zone()
    excluded = val | purge

    report = train.train_projected_memory_from_manifest(
        examples_jsonl=examples_path,
        pairs_jsonl=pairs_path,
        support_arrays_path=arrays_path,
        output_dir=tmp_path / "projected",
        embedding_backend="hash",
        embedding_model="unit-hash",
        hash_dim=32,
        steps=4,
        batch_size=8,
        device="cpu",
        val_window_ranges=SPLIT_VAL_RANGES,
        purge_gap=SPLIT_PURGE_GAP,
    )

    pair_rows = [json.loads(line) for line in pairs_path.read_text().splitlines()]
    expected_train_pairs = sum(
        1
        for r in pair_rows
        if r["target_window_index"] not in excluded
        and r["negative_window_index"] not in excluded
    )
    expected_val_pairs = sum(
        1 for r in pair_rows if r["target_window_index"] in val
    )
    split = json.loads((tmp_path / "projected" / "holdout_split.json").read_text())
    assert split["train_pair_count"] == expected_train_pairs
    assert split["val_pair_count"] == expected_val_pairs
    assert report["training"]["pair_rows_consumed"] == expected_train_pairs
    assert "heldout_true_memory_rank_median" in report["evaluation"]
    assert "heldout_recall_at_10_true_memory" in report["evaluation"]
    assert "heldout_source_hard_negative_margin_mean" in report["evaluation"]


def test_holdout_view_family_excluded_from_sampler_and_used_for_tierb(
    tmp_path: Path,
) -> None:
    examples_path, pairs_path, _ = _split_manifest(tmp_path)

    report = train.train_text_space_from_manifest(
        examples_jsonl=examples_path,
        pairs_jsonl=pairs_path,
        output_dir=tmp_path / "text_space",
        embedding_backend="hash",
        embedding_model="unit-hash",
        hash_dim=32,
        steps=4,
        batch_size=8,
        device="cpu",
        holdout_view_families="b",
    )

    rows = [json.loads(line) for line in examples_path.read_text().splitlines()]
    n_view_a = sum(1 for r in rows if r["view_name"] == "a")
    n_view_b = len(rows) - n_view_a
    split = json.loads((tmp_path / "text_space" / "holdout_split.json").read_text())
    assert split["train_example_count"] == n_view_a
    assert split["tierb_query_count"] == n_view_b
    assert report["training"]["example_count"] == n_view_a
    # view-a pairs only
    assert report["training"]["pair_rows_consumed"] == n_view_a // 2
    tierb = report["evaluation"]["heldout_view_recall"]
    assert "b" in tierb
    assert "recall_at_10" in tierb["b"]
    assert "heldout_view_same_label_recall_at_10" in report["evaluation"]


def test_checkpoints_saved_with_metadata(tmp_path: Path) -> None:
    import torch

    examples_path, pairs_path, arrays_path = _split_manifest(tmp_path)

    train.train_text_space_from_manifest(
        examples_jsonl=examples_path,
        pairs_jsonl=pairs_path,
        output_dir=tmp_path / "text_space",
        embedding_backend="hash",
        embedding_model="unit-hash",
        hash_dim=32,
        steps=6,
        batch_size=8,
        device="cpu",
        val_window_ranges=SPLIT_VAL_RANGES,
        purge_gap=0,
        eval_every=2,
        patience=10,
        argv_record=["--unit-test"],
    )
    train.train_projected_memory_from_manifest(
        examples_jsonl=examples_path,
        pairs_jsonl=pairs_path,
        support_arrays_path=arrays_path,
        output_dir=tmp_path / "projected",
        embedding_backend="hash",
        embedding_model="unit-hash",
        hash_dim=32,
        steps=6,
        batch_size=8,
        device="cpu",
        val_window_ranges=SPLIT_VAL_RANGES,
        purge_gap=0,
        eval_every=2,
        patience=10,
        argv_record=["--unit-test"],
    )

    for stem in (
        tmp_path / "text_space" / "text_space_adapter",
        tmp_path / "projected" / "projected_memory_bridge",
    ):
        for suffix in ("best", "final"):
            path = Path(f"{stem}_{suffix}.pt")
            assert path.exists(), path
            payload = torch.load(path, map_location="cpu", weights_only=False)
            for key in ("state_dict", "config", "seed", "lr", "argv", "best_step"):
                assert key in payload, (path, key)
            assert payload["argv"] == ["--unit-test"]


def test_validation_coupled_early_stopping(tmp_path: Path) -> None:
    examples_path, pairs_path, _ = _split_manifest(tmp_path)

    report = train.train_text_space_from_manifest(
        examples_jsonl=examples_path,
        pairs_jsonl=pairs_path,
        output_dir=tmp_path / "text_space",
        embedding_backend="hash",
        embedding_model="unit-hash",
        hash_dim=32,
        steps=500,
        batch_size=8,
        device="cpu",
        val_window_ranges=SPLIT_VAL_RANGES,
        purge_gap=0,
        eval_every=5,
        patience=2,
    )

    training = report["training"]
    assert training["early_stopped"] is True
    assert training["stopped_step"] < 500
    assert training["best_step"] is not None
    assert isinstance(training["eval_history"], list)
    assert len(training["eval_history"]) >= 3
    assert {"step", "metric"} <= set(training["eval_history"][0].keys())


def test_report_records_lr_seed_argv_and_dense_loss_trace(tmp_path: Path) -> None:
    examples_path, pairs_path, _ = _split_manifest(tmp_path)

    report = train.train_text_space_from_manifest(
        examples_jsonl=examples_path,
        pairs_jsonl=pairs_path,
        output_dir=tmp_path / "text_space",
        embedding_backend="hash",
        embedding_model="unit-hash",
        hash_dim=32,
        steps=120,
        batch_size=8,
        device="cpu",
        seed=7,
        lr=2e-3,
        argv_record=["--steps", "120"],
    )

    training = report["training"]
    assert training["seed"] == 7
    assert training["lr"] == 2e-3
    assert report["argv"] == ["--steps", "120"]
    # dense trace: every 50 steps plus first/last -> steps 1, 50, 100, 120
    steps_logged = [entry["step"] for entry in training["loss_trace"]]
    assert 50 in steps_logged and 100 in steps_logged and 120 in steps_logged


def test_cli_flags_parse() -> None:
    args = train.parse_args(
        [
            "--embedding-cache-dir",
            "some/cache/dir",
            "--val-window-ranges",
            "610:730,1490:1610",
            "--purge-gap",
            "30",
            "--holdout-view-families",
            "sparse_user_query,risk_manager_memo",
            "--eval-every",
            "250",
            "--patience",
            "8",
            "--steps",
            "5000",
        ]
    )
    assert str(args.embedding_cache_dir) == "some/cache/dir"
    assert args.val_window_ranges == "610:730,1490:1610"
    assert args.purge_gap == 30
    assert args.holdout_view_families == "sparse_user_query,risk_manager_memo"
    assert args.eval_every == 250
    assert args.patience == 8
    assert args.steps == 5000
