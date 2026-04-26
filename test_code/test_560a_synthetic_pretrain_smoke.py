import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.train_560a_synthetic_prior_pretrain import main


def test_560a_synthetic_pretrain_smoke_writes_compatible_checkpoint(tmp_path: Path) -> None:
    out_dir = tmp_path / "synthetic_pretrain"

    main(
        [
            "--output_dir",
            str(out_dir),
            "--device",
            "cpu",
            "--n_windows",
            "24",
            "--val_frac",
            "0.25",
            "--history_len",
            "6",
            "--future_len",
            "4",
            "--epochs",
            "1",
            "--batch_size",
            "6",
            "--hidden_dim",
            "16",
            "--memory_dim",
            "16",
            "--memory_layers",
            "1",
            "--memory_heads",
            "2",
            "--memory_ff",
            "32",
            "--model_hidden",
            "32",
            "--model_layers",
            "1",
            "--token_dim",
            "16",
            "--token_layers",
            "1",
            "--token_heads",
            "2",
            "--token_ff",
            "32",
            "--n_quantiles",
            "51",
            "--flow_steps",
            "2",
            "--seed",
            "560",
        ]
    )

    assert (out_dir / "args.json").exists()
    assert (out_dir / "training_history.json").exists()
    assert (out_dir / "best_model.pt").exists()
    assert (out_dir / "final_model.pt").exists()

    history = json.loads((out_dir / "training_history.json").read_text())
    assert len(history) == 1
    assert history[0]["epoch"] == 1
    assert history[0]["train_loss"] > 0.0
    assert history[0]["val_loss"] > 0.0

    payload = torch.load(out_dir / "best_model.pt", map_location="cpu", weights_only=False)
    assert payload["config"]["history_len"] == 6
    assert payload["config"]["future_len"] == 4
    assert payload["config"]["n_cells"] == 25
    assert "model_state_dict" in payload
