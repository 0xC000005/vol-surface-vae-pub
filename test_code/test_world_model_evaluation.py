from __future__ import annotations

import hashlib
import json
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.evaluation.part1_metrics import (
    latent_prediction_metrics,
    representation_health_metrics,
    retrieval_metrics,
)
from experiments.world.evaluation.part2_metrics import compact_path_sample_metrics
from experiments.world.evaluation.world_data import build_iv_world_windows
from experiments.world.evaluation.masked_multiview_data import (
    build_geometry_token_metadata,
    build_masked_multiview_batch,
    load_geometry_panel_values,
)
from experiments.world.evaluation.masked_multiview_metrics import (
    barlow_cross_correlation_metrics,
    flattened_time_rows,
    mask_visibility_summary,
    same_state_multiview_metrics,
)
from experiments.world.part1_jepa_latent.masked_multiview_jepa_smoke import (
    MaskedMultiviewJEPAConfig,
    MaskedMultiviewJEPAWorldModel,
    make_masked_view_features,
    masked_multiview_jepa_loss,
    torch_barlow_cross_correlation_loss,
)
from experiments.world.part1_jepa_latent.masked_multiview_barlow_smoke import (
    DirectMaskedMultiviewBarlowConfig,
    DirectMaskedMultiviewBarlowModel,
    direct_masked_multiview_barlow_loss,
)
from experiments.world.part1_jepa_latent.masked_multiview_barlow_probe_audit import (
    concatenate_feature_blocks,
    make_future_summary_targets,
    regression_metrics,
)
from experiments.world.part1_jepa_latent.masked_multiview_geometry_barlow_smoke import (
    GeometryAwareDirectBarlowConfig,
    GeometryAwareDirectBarlowModel,
    build_token_descriptor_matrix,
    geometry_masked_multiview_barlow_loss,
)
from experiments.world.part1_jepa_latent.jepa_smoke import (
    JEPAConfig,
    JEPAWorldModel,
    jepa_loss,
    retrieval_contrastive_loss,
    update_ema,
)
from experiments.world.part1_jepa_latent.horizon_jepa_smoke import (
    HorizonJEPAConfig,
    HorizonJEPAWorldModel,
    horizon_jepa_loss,
    select_horizon_delta_target,
    select_horizon_prefix,
    select_horizon_target,
)
from experiments.world.part1_jepa_latent.fixed_delta_pca_jepa import (
    fit_delta_pca_target,
    inverse_transform_delta_targets,
    make_horizon_delta_matrix,
    transform_delta_targets,
)
from experiments.world.part1_jepa_latent.target_encoder_distill import (
    DeltaTargetEncoder,
    TargetEncoderDistillConfig,
    evaluate_distilled_targets,
    target_encoder_distill_loss,
)
from experiments.world.part1_jepa_latent.frozen_target_jepa import (
    FrozenTargetJEPAConfig,
    FrozenTargetJEPAWorldModel,
    freeze_module,
    frozen_target_jepa_loss,
)
from experiments.world.part1_jepa_latent.direct_delta_pca_predictor import (
    DirectDeltaPCAConfig,
    DirectDeltaPCAPredictor,
    flatten_past_window,
)
from experiments.world.part1_jepa_latent.fused_context_delta_pca_predictor import (
    FusedContextDeltaPCAConfig,
    FusedContextDeltaPCAPredictor,
)
from experiments.world.part1_jepa_latent.fused_context_ema_jepa import (
    FusedContextEMAJEPAConfig,
    FusedContextEMAJEPAWorldModel,
    fused_context_ema_jepa_loss,
    horizon_delta_frames,
    horizon_delta_target_block,
    relative_to_last_observation,
)
from experiments.world.part1_jepa_latent.joint_path_pca_predictor import (
    FusedContextJointPathPCAConfig,
    FusedContextJointPathPCAPredictor,
    fit_joint_delta_path_pca_target,
    inverse_transform_joint_delta_path_targets,
    transform_joint_delta_path_targets,
)
from experiments.world.part1_jepa_latent.joint_path_pca_probe_audit import (
    build_arg_parser as build_joint_path_pca_probe_arg_parser,
)
from experiments.world.part1_jepa_latent.fused_context_probe_audit import (
    build_arg_parser as build_fused_context_probe_arg_parser,
    encode_fused_contexts,
)
from experiments.world.part1_jepa_latent.supervised_horizon_frame import (
    SupervisedHorizonConfig,
    SupervisedHorizonFrameModel,
    checkpoint_selection_score,
    context_correlation_loss,
    decode_horizon_prediction,
    epoch_checkpoint_path,
    make_horizon_frame_targets,
    save_epoch_checkpoint,
    soft_neighborhood_contrastive_loss,
    supervised_horizon_loss,
)
from experiments.world.part1_jepa_latent.context_probe_audit import (
    ridge_probe_metrics,
    ridge_probe_predict,
)
from experiments.world.part1_jepa_latent.score_context_runs import composite_part1_score
from experiments.world.part1_jepa_latent.score_masked_multiview_part1 import (
    extract_masked_multiview_scorecard_row,
    render_scorecard_markdown,
)
from experiments.world.part1_jepa_latent.masked_multiview_mask_artifact_audit import (
    classification_metrics,
    fit_predict_multiclass_ridge,
)
from experiments.world.part1_jepa_latent.masked_multiview_stratified_audit import (
    stratified_same_state_metrics,
)
from experiments.world.part1_jepa_latent.masked_multiview_downstream_probe_audit import (
    make_extended_future_targets,
)
from experiments.world.part1_jepa_latent.reference_package_check import (
    check_reference_package,
)


def _write_surface_npz(path, n_days: int = 20) -> np.ndarray:
    surface = np.arange(n_days * 25, dtype=np.float32).reshape(n_days, 5, 5)
    np.savez(
        path,
        surface=surface,
        ret=np.zeros(n_days, dtype=np.float32),
        price=np.ones(n_days, dtype=np.float32),
        slopes=np.zeros(n_days, dtype=np.float32),
        skews=np.zeros(n_days, dtype=np.float32),
        levels=np.ones(n_days, dtype=np.float32),
    )
    return surface


def _write_multi_factor_npz(path, n_days: int = 20) -> tuple[np.ndarray, np.ndarray]:
    level_columns = np.array(
        [
            "spx",
            "usdcad",
            "usdjpy",
            "dxy",
            "copper",
            "wheat",
            "crude_oil",
            "us2y",
            "us10y",
            "aaa_oas",
            "bbb_oas",
            "nikkei",
            "gold",
            "vix",
        ]
    )
    return_columns = np.array(
        [
            "spx_logret",
            "usdcad_logret",
            "usdjpy_logret",
            "dxy_logret",
            "copper_logret",
            "wheat_logret",
            "crude_oil_logret",
            "us2y_diff",
            "us10y_diff",
            "aaa_oas_diff",
            "bbb_oas_diff",
            "nikkei_logret",
            "gold_logret",
            "vix_logret",
        ]
    )
    levels = np.arange(n_days * len(level_columns), dtype=np.float32).reshape(
        n_days, len(level_columns)
    )
    returns = (1000.0 + levels).astype(np.float32)
    dates = np.arange(
        np.datetime64("2020-01-01"),
        np.datetime64("2020-01-01") + np.timedelta64(n_days, "D"),
    )
    np.savez(
        path,
        dates=dates,
        levels=levels,
        level_columns=level_columns,
        returns=returns,
        return_columns=return_columns,
    )
    return levels, returns


def _minimal_masked_multiview_artifact(metric_key: str) -> dict[str, object]:
    return {
        "literature_status": "supported_adjacent_direct_barlow_twins_for_same_state_masked_multiview",
        "loss_scaling": "canonical_mean_scaled_barlow",
        "val_metrics": {
            metric_key: {
                "alignment": {"mse": 0.25, "cosine_mean": 0.75},
                "retrieval": {
                    "mrr": 0.40,
                    "median_rank": 3.0,
                    "top1": 0.20,
                    "top5": 0.50,
                    "top10": 0.80,
                },
                "barlow": {
                    "diag_mean": 0.90,
                    "offdiag_abs_mean": 0.10,
                },
                "view_a_health": {
                    "variance_min": 0.01,
                    "variance_mean": 0.05,
                    "variance_max": 0.09,
                    "effective_rank": 12.0,
                    "participation_ratio": 8.0,
                    "offdiag_abs_mean": 0.15,
                    "singular_values": [4.0, 3.0, 2.0, 1.0],
                },
                "view_b_health": {
                    "variance_min": 0.02,
                    "variance_mean": 0.06,
                    "variance_max": 0.10,
                    "effective_rank": 11.0,
                    "participation_ratio": 7.0,
                    "offdiag_abs_mean": 0.16,
                    "singular_values": [5.0, 3.0, 1.0, 1.0],
                },
            },
            "visibility": {
                "overall": {
                    "observed_rate": 1.0,
                    "view_a_visible_rate": 0.91,
                    "view_b_visible_rate": 0.92,
                }
            },
        },
        "raw_val_baseline": {
            "retrieval": {"top1": 0.03, "top5": 0.19, "top10": 0.35}
        },
    }


def test_extract_masked_multiview_scorecard_row_handles_direct_barlow_artifact():
    row = extract_masked_multiview_scorecard_row(
        _minimal_masked_multiview_artifact("view_alignment"),
        artifact_path="results/world/masked_multiview_barlow_head070.json",
    )

    assert row["run"] == "HEAD070"
    assert row["metric_block"] == "view_alignment"
    assert row["objective_family"] == "masked_multiview_invariance"
    assert row["top1"] == pytest.approx(0.20)
    assert row["top5"] == pytest.approx(0.50)
    assert row["top10"] == pytest.approx(0.80)
    assert row["effective_rank_a"] == pytest.approx(12.0)
    assert row["effective_rank_b"] == pytest.approx(11.0)
    assert row["offdiag_abs_mean"] == pytest.approx(0.10)
    assert row["variance_min_a"] == pytest.approx(0.01)
    assert row["variance_max_b"] == pytest.approx(0.10)
    assert row["health_offdiag_abs_mean_a"] == pytest.approx(0.15)
    assert row["singular_top1_share_a"] == pytest.approx(0.4)
    assert row["singular_top4_share_b"] == pytest.approx(1.0)
    assert row["raw_top10"] == pytest.approx(0.35)


def test_extract_masked_multiview_scorecard_row_handles_ema_predictor_artifact():
    row = extract_masked_multiview_scorecard_row(
        _minimal_masked_multiview_artifact("predicted_target"),
        artifact_path="results/world/masked_multiview_jepa_head066.json",
    )

    assert row["run"] == "HEAD066"
    assert row["metric_block"] == "predicted_target"
    assert row["objective_family"] == "context_to_target_jepa"
    assert row["top1"] == pytest.approx(0.20)
    assert row["visible_rate_a"] == pytest.approx(0.91)


def test_render_scorecard_markdown_includes_reference_and_probe_rows():
    rows = [
        extract_masked_multiview_scorecard_row(
            _minimal_masked_multiview_artifact("predicted_target"),
            artifact_path="results/world/masked_multiview_jepa_head066.json",
        ),
        extract_masked_multiview_scorecard_row(
            _minimal_masked_multiview_artifact("view_alignment"),
            artifact_path="results/world/masked_multiview_barlow_head070.json",
        ),
    ]
    probe_summary = {
        "raw_surface_last": {
            "future_mean_delta": {"mse": 0.006, "r2": 0.53},
            "future_range": {"mse": 0.055, "r2": -2.97},
        },
        "barlow_clean_last": {
            "future_mean_delta": {"mse": 0.012, "r2": 0.16},
            "future_range": {"mse": 0.047, "r2": -2.43},
        },
    }

    text = render_scorecard_markdown(rows, probe_summary=probe_summary)

    assert "| HEAD070 |" in text
    assert "`masked_multiview_invariance`" in text
    assert "| run | family | block | top1 | top5 | top10 | eff rank A/B | sv top1 A/B | health offdiag A/B | raw top10 |" in text
    assert "| raw_surface_last |" in text
    assert "HEAD070 remains the Part 1 reference candidate" in text


def test_multiclass_ridge_predicts_linearly_separable_labels():
    train_x = np.asarray(
        [
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [-2.0, -2.0],
            [-3.0, -3.0],
        ],
        dtype=np.float32,
    )
    train_y = np.asarray(["a", "a", "b", "b", "c", "c"], dtype=object)
    val_x = np.asarray([[4.0, 0.0], [0.0, 4.0], [-4.0, -4.0]], dtype=np.float32)

    pred = fit_predict_multiclass_ridge(train_x, train_y, val_x, alpha=1e-3)

    assert pred.tolist() == ["a", "b", "c"]


def test_classification_metrics_include_majority_baseline_and_lift():
    truth = np.asarray(["a", "a", "b", "c"], dtype=object)
    pred = np.asarray(["a", "b", "b", "c"], dtype=object)

    metrics = classification_metrics(pred, truth)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["majority_accuracy"] == pytest.approx(0.50)
    assert metrics["accuracy_lift"] == pytest.approx(0.25)
    assert metrics["n_classes"] == 3


def test_stratified_same_state_metrics_groups_sequence_embeddings():
    view_a = np.asarray(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [[1.0, 1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, -1.0]],
        ],
        dtype=np.float32,
    )
    view_b = view_a.copy()
    labels = np.asarray(["surface", "surface", "time", "time"], dtype=object)

    rows = stratified_same_state_metrics(view_a, view_b, labels, min_windows=2)

    assert set(rows) == {"surface", "time"}
    assert rows["surface"]["n_windows"] == 2
    assert rows["surface"]["retrieval_top1"] == pytest.approx(1.0)
    assert rows["time"]["alignment_mse"] == pytest.approx(0.0)


def test_make_extended_future_targets_includes_tail_and_drawdown_tasks():
    past = np.asarray([[[1.0, 2.0], [2.0, 1.0]]], dtype=np.float32)
    future = np.asarray([[[3.0, 0.5], [1.0, 4.0], [4.0, 3.0]]], dtype=np.float32)
    labels = np.asarray([2], dtype=np.int64)

    targets = make_extended_future_targets(past, future, regime_labels=labels)

    np.testing.assert_allclose(
        targets["regression"]["future_terminal_delta"], [[2.0, 2.0]]
    )
    np.testing.assert_allclose(
        targets["regression"]["future_max_abs_step"], [[3.0, 3.5]]
    )
    np.testing.assert_allclose(targets["regression"]["future_drawdown"], [[2.0, 1.0]])
    np.testing.assert_array_equal(targets["classification"]["regime_label"], labels)


def test_check_reference_package_validates_reports_and_digest(tmp_path):
    report = tmp_path / "reports" / "head.md"
    report.parent.mkdir()
    report.write_text("# report\n", encoding="utf-8")
    artifact = tmp_path / "artifacts" / "checkpoint.pt"
    artifact.parent.mkdir()
    payload = b"reference"
    artifact.write_bytes(payload)
    manifest = tmp_path / "reference_manifest.json"
    manifest.write_text(
        json.dumps({"source_reports": ["reports/head.md"]}),
        encoding="utf-8",
    )
    digest = tmp_path / "reference_artifact_digests.json"
    digest.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "path": "artifacts/checkpoint.pt",
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = check_reference_package(
        root=tmp_path,
        manifest_path=manifest,
        digest_path=digest,
    )

    assert result["ok"]
    assert result["missing_reports"] == []
    assert result["artifact_mismatches"] == []
    assert result["checked_artifacts"] == 1


def test_build_iv_world_windows_uses_manifest_style_split(tmp_path):
    data_path = tmp_path / "surface.npz"
    surface = _write_surface_npz(data_path)
    regime_path = tmp_path / "regimes.npz"
    labels = np.arange(surface.shape[0] - 3 - 2 + 1, dtype=np.int32)
    np.savez(regime_path, labels=labels, history_len=3, future_len=2, n_regimes=5)

    train = build_iv_world_windows(
        data_path=data_path,
        split="train",
        history_len=3,
        future_len=2,
        test_start=16,
        val_size=3,
        max_windows=2,
        normalize=False,
        regime_path=regime_path,
    )
    val = build_iv_world_windows(
        data_path=data_path,
        split="val",
        history_len=3,
        future_len=2,
        test_start=16,
        val_size=3,
        normalize=False,
        regime_path=regime_path,
    )

    assert train.past_window.shape == (2, 3, 25)
    assert train.future_window.shape == (2, 2, 25)
    assert train.start_index.tolist() == [0, 1]
    np.testing.assert_array_equal(train.past_window[0, 0], surface[0].reshape(25))
    np.testing.assert_array_equal(train.future_window[0, 0], surface[3].reshape(25))
    assert train.regime_label.tolist() == [0, 1]

    assert val.past_window.shape == (3, 3, 25)
    assert val.future_window.shape == (3, 2, 25)
    assert val.start_index.tolist() == [8, 9, 10]
    assert val.regime_label.tolist() == [8, 9, 10]


def test_masked_multiview_batch_preserves_geometry_and_masks(tmp_path):
    surface_path = tmp_path / "surface.npz"
    factor_path = tmp_path / "factors.npz"
    _write_surface_npz(surface_path, n_days=24)
    _write_multi_factor_npz(factor_path, n_days=24)

    batch = build_masked_multiview_batch(
        surface_path=surface_path,
        multi_factor_path=factor_path,
        split="train",
        history_len=4,
        future_len=2,
        test_start=20,
        val_size=3,
        max_windows=2,
        normalize=False,
        seed=63,
        mask_families=("surface_maturity",),
    )

    assert batch.clean_values.shape == (2, 4, 58)
    assert batch.view_a_values.shape == batch.clean_values.shape
    assert batch.observed_mask.shape == batch.clean_values.shape
    assert batch.synthetic_mask_a.shape == batch.clean_values.shape
    assert batch.token_metadata.n_tokens == 58
    assert np.sum(batch.token_metadata.geometry_id == "iv_surface") == 25
    assert np.sum(batch.token_metadata.geometry_id == "vol_side_channel") == 5
    assert np.sum(batch.token_metadata.geometry_id == "factor_level") == 14
    assert np.sum(batch.token_metadata.geometry_id == "factor_return") == 14
    assert set(batch.mask_family_a.tolist()) == {"surface_maturity"}

    hidden = batch.observed_mask & ~batch.synthetic_mask_a
    assert hidden.sum() == 2 * 4 * 5
    np.testing.assert_array_equal(batch.view_a_values[hidden], np.zeros(hidden.sum()))
    visible = batch.observed_mask & batch.synthetic_mask_a
    np.testing.assert_allclose(batch.view_a_values[visible], batch.clean_values[visible])
    np.testing.assert_array_equal(batch.relative_index, np.arange(4))
    np.testing.assert_array_equal(batch.absolute_index[0], np.arange(4))
    assert batch.positive_index.shape == (2, 4)


def test_masked_multiview_keeps_real_missingness_separate(tmp_path):
    surface_path = tmp_path / "surface.npz"
    factor_path = tmp_path / "factors.npz"
    _write_surface_npz(surface_path, n_days=24)
    levels, returns = _write_multi_factor_npz(factor_path, n_days=24)
    levels[0, 0] = np.nan
    np.savez(
        factor_path,
        dates=np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-01") + np.timedelta64(24, "D"),
        ),
        levels=levels,
        level_columns=np.array(
            [
                "spx",
                "usdcad",
                "usdjpy",
                "dxy",
                "copper",
                "wheat",
                "crude_oil",
                "us2y",
                "us10y",
                "aaa_oas",
                "bbb_oas",
                "nikkei",
                "gold",
                "vix",
            ]
        ),
        returns=returns,
        return_columns=np.array(
            [
                "spx_logret",
                "usdcad_logret",
                "usdjpy_logret",
                "dxy_logret",
                "copper_logret",
                "wheat_logret",
                "crude_oil_logret",
                "us2y_diff",
                "us10y_diff",
                "aaa_oas_diff",
                "bbb_oas_diff",
                "nikkei_logret",
                "gold_logret",
                "vix_logret",
            ]
        ),
    )

    values, observed, meta = load_geometry_panel_values(
        surface_path=surface_path,
        multi_factor_path=factor_path,
        normalize=False,
    )
    spx_level = np.where((meta.geometry_id == "factor_level") & (meta.factor_id == "spx"))[0][0]
    assert not observed[0, spx_level]
    assert values[0, spx_level] == 0.0

    batch = build_masked_multiview_batch(
        surface_path=surface_path,
        multi_factor_path=factor_path,
        split="train",
        history_len=4,
        future_len=2,
        test_start=20,
        val_size=3,
        max_windows=1,
        normalize=False,
        seed=64,
        mask_families=("factor_family",),
    )
    assert not batch.observed_mask[0, 0, spx_level]
    factor_hidden = (
        np.isin(batch.token_metadata.geometry_id, ["factor_level", "factor_return"])[None, None, :]
        & ~batch.synthetic_mask_a
    )
    assert factor_hidden.any()


def test_masked_multiview_metrics_score_alignment_and_visibility(tmp_path):
    surface_path = tmp_path / "surface.npz"
    factor_path = tmp_path / "factors.npz"
    _write_surface_npz(surface_path, n_days=24)
    _write_multi_factor_npz(factor_path, n_days=24)
    batch = build_masked_multiview_batch(
        surface_path=surface_path,
        multi_factor_path=factor_path,
        split="train",
        history_len=4,
        future_len=2,
        test_start=20,
        val_size=3,
        max_windows=3,
        normalize=False,
        seed=65,
        mask_families=("surface_moneyness", "factor_family"),
    )
    clean_rows = flattened_time_rows(batch.clean_values)
    embeddings = np.eye(clean_rows.shape[0], dtype=np.float32)
    metrics = same_state_multiview_metrics(embeddings, embeddings.copy())
    visibility = mask_visibility_summary(batch)

    assert metrics["alignment"]["mse"] < 0.001
    assert metrics["retrieval"]["top1"] == 1.0
    assert metrics["barlow"]["diag_mean"] > 0.99
    assert visibility["overall"]["observed_rate"] == 1.0
    assert "iv_surface" in visibility["by_geometry"]
    assert "equity_risk" in visibility["by_family"]
    assert visibility["by_geometry"]["iv_surface"]["n_tokens"] == 25


def test_barlow_cross_correlation_penalizes_mismatched_views():
    x = np.eye(6, dtype=np.float32)
    y = x.copy()
    shuffled = x[[1, 0, 3, 2, 5, 4]]

    matching = barlow_cross_correlation_metrics(x, y)
    mismatched = barlow_cross_correlation_metrics(x, shuffled)

    assert matching["diag_loss"] < mismatched["diag_loss"]
    assert matching["diag_mean"] > mismatched["diag_mean"]


def test_masked_multiview_jepa_uses_mask_channels_and_scores_time_rows():
    import torch

    values = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    observed = torch.ones_like(values, dtype=torch.bool)
    synthetic = torch.ones_like(values, dtype=torch.bool)
    synthetic[:, 1, 2] = False
    view_values = torch.where(observed & synthetic, values, torch.zeros_like(values))

    features = make_masked_view_features(view_values, observed, synthetic)
    assert features.shape == (2, 3, 12)
    torch.testing.assert_close(features[..., :4], view_values)
    torch.testing.assert_close(features[..., 4:8], observed.float())
    torch.testing.assert_close(features[..., 8:], synthetic.float())

    cfg = MaskedMultiviewJEPAConfig(
        token_dim=4,
        input_dim=12,
        hidden_dim=8,
        latent_dim=4,
        predictor_hidden_dim=8,
        ema_decay=0.9,
    )
    model = MaskedMultiviewJEPAWorldModel(cfg)
    out = model(features, features)
    assert out["context"].shape == (2, 3, 4)
    assert out["predicted"].shape == (2, 3, 4)
    assert out["target"].shape == (2, 3, 4)

    loss, parts = masked_multiview_jepa_loss(out, barlow_weight=0.1)
    assert torch.isfinite(loss)
    assert set(parts) == {"alignment", "barlow", "barlow_diag_loss", "barlow_offdiag_loss", "loss"}


def test_direct_masked_multiview_barlow_scores_encoder_embeddings():
    import torch

    values = torch.randn(3, 4, 5)
    observed = torch.ones_like(values, dtype=torch.bool)
    synth_a = torch.ones_like(values, dtype=torch.bool)
    synth_b = torch.ones_like(values, dtype=torch.bool)
    synth_a[:, :, 1] = False
    synth_b[:, 2:, 3] = False
    view_a = torch.where(observed & synth_a, values, torch.zeros_like(values))
    view_b = torch.where(observed & synth_b, values, torch.zeros_like(values))

    features_a = make_masked_view_features(view_a, observed, synth_a)
    features_b = make_masked_view_features(view_b, observed, synth_b)
    cfg = DirectMaskedMultiviewBarlowConfig(
        token_dim=5,
        input_dim=15,
        hidden_dim=8,
        latent_dim=6,
    )
    model = DirectMaskedMultiviewBarlowModel(cfg)
    out = model(features_a, features_b)

    assert out["view_a"].shape == (3, 4, 6)
    assert out["view_b"].shape == (3, 4, 6)
    loss, parts = direct_masked_multiview_barlow_loss(out)
    assert torch.isfinite(loss)
    assert set(parts) == {"barlow", "barlow_diag_loss", "barlow_offdiag_loss", "loss"}


def test_barlow_loss_supports_canonical_mean_scaled_offdiag():
    import torch

    torch.manual_seed(69)
    view_a = torch.randn(4, 3, 5)
    view_b = view_a.roll(shifts=1, dims=1) + 0.05 * torch.randn(4, 3, 5)

    current_loss, current_parts = torch_barlow_cross_correlation_loss(
        view_a,
        view_b,
        offdiag_weight=0.005,
    )
    canonical_loss, _canonical_parts = torch_barlow_cross_correlation_loss(
        view_a,
        view_b,
        offdiag_weight=0.005,
        canonical_mean_scale=True,
    )
    expected = current_parts["barlow_diag_loss"] + 0.005 * (view_a.shape[-1] - 1) * current_parts[
        "barlow_offdiag_loss"
    ]

    assert canonical_loss.item() > current_loss.item()
    assert canonical_loss.item() == pytest.approx(expected)


def test_masked_multiview_probe_targets_and_regression_metrics():
    past = np.array(
        [
            [[1.0, 2.0], [2.0, 3.0]],
            [[0.0, 1.0], [1.0, 1.5]],
        ],
        dtype=np.float32,
    )
    future = np.array(
        [
            [[3.0, 5.0], [5.0, 7.0], [4.0, 6.0]],
            [[2.0, 3.0], [3.0, 5.0], [4.0, 4.0]],
        ],
        dtype=np.float32,
    )

    targets = make_future_summary_targets(past, future)
    assert set(targets) == {"future_mean_delta", "future_range"}
    np.testing.assert_allclose(targets["future_mean_delta"][0], [2.0, 3.0])
    np.testing.assert_allclose(targets["future_range"][1], [2.0, 2.0])

    perfect = regression_metrics(targets["future_mean_delta"], targets["future_mean_delta"])
    shifted = regression_metrics(
        targets["future_mean_delta"] + 1.0,
        targets["future_mean_delta"],
    )
    assert perfect["mse"] == pytest.approx(0.0)
    assert perfect["r2"] == pytest.approx(1.0)
    assert shifted["mse"] > perfect["mse"]
    assert shifted["r2"] < perfect["r2"]


def test_probe_feature_block_concatenation_checks_rows():
    left = np.ones((3, 2), dtype=np.float32)
    right = 2.0 * np.ones((3, 4), dtype=np.float32)
    combined = concatenate_feature_blocks(left, right)

    assert combined.shape == (3, 6)
    np.testing.assert_allclose(combined[:, :2], left)
    np.testing.assert_allclose(combined[:, 2:], right)
    with pytest.raises(ValueError, match="same row count"):
        concatenate_feature_blocks(left, right[:2])


def test_geometry_aware_barlow_encoder_uses_token_descriptors():
    import torch

    metadata = build_geometry_token_metadata(
        level_columns=["spx"],
        return_columns=["spx_logret"],
    )
    descriptors = build_token_descriptor_matrix(metadata)
    assert descriptors.shape[0] == metadata.n_tokens
    assert descriptors.shape[1] > 4
    assert np.isfinite(descriptors).all()

    cfg = GeometryAwareDirectBarlowConfig(
        n_tokens=metadata.n_tokens,
        token_descriptor_dim=descriptors.shape[1],
        token_hidden_dim=8,
        hidden_dim=12,
        latent_dim=6,
    )
    model = GeometryAwareDirectBarlowModel(cfg, token_descriptors=descriptors)
    values = torch.randn(2, 3, metadata.n_tokens)
    observed = torch.ones_like(values, dtype=torch.bool)
    synth_a = torch.ones_like(values, dtype=torch.bool)
    synth_b = torch.ones_like(values, dtype=torch.bool)
    synth_a[:, :, 0] = False
    synth_b[:, 1:, -1] = False
    view_a = torch.where(observed & synth_a, values, torch.zeros_like(values))
    view_b = torch.where(observed & synth_b, values, torch.zeros_like(values))

    out = model(view_a, view_b, observed, synth_a, synth_b)
    assert out["view_a"].shape == (2, 3, 6)
    assert out["view_b"].shape == (2, 3, 6)
    loss, parts = geometry_masked_multiview_barlow_loss(out)
    assert torch.isfinite(loss)
    assert set(parts) == {"barlow", "barlow_diag_loss", "barlow_offdiag_loss", "loss"}


def test_part1_metrics_detect_prediction_retrieval_and_rank():
    target = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    predicted = target + np.array(
        [
            [0.01, 0.00, 0.00],
            [0.00, 0.02, 0.00],
            [0.00, 0.00, 0.03],
            [0.01, 0.01, 0.00],
        ],
        dtype=np.float32,
    )

    pred_metrics = latent_prediction_metrics(predicted, target)
    retrieval = retrieval_metrics(predicted, target, top_k=(1, 2))
    health = representation_health_metrics(target)

    assert pred_metrics["mse"] < 0.001
    assert pred_metrics["cosine_mean"] > 0.99
    assert retrieval["top1"] == 1.0
    assert retrieval["mrr"] == 1.0
    assert health["effective_rank"] > 2.0
    assert health["variance_min"] > 0.0


def test_compact_path_sample_metrics_score_decoder_samples():
    truth = np.array(
        [
            [[0.10, 0.20], [0.11, 0.22], [0.12, 0.23]],
            [[0.30, 0.40], [0.33, 0.39], [0.35, 0.38]],
        ],
        dtype=np.float32,
    )
    offsets = np.array([-0.02, 0.0, 0.02], dtype=np.float32)
    samples = truth[:, None, :, :] + offsets[None, :, None, None]

    metrics = compact_path_sample_metrics(samples, truth)

    assert 0.0 <= metrics["coverage_90"] <= 1.0
    assert metrics["pairwise_distance_mean"] > 0.0
    assert metrics["variance_ratio"] > 0.0
    assert metrics["effective_rank"] > 0.0
    assert metrics["corr_frobenius"] >= 0.0


def test_jepa_smoke_model_forward_loss_and_ema_update():
    import torch

    torch.manual_seed(7)
    cfg = JEPAConfig(input_dim=5, hidden_dim=8, latent_dim=4, predictor_hidden_dim=8)
    model = JEPAWorldModel(cfg)
    past = torch.randn(6, 3, 5)
    future = torch.randn(6, 3, 5)

    out = model(past, future)
    loss, parts = jepa_loss(
        out,
        variance_weight=0.1,
        covariance_weight=0.1,
        retrieval_weight=0.1,
    )
    before = [p.detach().clone() for p in model.target_encoder.parameters()]
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(1.0)
    update_ema(model.context_encoder, model.target_encoder, decay=0.5)
    after = list(model.target_encoder.parameters())

    assert out["context"].shape == (6, 4)
    assert out["target"].shape == (6, 4)
    assert out["predicted"].shape == (6, 4)
    assert torch.isfinite(loss)
    assert parts["prediction"] >= 0.0
    assert parts["retrieval"] > 0.0
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_retrieval_contrastive_loss_prefers_matching_pairs():
    import torch

    predicted = torch.eye(4)
    target = torch.eye(4)
    shuffled = target[[1, 0, 3, 2]]

    matching = retrieval_contrastive_loss(predicted, target, temperature=0.1)
    mismatched = retrieval_contrastive_loss(predicted, shuffled, temperature=0.1)

    assert matching < mismatched


def test_horizon_jepa_selects_prefixes_and_scores_loss():
    import torch

    torch.manual_seed(11)
    cfg = HorizonJEPAConfig(
        input_dim=5,
        hidden_dim=8,
        latent_dim=4,
        predictor_hidden_dim=8,
        horizons=(1, 3),
    )
    model = HorizonJEPAWorldModel(cfg)
    past = torch.randn(6, 4, 5)
    future = torch.randn(6, 5, 5)

    prefix = select_horizon_prefix(future, horizon=3)
    frame = select_horizon_target(future, horizon=3, target_mode="frame")
    prefix_target = select_horizon_target(future, horizon=3, target_mode="prefix")
    out = model(past, future)
    loss, parts = horizon_jepa_loss(
        out,
        variance_weight=0.1,
        covariance_weight=0.1,
        retrieval_weight=0.1,
    )

    assert prefix.shape == (6, 3, 5)
    assert frame.shape == (6, 1, 5)
    assert prefix_target.shape == (6, 3, 5)
    assert torch.equal(frame[:, 0, :], future[:, 2, :])
    assert out["context"].shape == (6, 4)
    assert out["target"].shape == (6, 2, 4)
    assert out["predicted"].shape == (6, 2, 4)
    assert torch.isfinite(loss)
    assert parts["prediction"] >= 0.0
    assert parts["retrieval"] > 0.0


def test_horizon_jepa_delta_target_subtracts_last_past_frame():
    import torch

    past = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[-1.0, 0.5], [2.0, -3.0]],
        ]
    )
    future = torch.tensor(
        [
            [[4.0, 5.0], [7.0, 9.0], [10.0, 12.0]],
            [[3.0, -1.0], [4.0, -2.0], [8.0, 1.0]],
        ]
    )

    delta_frame = select_horizon_delta_target(
        past,
        future,
        horizon=2,
        target_mode="frame",
    )
    delta_prefix = select_horizon_delta_target(
        past,
        future,
        horizon=2,
        target_mode="prefix",
    )

    torch.testing.assert_close(delta_frame[:, 0, :], future[:, 1, :] - past[:, -1, :])
    torch.testing.assert_close(delta_prefix, future[:, :2, :] - past[:, -1:, :])


def test_fixed_delta_pca_target_round_trips_rank_two_deltas():
    train_deltas = np.array(
        [
            [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]],
            [[2.0, 0.0, 4.0], [0.0, 2.0, -2.0]],
            [[-1.0, 0.0, -2.0], [0.0, -1.0, 1.0]],
            [[-2.0, 0.0, -4.0], [0.0, -2.0, 2.0]],
        ],
        dtype=np.float32,
    )
    target = fit_delta_pca_target(train_deltas, target_dim=2)
    z = transform_delta_targets(train_deltas, target)
    reconstructed = inverse_transform_delta_targets(z, target)

    assert z.shape == (4, 2, 2)
    np.testing.assert_allclose(reconstructed, train_deltas, atol=1e-5)


def test_make_horizon_delta_matrix_uses_requested_horizons():
    past = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, 1.0], [2.0, 3.0]],
        ],
        dtype=np.float32,
    )
    future = np.array(
        [
            [[4.0, 6.0], [5.0, 9.0], [7.0, 10.0]],
            [[3.0, 5.0], [4.0, 8.0], [8.0, 11.0]],
        ],
        dtype=np.float32,
    )

    deltas = make_horizon_delta_matrix(past, future, horizons=(1, 3))

    assert deltas.shape == (2, 2, 2)
    np.testing.assert_allclose(deltas[:, 0, :], future[:, 0, :] - past[:, -1, :])
    np.testing.assert_allclose(deltas[:, 1, :], future[:, 2, :] - past[:, -1, :])


def test_target_encoder_distill_model_outputs_horizon_codes():
    import torch

    torch.manual_seed(29)
    cfg = TargetEncoderDistillConfig(
        input_dim=5,
        hidden_dim=12,
        target_dim=3,
        horizons=(1, 3),
    )
    model = DeltaTargetEncoder(cfg)
    deltas = torch.randn(7, 2, 5)
    target = torch.randn(7, 2, 3)

    predicted = model(deltas)
    loss, parts = target_encoder_distill_loss(predicted, target)

    assert predicted.shape == (7, 2, 3)
    assert torch.isfinite(loss)
    assert parts["target_mse"] >= 0.0


def test_evaluate_distilled_targets_scores_perfect_fixed_pca_codes():
    train_deltas = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            [[1.0, 1.0, 1.0], [-1.0, 1.0, 2.0]],
        ],
        dtype=np.float32,
    )
    pca = fit_delta_pca_target(train_deltas, target_dim=3)
    target_z = transform_delta_targets(train_deltas, pca)

    metrics = evaluate_distilled_targets(
        predicted_z=target_z,
        target_z=target_z,
        truth_delta=train_deltas,
        pca_target=pca,
        horizons=(1, 3),
    )

    assert metrics["overall_prediction"]["mse"] == 0.0
    assert metrics["overall_retrieval"]["mrr_mean"] == 1.0
    assert metrics["overall_delta_decode"]["mse_mean"] < 1e-10
    assert metrics["predicted_health"]["effective_rank"] > 1.0


def test_frozen_target_jepa_predicts_horizon_codes_and_freezes_target_encoder():
    import torch

    torch.manual_seed(31)
    cfg = FrozenTargetJEPAConfig(
        input_dim=5,
        hidden_dim=12,
        context_dim=4,
        target_dim=3,
        predictor_hidden_dim=10,
        horizons=(1, 3),
    )
    model = FrozenTargetJEPAWorldModel(cfg)
    target_encoder = DeltaTargetEncoder(
        TargetEncoderDistillConfig(input_dim=5, hidden_dim=12, target_dim=3, horizons=(1, 3))
    )
    freeze_module(target_encoder)
    past = torch.randn(7, 4, 5)
    target = torch.randn(7, 2, 3)

    predicted = model(past)
    loss, parts = frozen_target_jepa_loss(predicted, target)

    assert predicted.shape == (7, 2, 3)
    assert torch.isfinite(loss)
    assert parts["prediction"] >= 0.0
    assert all(not param.requires_grad for param in target_encoder.parameters())


def test_direct_delta_pca_predictor_flattens_past_and_predicts_horizon_codes():
    import torch

    torch.manual_seed(37)
    cfg = DirectDeltaPCAConfig(
        input_dim=6,
        hidden_dim=10,
        target_dim=3,
        horizons=(1, 3),
    )
    model = DirectDeltaPCAPredictor(cfg)
    past = torch.randn(5, 3, 2)

    flat = flatten_past_window(past)
    predicted = model(past)

    assert flat.shape == (5, 6)
    assert predicted.shape == (5, 2, 3)


def test_fused_context_delta_pca_predictor_combines_sequence_and_direct_branches():
    import torch

    torch.manual_seed(41)
    cfg = FusedContextDeltaPCAConfig(
        input_dim=2,
        flat_input_dim=6,
        hidden_dim=10,
        context_dim=4,
        target_dim=3,
        predictor_hidden_dim=9,
        horizons=(1, 3),
    )
    model = FusedContextDeltaPCAPredictor(cfg)
    past = torch.randn(5, 3, 2)

    predicted, context = model(past, return_context=True)

    assert predicted.shape == (5, 2, 3)
    assert context.shape == (5, 4)


def test_fused_context_probe_audit_encodes_contexts():
    import torch

    torch.manual_seed(43)
    cfg = FusedContextDeltaPCAConfig(
        input_dim=2,
        flat_input_dim=6,
        hidden_dim=10,
        context_dim=4,
        target_dim=3,
        predictor_hidden_dim=9,
        horizons=(1, 3),
    )
    model = FusedContextDeltaPCAPredictor(cfg)
    past = np.random.default_rng(43).normal(size=(5, 3, 2)).astype(np.float32)

    contexts = encode_fused_contexts(
        model,
        past,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert contexts.shape == (5, 4)


def test_fused_context_probe_audit_accepts_eval_split():
    parser = build_fused_context_probe_arg_parser()

    args = parser.parse_args(["--eval_split", "test", "--max_eval_windows", "17"])

    assert args.eval_split == "test"
    assert args.max_eval_windows == 17


def test_fused_context_ema_jepa_uses_relative_context_and_frozen_target():
    import torch

    torch.manual_seed(56)
    cfg = FusedContextEMAJEPAConfig(
        input_dim=2,
        flat_input_dim=6,
        hidden_dim=10,
        latent_dim=4,
        predictor_hidden_dim=9,
        horizons=(1, 3),
    )
    model = FusedContextEMAJEPAWorldModel(cfg)
    past = torch.randn(5, 3, 2)
    future = torch.randn(5, 4, 2)

    rel = relative_to_last_observation(past)
    deltas = horizon_delta_frames(past, future, horizons=cfg.horizons)
    out = model(past, future)
    loss, parts = fused_context_ema_jepa_loss(out)

    assert rel.shape == past.shape
    assert deltas.shape == (5, 2, 2)
    assert out["context"].shape == (5, 4)
    assert out["predicted"].shape == (5, 2, 4)
    assert out["target"].shape == (5, 2, 4)
    assert torch.isfinite(loss)
    assert parts["prediction"] >= 0.0
    assert all(not param.requires_grad for param in model.target_encoder.parameters())


def test_fused_context_ema_jepa_prefix_targets_use_future_blocks():
    import torch

    torch.manual_seed(58)
    cfg = FusedContextEMAJEPAConfig(
        input_dim=2,
        flat_input_dim=6,
        hidden_dim=10,
        latent_dim=4,
        predictor_hidden_dim=9,
        horizons=(1, 3),
        target_block="prefix",
    )
    model = FusedContextEMAJEPAWorldModel(cfg)
    past = torch.randn(5, 3, 2)
    future = torch.randn(5, 4, 2)

    frame = horizon_delta_target_block(past, future, horizon=3, target_block="frame")
    prefix = horizon_delta_target_block(past, future, horizon=3, target_block="prefix")
    out = model(past, future)

    assert frame.shape == (5, 1, 2)
    assert prefix.shape == (5, 3, 2)
    torch.testing.assert_close(prefix[:, -1:, :], frame)
    assert out["predicted"].shape == (5, 2, 4)
    assert out["target"].shape == (5, 2, 4)


def test_joint_path_pca_target_roundtrips_and_predictor_shapes():
    import torch

    torch.manual_seed(60)
    deltas = np.random.default_rng(60).normal(size=(8, 2, 3)).astype(np.float32)
    target = fit_joint_delta_path_pca_target(deltas, horizons=(1, 3), target_dim=6)
    z = transform_joint_delta_path_targets(deltas, target)
    decoded = inverse_transform_joint_delta_path_targets(z, target)

    assert z.shape == (8, 6)
    assert decoded.shape == deltas.shape
    np.testing.assert_allclose(decoded, deltas, atol=1e-5)

    cfg = FusedContextJointPathPCAConfig(
        input_dim=2,
        flat_input_dim=6,
        hidden_dim=10,
        context_dim=4,
        target_dim=5,
        predictor_hidden_dim=9,
        horizons=(1, 3),
    )
    model = FusedContextJointPathPCAPredictor(cfg)
    past = torch.randn(5, 3, 2)
    predicted, context = model(past, return_context=True)

    assert predicted.shape == (5, 5)
    assert context.shape == (5, 4)


def test_joint_path_pca_probe_audit_accepts_eval_split():
    parser = build_joint_path_pca_probe_arg_parser()
    args = parser.parse_args(
        [
            "--eval_split",
            "test",
            "--max_eval_windows",
            "17",
            "--checkpoint",
            "models/world/checkpoints/part1_jepa_latent/joint_path_pca_head060.pt",
        ]
    )

    assert args.eval_split == "test"
    assert args.max_eval_windows == 17
    assert args.checkpoint.endswith("joint_path_pca_head060.pt")


def test_horizon_jepa_trainable_target_gets_regularization_gradients():
    import torch

    torch.manual_seed(13)
    cfg = HorizonJEPAConfig(
        input_dim=5,
        hidden_dim=8,
        latent_dim=4,
        predictor_hidden_dim=8,
        horizons=(1, 3),
    )
    model = HorizonJEPAWorldModel(cfg, target_encoder_mode="trainable")
    past = torch.randn(6, 4, 5)
    future = torch.randn(6, 5, 5)

    out = model(past, future)
    loss, _parts = horizon_jepa_loss(
        out,
        variance_weight=0.1,
        covariance_weight=0.1,
        retrieval_weight=0.1,
        target_regularization_grad=True,
    )
    loss.backward()

    target_grad = next(model.target_encoder.parameters()).grad
    assert target_grad is not None
    assert torch.isfinite(target_grad).all()


def test_supervised_horizon_delta_targets_reconstruct_frames():
    import torch

    torch.manual_seed(17)
    cfg = SupervisedHorizonConfig(
        input_dim=5,
        hidden_dim=8,
        context_dim=4,
        predictor_hidden_dim=8,
        horizons=(1, 3),
        target_mode="delta",
    )
    model = SupervisedHorizonFrameModel(cfg)
    past = torch.randn(6, 4, 5)
    future = torch.randn(6, 5, 5)

    target, frame = make_horizon_frame_targets(
        past,
        future,
        horizons=cfg.horizons,
        target_mode="delta",
    )
    pred_target = model(past)
    pred_frame = decode_horizon_prediction(
        past,
        pred_target,
        target_mode="delta",
    )
    loss, parts = supervised_horizon_loss(pred_target, target, frame, past)
    contrastive_loss, contrastive_parts = supervised_horizon_loss(
        pred_target,
        target,
        frame,
        past,
        retrieval_weight=0.1,
    )

    assert target.shape == (6, 2, 5)
    assert frame.shape == (6, 2, 5)
    assert pred_target.shape == (6, 2, 5)
    assert pred_frame.shape == (6, 2, 5)
    torch.testing.assert_close(target[:, 1, :], future[:, 2, :] - past[:, -1, :])
    assert torch.isfinite(loss)
    assert torch.isfinite(contrastive_loss)
    assert parts["target_mse"] >= 0.0
    assert contrastive_parts["retrieval"] > 0.0


def test_supervised_horizon_loss_adds_context_regularizers():
    import torch

    past = torch.randn(8, 4, 3)
    predicted_target = torch.zeros(8, 2, 3)
    target = torch.zeros(8, 2, 3)
    frame = decode_horizon_prediction(past, target, target_mode="delta")
    context = torch.randn(8, 5)

    base_loss, base_parts = supervised_horizon_loss(
        predicted_target,
        target,
        frame,
        past,
        target_mode="delta",
        frame_weight=0.0,
    )
    regularized_loss, parts = supervised_horizon_loss(
        predicted_target,
        target,
        frame,
        past,
        target_mode="delta",
        frame_weight=0.0,
        context=context,
        context_variance_weight=0.1,
        context_covariance_weight=0.1,
        context_variance_gamma=0.5,
    )

    assert base_parts["context_variance"] == 0.0
    assert base_parts["context_covariance"] == 0.0
    assert parts["context_variance"] >= 0.0
    assert parts["context_covariance"] >= 0.0
    assert regularized_loss > base_loss


def test_context_correlation_loss_penalizes_duplicate_dimensions():
    import torch

    torch.manual_seed(23)
    context = torch.randn(64, 4)
    duplicated = context.clone()
    duplicated[:, 1] = duplicated[:, 0]

    assert context_correlation_loss(duplicated) > context_correlation_loss(context)


def test_soft_neighborhood_contrastive_loss_prefers_aligned_predictions():
    import torch

    target = torch.eye(6)
    aligned = target.clone()
    permuted = torch.roll(target, shifts=1, dims=0)

    aligned_loss = soft_neighborhood_contrastive_loss(
        aligned,
        target,
        temperature=0.1,
        target_temperature=0.2,
    )
    permuted_loss = soft_neighborhood_contrastive_loss(
        permuted,
        target,
        temperature=0.1,
        target_temperature=0.2,
    )

    assert aligned_loss < permuted_loss


def test_checkpoint_selection_score_prefers_requested_metric():
    low_mse = {"val_mse": 0.01, "val_mrr_mean": 0.02, "val_top5_mean": 0.03}
    high_mrr = {"val_mse": 0.02, "val_mrr_mean": 0.05, "val_top5_mean": 0.02}

    assert checkpoint_selection_score(low_mse, "mse") > checkpoint_selection_score(high_mrr, "mse")
    assert checkpoint_selection_score(high_mrr, "mrr") > checkpoint_selection_score(low_mse, "mrr")
    assert checkpoint_selection_score(low_mse, "top5") > checkpoint_selection_score(high_mrr, "top5")


def test_save_epoch_checkpoint_writes_numbered_checkpoint(tmp_path):
    import torch

    cfg = SupervisedHorizonConfig(input_dim=3, hidden_dim=4, context_dim=2, predictor_hidden_dim=5)
    model = SupervisedHorizonFrameModel(cfg)
    row = {"epoch": 7, "val_mse": 0.1, "val_mrr_mean": 0.2}

    checkpoint_path = save_epoch_checkpoint(
        tmp_path,
        epoch=7,
        model=model,
        cfg=cfg,
        epoch_summary=row,
        args={"selection_metric": "mrr"},
    )

    assert checkpoint_path == epoch_checkpoint_path(tmp_path, 7)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 7
    assert payload["epoch_summary"] == row
    assert payload["config"]["input_dim"] == 3


def test_ridge_probe_predict_recovers_linear_multivariate_targets():
    train_x = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ],
        dtype=np.float64,
    )
    weights = np.array([[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]], dtype=np.float64)
    bias = np.array([0.5, -0.25, 0.75], dtype=np.float64)
    train_y = train_x @ weights + bias
    val_x = np.array([[0.5, 0.5], [3.0, -2.0]], dtype=np.float64)
    val_y = val_x @ weights + bias

    pred = ridge_probe_predict(train_x, train_y, val_x, alpha=1e-9)

    assert np.mean((pred - val_y) ** 2) < 1e-10


def test_ridge_probe_metrics_reports_horizon_metrics():
    train_context = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ],
        dtype=np.float64,
    )
    val_context = np.array([[0.5, 0.5], [3.0, -2.0]], dtype=np.float64)
    weights_h1 = np.array([[1.0, -1.0], [0.5, 0.25]], dtype=np.float64)
    weights_h5 = np.array([[0.25, 1.5], [-1.0, 0.75]], dtype=np.float64)
    train_targets = np.stack(
        [train_context @ weights_h1, train_context @ weights_h5],
        axis=1,
    )
    val_targets = np.stack(
        [val_context @ weights_h1, val_context @ weights_h5],
        axis=1,
    )

    metrics = ridge_probe_metrics(
        train_context,
        val_context,
        train_targets,
        val_targets,
        horizons=(1, 5),
        alpha=1e-9,
    )

    assert metrics["overall_prediction"]["mse"] < 1e-10
    assert metrics["per_horizon"]["1"]["prediction"]["mse"] < 1e-10
    assert metrics["per_horizon"]["5"]["prediction"]["mse"] < 1e-10


def test_composite_part1_score_rewards_rank_probe_and_retrieval():
    weaker_train = {
        "best_val_metrics": {
            "overall_prediction": {"mse": 0.019},
            "overall_retrieval": {"mrr_mean": 0.04, "top5_mean": 0.05, "top10_mean": 0.10},
        },
        "raw_persistence_baseline": {
            "prediction": {"mse_mean": 0.02},
            "retrieval": {"top5_mean": 0.04, "top10_mean": 0.09},
        },
    }
    weaker_audit = {
        "config": {"context_dim": 10},
        "val_context_health": {
            "effective_rank": 2.0,
            "offdiag_abs_mean": 0.6,
        },
        "ridge_probe_target_metrics": {
            "overall_prediction": {"mse": 0.018},
            "overall_retrieval": {"mrr_mean": 0.08},
        },
        "zero_delta_target_baseline": {
            "overall_prediction": {"mse": 0.02},
        },
    }
    stronger_train = {
        "best_val_metrics": {
            "overall_prediction": {"mse": 0.015},
            "overall_retrieval": {"mrr_mean": 0.06, "top5_mean": 0.07, "top10_mean": 0.12},
        },
        "raw_persistence_baseline": {
            "prediction": {"mse_mean": 0.02},
            "retrieval": {"top5_mean": 0.04, "top10_mean": 0.09},
        },
    }
    stronger_audit = {
        "config": {"context_dim": 10},
        "val_context_health": {
            "effective_rank": 5.0,
            "offdiag_abs_mean": 0.3,
        },
        "ridge_probe_target_metrics": {
            "overall_prediction": {"mse": 0.015},
            "overall_retrieval": {"mrr_mean": 0.11},
        },
        "zero_delta_target_baseline": {
            "overall_prediction": {"mse": 0.02},
        },
    }

    assert composite_part1_score(stronger_train, stronger_audit)["score"] > composite_part1_score(
        weaker_train,
        weaker_audit,
    )["score"]


def test_composite_part1_score_penalizes_frame_mse_worse_than_persistence():
    audit = {
        "config": {"context_dim": 10},
        "val_context_health": {
            "effective_rank": 5.0,
            "offdiag_abs_mean": 0.3,
        },
        "ridge_probe_target_metrics": {
            "overall_prediction": {"mse": 0.015},
            "overall_retrieval": {"mrr_mean": 0.11},
        },
        "zero_delta_target_baseline": {
            "overall_prediction": {"mse": 0.02},
        },
    }
    beats_persistence = {
        "best_val_metrics": {
            "overall_prediction": {"mse": 0.018},
            "overall_retrieval": {"mrr_mean": 0.06, "top5_mean": 0.07, "top10_mean": 0.12},
        },
        "raw_persistence_baseline": {
            "prediction": {"mse_mean": 0.02},
            "retrieval": {"top5_mean": 0.04, "top10_mean": 0.09},
        },
    }
    misses_persistence = {
        "best_val_metrics": {
            "overall_prediction": {"mse": 0.022},
            "overall_retrieval": {"mrr_mean": 0.06, "top5_mean": 0.07, "top10_mean": 0.12},
        },
        "raw_persistence_baseline": {
            "prediction": {"mse_mean": 0.02},
            "retrieval": {"top5_mean": 0.04, "top10_mean": 0.09},
        },
    }

    assert composite_part1_score(beats_persistence, audit)["score"] > composite_part1_score(
        misses_persistence,
        audit,
    )["score"]


def test_composite_part1_score_penalizes_topk_below_persistence():
    audit = {
        "config": {"context_dim": 10},
        "val_context_health": {
            "effective_rank": 5.0,
            "offdiag_abs_mean": 0.3,
        },
        "ridge_probe_target_metrics": {
            "overall_prediction": {"mse": 0.015},
            "overall_retrieval": {"mrr_mean": 0.11},
        },
        "zero_delta_target_baseline": {
            "overall_prediction": {"mse": 0.02},
        },
    }
    stronger_topk = {
        "best_val_metrics": {
            "overall_prediction": {"mse": 0.018},
            "overall_retrieval": {"mrr_mean": 0.06, "top5_mean": 0.09, "top10_mean": 0.14},
        },
        "raw_persistence_baseline": {
            "prediction": {"mse_mean": 0.02},
            "retrieval": {"top5_mean": 0.08, "top10_mean": 0.13},
        },
    }
    weaker_topk = {
        "best_val_metrics": {
            "overall_prediction": {"mse": 0.018},
            "overall_retrieval": {"mrr_mean": 0.06, "top5_mean": 0.06, "top10_mean": 0.12},
        },
        "raw_persistence_baseline": {
            "prediction": {"mse_mean": 0.02},
            "retrieval": {"top5_mean": 0.08, "top10_mean": 0.13},
        },
    }

    assert composite_part1_score(stronger_topk, audit)["score"] > composite_part1_score(
        weaker_topk,
        audit,
    )["score"]
