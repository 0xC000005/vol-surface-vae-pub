import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_component_backtest import (
    _build_story_args,
)


def test_component_backtest_passes_broad_support_bank_and_policy(tmp_path: Path) -> None:
    args = SimpleNamespace(
        bridge_report="bridge.json",
        bridge_arrays="bridge.npz",
        pipeline_report="pipeline.json",
        support_bank_report="support.json",
        support_bank_arrays="support.npz",
        checkpoint="model.pt",
        query_role="anchor",
        memory_prior_mode="diverse_topk_narrative_start_checked",
        memory_prior_top_k=8,
        memory_prior_temperature=0.2,
        memory_prior_diverse_max_pairwise_cosine=0.95,
        memory_prior_diverse_min_index_gap=30,
        memory_prior_quality_guard_candidate_pool_size=24,
        memory_prior_quality_guard_mixture_size=3,
        memory_prior_quality_guard_max_mixtures=128,
        memory_prior_quality_guard_min_candidate_mixtures=4,
        memory_prior_quality_guard_max_candidate_entropy_quantile=-1.0,
        memory_prior_quality_guard_min_support_weight_max_quantile=0.25,
        memory_prior_quality_guard_probability_temperature=-1.0,
        start_distance_penalty=0.02,
        implication_alignment_weight=0.25,
        decoder_steps=12,
        batch_size=4,
        eval_batch_size=4,
        samples=4,
        n_steps=30,
        chunk_size=4,
        temperature=0.5,
        seed=7,
        seed_stride=11,
        device="cpu",
        max_paths=1,
        response_preview_samples_per_component=5,
        response_preview_alpha=0.8,
        response_preview_temperature=0.7,
        response_preview_blend=0.35,
        response_preview_objective="factor_portfolio",
    )
    query = {"kind": "rich_summary", "window_id": "joint39_test_0001", "window_index": 3}

    story_args = _build_story_args(
        query=query,
        mode="component_prefix_mixture",
        output_dir=tmp_path,
        args=args,
    )

    assert story_args.support_bank_report == "support.json"
    assert story_args.support_bank_arrays == "support.npz"
    assert story_args.memory_prior_mode == "diverse_topk_narrative_start_checked"
    assert story_args.memory_prior_diverse_min_index_gap == 30
    assert story_args.memory_prior_quality_guard_candidate_pool_size == 24
    assert story_args.memory_prior_quality_guard_max_mixtures == 128
    assert story_args.memory_prior_quality_guard_probability_temperature == -1.0
    assert story_args.start_distance_penalty == 0.02
    assert story_args.implication_alignment_weight == 0.25
    assert story_args.response_preview_samples_per_component == 5
    assert story_args.response_preview_alpha == 0.8
    assert story_args.response_preview_temperature == 0.7
    assert story_args.response_preview_blend == 0.35
    assert story_args.response_preview_objective == "factor_portfolio"
