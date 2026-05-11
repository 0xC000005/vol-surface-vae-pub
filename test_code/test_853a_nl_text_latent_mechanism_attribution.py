import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_text_latent_mechanism_attribution import (
    ANCHOR_ONLY_POLICY,
    INCUMBENT_POLICY,
    NO_NEGATIVE_POLICY,
    build_mechanism_attribution,
)


def _metric(mean: float) -> dict[str, object]:
    return {
        "mean": mean,
        "min": mean - 0.01,
        "max": mean + 0.01,
        "values": [mean - 0.01, mean + 0.01],
    }


def _policy(
    target_cosine: float,
    gap: float,
    margin: float,
    recall1: float,
    recall3: float,
) -> dict[str, dict[str, object]]:
    return {
        "heldout_mean_target_cosine": _metric(target_cosine),
        "heldout_hard_negative_mean_gap": _metric(gap),
        "heldout_hard_negative_mean_margin": _metric(margin),
        "heldout_recall_at_1_test_pool": _metric(recall1),
        "heldout_recall_at_3_test_pool": _metric(recall3),
    }


def test_build_mechanism_attribution_prefers_mlp_hard_negative_incumbent() -> None:
    stability = {
        "summary": {
            INCUMBENT_POLICY: _policy(0.86, 0.93, 0.74, 0.08, 0.20),
            NO_NEGATIVE_POLICY: _policy(0.855, 0.36, 0.18, 0.07, 0.19),
            ANCHOR_ONLY_POLICY: _policy(0.828, 0.42, 0.24, 0.06, 0.18),
        },
        "winners_by_metric": {
            "heldout_hard_negative_mean_gap": [INCUMBENT_POLICY, INCUMBENT_POLICY],
            "heldout_hard_negative_mean_margin": [INCUMBENT_POLICY, INCUMBENT_POLICY],
        },
    }
    clip_sweep = {
        "comparison": {
            "clip_mse5": {
                "heldout_mean_target_cosine": {
                    "clip": 0.78,
                    "mlp": 0.856,
                    "clip_minus_mlp": -0.076,
                },
                "heldout_hard_negative_mean_gap": {
                    "clip": 0.77,
                    "mlp": 0.92,
                    "clip_minus_mlp": -0.15,
                },
                "heldout_hard_negative_mean_margin": {
                    "clip": 0.56,
                    "mlp": 0.74,
                    "clip_minus_mlp": -0.18,
                },
                "heldout_recall_at_1_test_pool": {
                    "clip": 0.11,
                    "mlp": 0.07,
                    "clip_minus_mlp": 0.04,
                },
                "heldout_recall_at_3_test_pool": {
                    "clip": 0.16,
                    "mlp": 0.19,
                    "clip_minus_mlp": -0.03,
                },
            }
        }
    }

    report = build_mechanism_attribution(stability, clip_sweep)

    assert report["status"] == "ok"
    assert report["promotion_decision"]["promote"] == INCUMBENT_POLICY
    assert report["clip_mse_sweep"]["status"] == "not_promoted"
    deltas = report["mechanism_deltas"][
        "add_hard_negatives_vs_multi_caption_no_negatives"
    ]
    assert deltas["heldout_hard_negative_mean_gap"] == 0.57
    assert report["next_principled_step"]["title"].startswith(
        "Train a supervised-contrastive MLP bridge"
    )
