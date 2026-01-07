"""
Test causal masking in ctx_encoder.

Verifies:
1. ctx_emb_t depends on x_{1:t-1} only (NOT x_t)
2. ctx_emb_0 is zeros (no prior context)
3. Changing x_t does NOT affect ctx_emb_t
4. Changing x_{t-1} DOES affect ctx_emb_t
5. Full model still works
"""

import torch
import sys
sys.path.insert(0, ".")

from vae.cvae_two_stage import CVAETwoStage
from config.two_stage_config import TwoStageConfig


def test_causal_ctx_encoder():
    print("=" * 60)
    print("Testing Causal ctx_encoder")
    print("=" * 60)

    config = TwoStageConfig.get_model_config()
    config["device"] = "cpu"
    model = CVAETwoStage(config)
    model.eval()

    B, T = 2, 10
    surface = torch.randn(B, T, 5, 5)

    # Test 1: ctx_emb_0 should be zeros
    print("\n[Test 1] ctx_emb_0 is zeros...")
    with torch.no_grad():
        ctx_emb = model.ctx_encoder({"surface": surface})

    assert torch.allclose(ctx_emb[:, 0, :], torch.zeros_like(ctx_emb[:, 0, :])), \
        "ctx_emb_0 should be zeros!"
    print("  ✓ ctx_emb_0 is zeros (no prior context)")

    # Test 2: Changing x_t should NOT change ctx_emb_t
    print("\n[Test 2] ctx_emb_t independent of x_t...")
    surface_modified = surface.clone()
    surface_modified[:, 5, :, :] = torch.randn(B, 5, 5)  # Change x_5

    with torch.no_grad():
        ctx_emb_modified = model.ctx_encoder({"surface": surface_modified})

    # ctx_emb_5 should be SAME (it only depends on x_{0:4})
    diff_at_t = (ctx_emb[:, 5, :] - ctx_emb_modified[:, 5, :]).abs().max().item()
    assert diff_at_t < 1e-6, f"ctx_emb_5 changed when x_5 changed! diff={diff_at_t}"
    print(f"  ✓ Changing x_5 does NOT change ctx_emb_5 (diff={diff_at_t:.2e})")

    # Test 3: Changing x_{t-1} SHOULD change ctx_emb_t
    # Note: With random weights, differences may be small
    print("\n[Test 3] ctx_emb_t depends on x_{t-1}...")
    surface_modified2 = surface.clone()
    surface_modified2[:, 4, :, :] = torch.randn(B, 5, 5)  # Change x_4

    with torch.no_grad():
        ctx_emb_modified2 = model.ctx_encoder({"surface": surface_modified2})

    # ctx_emb_5 should be DIFFERENT (it depends on x_{0:4}, including x_4)
    diff_from_prev = (ctx_emb[:, 5, :] - ctx_emb_modified2[:, 5, :]).abs().mean().item()
    assert diff_from_prev > 1e-6, f"ctx_emb_5 didn't change when x_4 changed! diff={diff_from_prev}"
    print(f"  ✓ Changing x_4 DOES change ctx_emb_5 (diff={diff_from_prev:.6f})")

    # Test 4: Future changes don't affect current ctx_emb
    print("\n[Test 4] Future changes don't affect past ctx_emb...")
    surface_modified3 = surface.clone()
    surface_modified3[:, 8:, :, :] = torch.randn(B, 2, 5, 5)  # Change x_8, x_9

    with torch.no_grad():
        ctx_emb_modified3 = model.ctx_encoder({"surface": surface_modified3})

    # ctx_emb_5 should be SAME (x_8, x_9 are in the future)
    diff_from_future = (ctx_emb[:, 5, :] - ctx_emb_modified3[:, 5, :]).abs().max().item()
    assert diff_from_future < 1e-6, f"ctx_emb_5 changed when future changed! diff={diff_from_future}"
    print(f"  ✓ Future changes don't affect ctx_emb_5 (diff={diff_from_future:.2e})")

    # Test 5: Full model forward pass still works
    print("\n[Test 5] Full model forward pass...")
    batch = {"surface": surface}
    try:
        result = model(batch, return_full_sequence=True)
        print(f"  ✓ Forward pass successful")
        print(f"    recon shape: {result[0].shape}")
    except Exception as e:
        print(f"  ✗ Forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 6: Check shape consistency
    print("\n[Test 6] Shape consistency...")
    assert ctx_emb.shape == (B, T, config["ctx_embedding_dim"]), \
        f"Expected shape ({B}, {T}, {config['ctx_embedding_dim']}), got {ctx_emb.shape}"
    print(f"  ✓ ctx_emb shape: {ctx_emb.shape}")

    # Test 7: Verify ctx_emb_1 depends on x_0
    print("\n[Test 7] ctx_emb_1 depends on x_0...")
    surface_modified4 = surface.clone()
    surface_modified4[:, 0, :, :] = torch.randn(B, 5, 5)  # Change x_0

    with torch.no_grad():
        ctx_emb_modified4 = model.ctx_encoder({"surface": surface_modified4})

    # ctx_emb_1 should be DIFFERENT (it depends on x_0)
    diff_ctx1 = (ctx_emb[:, 1, :] - ctx_emb_modified4[:, 1, :]).abs().mean().item()
    assert diff_ctx1 > 1e-6, f"ctx_emb_1 didn't change when x_0 changed! diff={diff_ctx1}"
    print(f"  ✓ Changing x_0 DOES change ctx_emb_1 (diff={diff_ctx1:.6f})")

    print("\n" + "=" * 60)
    print("All tests passed! Causal ctx_encoder working correctly.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_causal_ctx_encoder()
    sys.exit(0 if success else 1)
