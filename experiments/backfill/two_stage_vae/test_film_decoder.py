"""
Test FiLM decoder implementation.

Verifies:
1. Forward pass works with new signature
2. z cannot be ignored (different z → different output)
3. z modulates output via FiLM (direct path)
4. Temporal dynamics preserved (LSTM hidden state flows)
5. Gradient flows through both z paths (LSTM and FiLM)
"""

import torch
import sys
sys.path.insert(0, ".")

from vae.cvae_two_stage import CVAETwoStage, TwoStageDecoder
from config.two_stage_config import TwoStageConfig


def test_film_decoder():
    print("=" * 60)
    print("Testing FiLM Decoder Implementation")
    print("=" * 60)

    # Get config
    config = TwoStageConfig.get_model_config()
    config["device"] = "cpu"  # For testing

    # Create model
    model = CVAETwoStage(config)
    model.eval()

    B, T = 2, 10  # Batch size, sequence length
    ctx_dim = config["ctx_embedding_dim"]  # 3
    z_dim = config["latent_dim"]  # 16

    # Create test inputs
    ctx_emb = torch.randn(B, T, ctx_dim)
    z = torch.randn(B, T, z_dim)

    # Test 1: Forward pass works
    print("\n[Test 1] Forward pass...")
    try:
        output = model.decoder(ctx_emb, z)
        print(f"  ✓ Output shape: {output.shape}")
        assert output.shape == (B, T, 5, 5), f"Expected (B, T, 5, 5), got {output.shape}"
        print("  ✓ Forward pass successful")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        return False

    # Test 2: Different z → different output (z affects output)
    # Note: With random weights, differences may be small; trained models show larger effects
    print("\n[Test 2] z affects output (cannot be ignored)...")
    z2 = torch.randn(B, T, z_dim)  # Different z
    output2 = model.decoder(ctx_emb, z2)
    diff = (output - output2).abs().mean().item()
    print(f"  Mean absolute difference: {diff:.6f}")
    assert diff > 0.0001, f"z seems to be ignored! diff={diff}"
    print("  ✓ Different z → different output")

    # Test 3: z has direct FiLM path (test gamma/beta effect)
    print("\n[Test 3] z FiLM path works...")
    # Check that gamma_net and beta_net exist and produce different outputs
    gamma1 = model.decoder.gamma_net(z)
    gamma2 = model.decoder.gamma_net(z2)
    gamma_diff = (gamma1 - gamma2).abs().mean().item()
    print(f"  gamma difference for different z: {gamma_diff:.6f}")
    assert gamma_diff > 0.001, f"gamma_net not responding to z! diff={gamma_diff}"
    print("  ✓ z directly affects gamma (FiLM path working)")

    # Test 4: ctx_emb still affects output (through LSTM)
    # Note: ctx_emb is only 3 dims (tiny bottleneck) vs z's 16 dims
    # So we expect smaller effect from ctx_emb, which is by design
    print("\n[Test 4] ctx_emb affects output (LSTM path)...")
    ctx_emb2 = torch.randn(B, T, ctx_dim)  # Different ctx
    output3 = model.decoder(ctx_emb2, z)
    diff_ctx = (output - output3).abs().mean().item()
    print(f"  Mean absolute difference: {diff_ctx:.6f}")
    assert diff_ctx > 0.0001, f"ctx_emb seems to have no effect! diff={diff_ctx}"
    print(f"  ✓ Different ctx_emb → different output (smaller effect expected due to 3-dim bottleneck)")

    # Test 5: Temporal dynamics (LSTM hidden state flows)
    print("\n[Test 5] Temporal dynamics...")
    # If we change z at t=0, it should affect output at t=5 (via hidden state)
    z_modified = z.clone()
    z_modified[:, 0, :] = torch.randn(B, z_dim)  # Change only t=0
    output4 = model.decoder(ctx_emb, z_modified)
    diff_t5 = (output[:, 5, :, :] - output4[:, 5, :, :]).abs().mean().item()
    print(f"  Change at t=0 affects t=5 by: {diff_t5:.6f}")
    assert diff_t5 > 0.00001, f"No temporal dynamics! diff={diff_t5}"
    print("  ✓ Temporal dynamics preserved")

    # Test 6: Gradients flow through both z paths
    print("\n[Test 6] Gradient flow...")
    ctx_emb_grad = torch.randn(B, T, ctx_dim, requires_grad=True)
    z_grad = torch.randn(B, T, z_dim, requires_grad=True)
    output_grad = model.decoder(ctx_emb_grad, z_grad)
    loss = output_grad.sum()
    loss.backward()

    assert ctx_emb_grad.grad is not None, "No gradient for ctx_emb!"
    assert z_grad.grad is not None, "No gradient for z!"
    assert ctx_emb_grad.grad.abs().sum() > 0, "ctx_emb gradient is zero!"
    assert z_grad.grad.abs().sum() > 0, "z gradient is zero!"
    print(f"  ctx_emb grad norm: {ctx_emb_grad.grad.norm():.4f}")
    print(f"  z grad norm: {z_grad.grad.norm():.4f}")
    print("  ✓ Gradients flow through both paths")

    # Test 7: Full model forward pass
    print("\n[Test 7] Full model forward pass...")
    surface = torch.randn(B, T, 5, 5)
    batch = {"surface": surface}
    try:
        result = model(batch)
        print(f"  ✓ Full forward pass successful")
        print(f"    recon shape: {result[0].shape}")
        print(f"    z_mean shape: {result[1].shape}")
    except Exception as e:
        print(f"  ✗ Full forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 8: get_surface_given_conditions works
    print("\n[Test 8] get_surface_given_conditions...")
    context = torch.randn(B, 30, 5, 5)  # 30 context days
    batch_ctx = {"surface": context}
    try:
        generated = model.get_surface_given_conditions(batch_ctx, horizon=5)
        print(f"  ✓ Generated shape: {generated.shape}")
        assert generated.shape == (B, 5, 5, 5), f"Expected (B, 5, 5, 5), got {generated.shape}"
    except Exception as e:
        print(f"  ✗ get_surface_given_conditions failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 9: decode_with_predicted works
    print("\n[Test 9] decode_with_predicted...")
    ctx_emb_test = torch.randn(B, 5, ctx_dim)
    z_test = torch.randn(B, 5, z_dim)
    try:
        decoded = model.decode_with_predicted(ctx_emb_test, z_test)
        print(f"  ✓ Decoded shape: {decoded.shape}")
        assert decoded.shape == (B, 5, 5, 5), f"Expected (B, 5, 5, 5), got {decoded.shape}"
    except Exception as e:
        print(f"  ✗ decode_with_predicted failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("All tests passed! FiLM decoder working correctly.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_film_decoder()
    sys.exit(0 if success else 1)
