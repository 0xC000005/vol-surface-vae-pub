"""Quick diagnostic: Is the encoder output actually different for different histories?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import torch
from diffusion.block_ar.block_ar_ddpm import BlockARConfig, ConditionalBlockARDDPM
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset

# Load model (regular weights, not EMA)
checkpoint = torch.load(
    "models/backfill/block_ar/best_coverage_model.pt",
    map_location="cpu", weights_only=False,
)
model_config = checkpoint["config"]
if isinstance(model_config, dict):
    model_config = BlockARConfig(**model_config)

model = ConditionalBlockARDDPM(model_config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load test data
data = np.load("data/vol_surface_with_ret.npz")
surfaces = data["surface"]
dataset = VolSurfaceDataset(surfaces, 30, 30, start_idx=4540)

# Get two different histories
batch1 = dataset[0]
batch2 = dataset[100]
history1 = batch1["history"].unsqueeze(0)  # (1, 30, 5, 5)
history2 = batch2["history"].unsqueeze(0)

print("=" * 60)
print("CONDITIONALITY DIAGNOSTIC")
print("=" * 60)

# 1. Check encoder outputs
with torch.no_grad():
    cond1 = model.encoder(history1, mask=None)
    cond2 = model.encoder(history2, mask=None)
    cond_null = model.encoder.null_embedding

print("\n--- Encoder Outputs ---")
print(f"Cond1 (history 0):   mean={cond1.mean():.4f}, std={cond1.std():.4f}, norm={cond1.norm():.4f}")
print(f"Cond2 (history 100): mean={cond2.mean():.4f}, std={cond2.std():.4f}, norm={cond2.norm():.4f}")
print(f"Null embedding:      mean={cond_null.mean():.4f}, std={cond_null.std():.4f}, norm={cond_null.norm():.4f}")
print(f"Distance(c1, c2):    {(cond1 - cond2).norm():.4f}")
print(f"Distance(c1, null):  {(cond1 - cond_null).norm():.4f}")
print(f"Distance(c2, null):  {(cond2 - cond_null).norm():.4f}")
print(f"Cond1 values: {cond1[0, :8].numpy()}")
print(f"Cond2 values: {cond2[0, :8].numpy()}")

# 2. Check if denoiser actually uses the conditioning
print("\n--- Denoiser Sensitivity to Conditioning ---")
noisy = torch.randn(1, 10, 25)
positions = torch.arange(10).unsqueeze(0)
noise_levels = torch.full((1, 10), 50)

with torch.no_grad():
    pred1 = model.denoiser(noisy, cond1, positions, noise_levels)
    pred2 = model.denoiser(noisy, cond2, positions, noise_levels)
    pred_null = model.denoiser(noisy, cond_null.expand(1, -1), positions, noise_levels)

diff_12 = (pred1 - pred2).abs().mean().item()
diff_1null = (pred1 - pred_null).abs().mean().item()
print(f"Pred diff (cond1 vs cond2):  {diff_12:.6f}")
print(f"Pred diff (cond1 vs null):   {diff_1null:.6f}")
print(f"Pred magnitude:              {pred1.abs().mean():.6f}")
print(f"Relative sensitivity (c1 vs c2): {diff_12 / pred1.abs().mean() * 100:.2f}%")

# 3. Check the full sampling path — does different conditioning produce different samples?
print("\n--- Sample Sensitivity ---")
torch.manual_seed(42)
with torch.no_grad():
    sample1 = model.sample(history1, n_samples=5, max_residual=20)
torch.manual_seed(42)
with torch.no_grad():
    sample2 = model.sample(history2, n_samples=5, max_residual=20)

# Same seed but different histories — samples should differ if conditionality works
diff_samples = (sample1 - sample2).abs().mean().item()
print(f"Sample diff (same seed, diff history): {diff_samples:.6f}")
print(f"Sample1 range: [{sample1.min():.4f}, {sample1.max():.4f}]")
print(f"Sample2 range: [{sample2.min():.4f}, {sample2.max():.4f}]")

# 4. Check per-block: does conditioning change across blocks?
print("\n--- Per-Block Conditioning Analysis ---")
with torch.no_grad():
    # Block 1: only history
    cond_block1 = model.encoder(history1, mask=None)
    print(f"Block 1 cond norm: {cond_block1.norm():.4f}")

    # Block 2: history + generated block 1
    # Simulate by extending history
    extended = torch.cat([history1, torch.randn(1, 10, 5, 5) * 0.1], dim=1)
    cond_block2 = model.encoder(extended, mask=None)
    print(f"Block 2 cond norm: {cond_block2.norm():.4f}")
    print(f"Distance(block1, block2 cond): {(cond_block1 - cond_block2).norm():.4f}")

# 5. Check weight magnitudes in the bottleneck
print("\n--- Encoder Weight Analysis ---")
for name, p in model.encoder.named_parameters():
    print(f"  {name}: shape={list(p.shape)}, mean={p.data.mean():.4f}, std={p.data.std():.4f}, norm={p.data.norm():.4f}")

# 6. Check denoiser input projection weights for the conditioning portion
print("\n--- Denoiser Input Projection Weight Analysis ---")
# The input_proj takes [noisy(25) + cond(16) + pos_embed(16) + noise_embed(16)] = 73
# Check how much weight the conditioning part gets
weight = model.denoiser.input_proj[0].weight.data  # (hidden, 73)
frame_weight = weight[:, :25].norm()
cond_weight = weight[:, 25:41].norm()
pos_weight = weight[:, 41:57].norm()
noise_weight = weight[:, 57:73].norm()
print(f"Frame (0:25) weight norm:     {frame_weight:.4f}")
print(f"Condition (25:41) weight norm: {cond_weight:.4f}")
print(f"Position (41:57) weight norm:  {pos_weight:.4f}")
print(f"Noise (57:73) weight norm:     {noise_weight:.4f}")
print(f"Cond/Total ratio: {cond_weight / (frame_weight + cond_weight + pos_weight + noise_weight) * 100:.1f}%")

# 7. Check if the model learned to use conditioning during training (MCVD masking)
print("\n--- MCVD Training Analysis ---")
print(f"p_mask during training: {model_config.p_mask}")
print(f"This means ~{model_config.p_mask*100:.0f}% of training, past/future were independently masked")
print(f"25% FORWARD (past visible), 25% BACKWARD (future visible)")
print(f"25% INTERPOLATION (both visible), 25% UNCONDITIONAL (neither)")
print(f"Cond augmentation sigma: {model_config.cond_aug_sigma}")

# 8. Compare EMA vs regular weights conditioning
print("\n--- EMA vs Regular Weight Conditioning ---")
# Load EMA model
model_ema = ConditionalBlockARDDPM(model_config)
state_dict = model_ema.state_dict()
for name in state_dict:
    if name in checkpoint["ema_params"]:
        state_dict[name] = checkpoint["ema_params"][name]
model_ema.load_state_dict(state_dict)
model_ema.eval()

with torch.no_grad():
    cond1_ema = model_ema.encoder(history1, mask=None)
    cond2_ema = model_ema.encoder(history2, mask=None)

print(f"EMA Cond1 norm:     {cond1_ema.norm():.4f}")
print(f"EMA Cond2 norm:     {cond2_ema.norm():.4f}")
print(f"EMA Distance(c1,c2): {(cond1_ema - cond2_ema).norm():.4f}")
print(f"Reg Distance(c1,c2): {(cond1 - cond2).norm():.4f}")

print("\n--- Diagnosis Complete ---")
