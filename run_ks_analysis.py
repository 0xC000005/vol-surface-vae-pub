import os
import sys
import json
import torch
import inspect
import numpy as np
import scipy.stats as stats

# Ensure imports work
sys.path.append(os.path.abspath('.'))

from diffusion.block_ar.single_pass_ar import SinglePassBlockAR, SinglePassConfig, normalize_iv, denormalize_iv

def load_model(ckpt_path):
    print(f"Loading model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = checkpoint["config"]
    
    valid_keys = inspect.signature(SinglePassConfig).parameters.keys()
    filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    
    config = SinglePassConfig(**filtered_cfg)
    model = SinglePassBlockAR(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

def load_data():
    print("Loading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    
    # Use the validation/test set. Let's match test splits: 4540:
    test_surfaces = surfaces[4540:]
    return test_surfaces

def generate_samples(model, history_grids, num_samples=50):
    history_grids_norm = normalize_iv(history_grids)
    
    with torch.no_grad():
        samples_norm = model.sample_batched(history_grids_norm, n_samples=num_samples, batch_size=16)
        
    return denormalize_iv(samples_norm)

def main():
    model_bptt_12 = load_model("models/backfill/afcrps_164a_v3_percell_bptt/best_model.pt") # Best is ep12
    model_bptt_80 = load_model("models/backfill/afcrps_164a_v3_percell_bptt/final_model.pt") # Final is ep80
    model_sp_11 = load_model("models/backfill/afcrps_164a_v3_percell_bptt_softplus/best_model.pt") # Best is ep11
    model_sp_80 = load_model("models/backfill/afcrps_164a_v3_percell_bptt_softplus/final_model.pt") # Final is ep80
    
    test_surfaces = load_data()
    print(f"Test surfaces shape: {test_surfaces.shape}")
    
    # Take 100 windows for evaluation
    np.random.seed(42)
    start_idxs = np.random.choice(len(test_surfaces) - 60, size=100, replace=False)
    
    history_list = []
    future_list = []
    for idx in start_idxs:
        history_list.append(test_surfaces[idx:idx+30])
        future_list.append(test_surfaces[idx+30:idx+60])
        
    history_tensor = torch.tensor(np.stack(history_list), dtype=torch.float32).to(model_bptt_12.device)
    future_tensor = np.stack(future_list)
    
    print("Generating samples...")
    N_SAMPLES = 20
    s_bptt_12 = generate_samples(model_bptt_12, history_tensor, num_samples=N_SAMPLES).cpu().numpy()
    s_bptt_80 = generate_samples(model_bptt_80, history_tensor, num_samples=N_SAMPLES).cpu().numpy()
    s_sp_11 = generate_samples(model_sp_11, history_tensor, num_samples=N_SAMPLES).cpu().numpy()
    s_sp_80 = generate_samples(model_sp_80, history_tensor, num_samples=N_SAMPLES).cpu().numpy()
    
    # Spatial analysis
    print("\n" + "="*80)
    print("SPATIAL ANALYSIS OF KS-LEVELS (ALL 25 CELLS)")
    print("="*80)
    
    for c in range(5):
        for r in range(5):
            gt_vals = future_tensor[:, :, r, c].flatten()
            bp12 = s_bptt_12[:, :, :, r, c].flatten()
            sp11 = s_sp_11[:, :, :, r, c].flatten()
            
            ks_bp12 = stats.ks_2samp(gt_vals, bp12).statistic
            ks_sp11 = stats.ks_2samp(gt_vals, sp11).statistic
            
            print(f"Cell({r},{c}) | GT Mean: {np.mean(gt_vals):.4f} Std: {np.std(gt_vals):.4f} | "
                  f"KS BP12: {ks_bp12:.3f} | KS SP11: {ks_sp11:.3f} | "
                  f"BP12 Mean: {np.mean(bp12):.4f} Std: {np.std(bp12):.4f} | "
                  f"SP11 Mean: {np.mean(sp11):.4f} Std: {np.std(sp11):.4f} | "
                  f"SP11 Skew: {stats.skew(sp11):.2f} Kurt: {stats.kurtosis(sp11):.2f}")

    print("\n" + "="*80)
    print("DETAILED COMPARISON OF FAILING VS PASSING CELLS (BPTT-12 vs SP-11 vs SP-80)")
    print("="*80)
    
    # Cells to detail based on Claude's spatial pattern map:
    # Fails: (0,0), (0,1), (0,4), (1,0)
    # Passes: (0,2), (0,3), (1,3), (4,4)
    detailed_cells = [(0,0), (0,4), (0,2), (2,2), (4,4)]
    
    for (r, c) in detailed_cells:
        print(f"\n--- Cell ({r}, {c}) ---")
        gt_vals = future_tensor[:, :, r, c].flatten()
        bp12 = s_bptt_12[:, :, :, r, c].flatten()
        bp80 = s_bptt_80[:, :, :, r, c].flatten()
        sp11 = s_sp_11[:, :, :, r, c].flatten()
        sp80 = s_sp_80[:, :, :, r, c].flatten()
        
        models = {'GT': gt_vals, 'BPTT-ep12': bp12, 'BPTT-ep80': bp80, 'SP-ep11': sp11, 'SP-ep80': sp80}
        
        for name, vals in models.items():
            if name == 'GT':
                ks = 0.0
            else:
                ks = stats.ks_2samp(gt_vals, vals).statistic
            
            mean = np.mean(vals)
            std = np.std(vals)
            skew = stats.skew(vals)
            kurt = stats.kurtosis(vals)
            p01, p05, p50, p95, p99 = np.percentile(vals, [1, 5, 50, 95, 99])
            
            print(f"{name:10s} | KS: {ks:.3f} | Mean: {mean:.4f} | Std: {std:.4f} | "
                  f"Skew: {skew:5.2f} | Kurt: {kurt:5.2f} | "
                  f"P01: {p01:.4f} P50: {p50:.4f} P99: {p99:.4f}")

if __name__ == "__main__":
    main()