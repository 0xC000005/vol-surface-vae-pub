"""Population-level gold mean-reversion stats for the unified gold case study.

Computes, over the 939a support bank (4010 train windows), the relationship between the
PREFIX gold move (30-day history) and the FORWARD gold move (next 30 days), for both
directions — to upgrade the case study beyond the n=12 vignette and to match/replace the
existing 'safe-haven gold' diagnostic (which reported the gold-UP direction).

Run: PYTHONPATH=. uv run --no-sync python experiments/backfill/block_ar/case_study_gold_meanreversion_population.py
"""
import numpy as np

SB = "experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_support_bank_train_all_939a"
GOLD, CRUDE, DXY, VIX, SPX = 37, 31, 28, 38, 25


def main():
    z = np.load(f"{SB}/support_bank_arrays.npz")
    h, f = z["history_raw"], z["future_raw"]
    pre_g = h[:, -1, GOLD] - h[:, 0, GOLD]      # prefix (history) gold move
    fut_g = f[:, -1, GOLD] - h[:, -1, GOLD]      # forward (next 30d) gold move
    pre_c = h[:, -1, CRUDE] - h[:, 0, CRUDE]
    pre_dxy = h[:, -1, DXY] - h[:, 0, DXY]
    pre_vix = h[:, -1, VIX] - h[:, 0, VIX]
    pre_spx = h[:, -1, SPX] - h[:, 0, SPX]

    def report(mask, name):
        n = int(mask.sum())
        fg = fut_g[mask]
        corr = float(np.corrcoef(pre_g[mask], fg)[0, 1])
        print(f"{name:34s} n={n:4d} | fwd gold mean {fg.mean():+6.1f} "
              f"median {np.median(fg):+6.1f} | %up {100*np.mean(fg>0):4.1f}% | "
              f"corr(prefix,fwd) {corr:+.2f}")

    print(f"Full-population corr(prefix gold, forward gold) = "
          f"{np.corrcoef(pre_g, fut_g)[0,1]:+.2f} (n={h.shape[0]})\n")
    print("Gold-DOWN prefixes (the unified case study's primary direction):")
    report(pre_g < 0, "gold sold in prefix")
    report((pre_g < 0) & (pre_c < 0), "oil+gold sold in prefix")
    report((pre_g < 0) & (pre_c < 0) & (pre_dxy > 0) & (pre_vix > 0) & (pre_spx < 0),
           "full dollar-squeeze prefix")
    print("\nGold-UP prefixes (the existing safe-haven direction, for the mirror panel):")
    report(pre_g > 0, "gold up in prefix")


if __name__ == "__main__":
    main()
