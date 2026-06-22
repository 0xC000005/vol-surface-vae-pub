"""THROWAWAY PROTOTYPE — Approach 4: narrative-led innovation, re-based onto the user's start.

NOT production. NOT wired into the demo. A demonstrator so we can SEE the design behave
before writing the spec. Reuses two already-saved runs of the SAME narrative:
  - narrative-led  : narrative picked its own start (2015-07-15)  -> _probe_narrative_led/
  - fixed-start    : user pinned 2008-10-24                       -> _probe_gold_support_2185/

Approach 4 = take the narrative-led GENERATED forward path (the regime's innovation),
express it as a per-factor return, and re-anchor it onto the user's chosen start LEVEL.
Direction comes from the narrative-led regime; level comes from the user's pick.
"""
import numpy as np, json

NL_DIR = "experiments/backfill/block_ar/nl_scenario_demo_outputs/_probe_narrative_led"
FIXED_DIR = "experiments/backfill/block_ar/nl_scenario_demo_outputs/_probe_gold_support_2185"
SB = "experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_support_bank_train_all_939a"

FACTORS = {"SPX": 25, "USDJPY": 27, "DXY": 28, "CRUDE_OIL": 31,
           "US2Y": 32, "US10Y": 33, "GOLD": 37, "VIX": 38}


def _median_path(arr_dir, variant):
    z = np.load(f"{arr_dir}/prefix_latent_story_smoke_arrays.npz", allow_pickle=True)
    gs = z["generated_states"]          # (V, S, 30, 39) level paths
    start = z["requested_raw"][variant]  # (39,) day-0 levels
    med_terminal = np.median(gs[variant, :, -1, :], axis=0)  # (39,)
    return start, med_terminal, z["start_indices"]


def approach4(user_start_window):
    """Re-base the narrative-led innovation onto the user's chosen start window."""
    wm = json.load(open(f"{SB}/support_bank_report.json"))["window_metadata"]
    user_levels = np.load(f"{SB}/support_bank_arrays.npz")["history_raw"][user_start_window, -1, :]
    user_date = wm[user_start_window].get("calendar_end_date")

    nl_start, nl_term, nl_idx = _median_path(NL_DIR, variant=1)   # narrative-led operational
    nl_start_date = wm[int(np.asarray(nl_idx).reshape(-1)[1])].get("calendar_end_date")
    fx_start, fx_term, _ = _median_path(FIXED_DIR, variant=1)     # fixed-start operational

    print(f"  Narrative-led start  : {nl_start_date} (the narrative's own pick)")
    print(f"  Your chosen start    : {user_date} (window {user_start_window})\n")
    print(f"  {'factor':9} | {'CURRENT demo (fixed-start)':>28} | {'APPROACH 4 (re-based)':>30}")
    print(f"  {'-'*9}-+-{'-'*28}-+-{'-'*30}")
    for name, c in FACTORS.items():
        # narrative-led regime as a return, then re-base onto the user's level
        nl_ret = nl_term[c] / nl_start[c] - 1.0
        rebased = user_levels[c] * (1.0 + nl_ret)
        cur_dir = "UP" if fx_term[c] > fx_start[c] else "DOWN"
        a4_dir = "UP" if rebased > user_levels[c] else "DOWN"
        print(f"  {name:9} | {fx_start[c]:8.1f}->{fx_term[c]:8.1f} {cur_dir:>4}     | "
              f"{user_levels[c]:8.1f}->{rebased:8.1f} {a4_dir:>4} ({nl_ret*100:+5.1f}% regime)")


if __name__ == "__main__":
    print("=" * 80)
    print("APPROACH 4 PROTOTYPE — your 'higher rates / firmer dollar' narrative")
    print("=" * 80)
    approach4(user_start_window=2185)  # 2008-10-24 (the start you were testing)
