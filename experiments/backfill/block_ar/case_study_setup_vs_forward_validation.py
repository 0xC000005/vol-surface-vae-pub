"""Reproducible validation for the 'conditional expectation-correction' case study.

For the dollar-squeeze narrative ("oil plus gold are being sold"), this re-runs the live
pipeline from the 2008-10-24 start, then for each retrieved support window checks:
  (a) SETUP  — did crude+gold actually fall over the support window's own 30-day HISTORY?
               (validates the retrieval matched the described condition)
  (b) FORWARD — what did crude+gold do in the realized 30 days AFTER the support window?
               (shows the conditional history: a sell-off setup tends to precede a rebound)

Run:  PYTHONPATH=. uv run --no-sync python experiments/backfill/block_ar/case_study_setup_vs_forward_validation.py
Needs OPENAI_API_KEY (.env) + cuda. Writes its run to _probe_setup_check/.
"""
import numpy as np, json
from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import build_prefix_latent_run_args
from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import run_prefix_latent_story_smoke

NARRATIVE = ("Dollar funding is tight: DXY is surging, USDJPY is breaking lower, equities and BBB "
             "credit are under pressure, and oil plus gold are being sold.")
START_WINDOW = 2185  # 2008-10-24 day-0
OUT = "experiments/backfill/block_ar/nl_scenario_demo_outputs/_probe_setup_check"
SB = "experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_support_bank_train_all_939a"
CRUDE, GOLD = 31, 37


def _support_windows(report):
    out = {}
    def walk(o):
        if isinstance(o, dict):
            if "support_window_index" in o and "weight" in o and not o.get("skipped"):
                i, w = int(o["support_window_index"]), float(o.get("weight", 0) or 0)
                if i not in out or w > out[i]:
                    out[i] = w
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(report)
    return out


def main():
    args = build_prefix_latent_run_args(start_mode="explicit_start_window",
                                        explicit_start_window_index=START_WINDOW,
                                        samples=6, live_story=True, story=NARRATIVE, output_dir=OUT)
    report = run_prefix_latent_story_smoke(args)
    supports = _support_windows(report)

    z = np.load(f"{SB}/support_bank_arrays.npz")
    wm = json.load(open(f"{SB}/support_bank_report.json"))["window_metadata"]
    hist, fut = z["history_raw"], z["future_raw"]

    n = len(supports)
    c_setup = g_setup = c_fwd_up = g_fwd_up = 0
    print(f"{'win':>5} {'date':>11} {'wt':>5} | crude hist>fwd | gold hist>fwd")
    for i, w in sorted(supports.items(), key=lambda x: -x[1]):
        c0, c1, cf = float(hist[i, 0, CRUDE]), float(hist[i, -1, CRUDE]), float(fut[i, -1, CRUDE])
        g0, g1, gf = float(hist[i, 0, GOLD]), float(hist[i, -1, GOLD]), float(fut[i, -1, GOLD])
        c_setup += c1 < c0; g_setup += g1 < g0
        c_fwd_up += cf > c1; g_fwd_up += gf > g1
        print(f"{i:>5} {wm[i].get('calendar_end_date'):>11} {w:>5.2f} | "
              f"{c0:5.0f}>{c1:5.0f}>{cf:5.0f} | {g0:5.0f}>{g1:5.0f}>{gf:5.0f}")
    print(f"\nSETUP (history):  crude DOWN {c_setup}/{n}, gold DOWN {g_setup}/{n}  <- retrieval matches the description")
    print(f"FORWARD (after):  crude UP {c_fwd_up}/{n}, gold UP {g_fwd_up}/{n}  <- conditional history (rebound)")


if __name__ == "__main__":
    main()
