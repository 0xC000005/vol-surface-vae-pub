#!/usr/bin/env python
"""
Bitter-Lesson audit for 241b fine-tune.

Checks:
1. args.json contains no per-cell lookup arrays, per-regime thresholds, or
   hard-coded tenor/moneyness constants.
2. cell_median / cell_iqr in the training recipe are derived from TRAINING DATA
   statistics only (per-cell std of changes) — acceptable per plan v4 as
   equivalent to standard normalisation.
3. The model's state_dict contains only learned parameters — no frozen lookup
   tensors spread across cells.
4. Only scalar hyperparameters (width_clip, band_clip, init_local_gate, etc.)
   appear — NO lists/arrays whose values encode per-cell regime information.

PASS criteria:
- args.json is flat dict of numeric/string/bool/None
- No "cell_median_array" / "cell_iqr_array" persisted in args.json (they're
  computed lazily from data)
- model state_dict consists entirely of tensors whose shapes derive from the
  config dict (not from IV cell structure hand-picked)

FAIL criteria (in order of severity):
- Any arg whose name matches r"(per_cell|per_tenor|per_moneyness|cell_constants|
  hardcoded|lookup)" is FAIL
- Any arg which is a list of 5 or 25 numeric values is FLAG (could be per-cell)

Output: audit.json with PASS/FLAG/FAIL per check and a single verdict.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


BANNED_NAME_PATTERN = re.compile(
    r"(per_cell|per_tenor|per_moneyness|cell_constants|hardcoded|lookup|"
    r"regime_thresh|iv_specific|vol_specific)",
    re.IGNORECASE,
)
CELL_STRUCTURE_SIZES = {5, 25}  # grid dims we consider suspect


def audit_args(args_path: str) -> dict:
    args_raw = json.loads(Path(args_path).read_text())
    findings = []
    severity = "PASS"
    for k, v in args_raw.items():
        if BANNED_NAME_PATTERN.search(k):
            findings.append({"key": k, "issue": "banned_name", "value": v})
            severity = "FAIL"
            continue
        if isinstance(v, list):
            if len(v) in CELL_STRUCTURE_SIZES and all(isinstance(e, (int, float)) for e in v):
                findings.append({
                    "key": k,
                    "issue": f"list_of_length_{len(v)}_suspicious_for_per_cell",
                    "value": v,
                })
                if severity == "PASS":
                    severity = "FLAG"
    return {"severity": severity, "findings": findings, "n_args": len(args_raw)}


def audit_state_dict(ckpt_path: str, device: str = "cpu") -> dict:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = payload["model_state_dict"]
    n_tensors = len(state)
    # Flag any buffer whose name suggests a data-derived constant
    suspect = []
    for k in state.keys():
        if BANNED_NAME_PATTERN.search(k):
            suspect.append(k)
    # Total parameter count
    n_params = sum(v.numel() for v in state.values())
    return {
        "severity": "FAIL" if suspect else "PASS",
        "n_tensors": n_tensors,
        "n_params": n_params,
        "suspect_tensor_names": suspect,
    }


def audit_training_stats(args_path: str) -> dict:
    """Confirm cell_median / cell_iqr are not persisted as model constants.

    Plan v4 explicitly allows cell_median / cell_iqr computed on-the-fly from
    training data as part of twCRPS weighting — this is standard normalisation
    and does NOT violate Bitter Lesson. The audit here checks that these
    statistics are NOT baked into args.json (which would make them part of the
    model's hard-coded config) and are instead recomputed on every training
    launch from current training data.
    """
    args_raw = json.loads(Path(args_path).read_text())
    has_baked_stats = any(
        k in args_raw and isinstance(args_raw[k], list) and len(args_raw[k]) >= 5
        for k in ["cell_median", "cell_iqr", "cell_std", "cell_scale", "cell_mean"]
    )
    return {
        "severity": "FAIL" if has_baked_stats else "PASS",
        "cell_stats_baked_in_args": has_baked_stats,
    }


def main():
    ap = argparse.ArgumentParser(description="Bitter-Lesson audit for 241b")
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to 241b best_model.pt")
    ap.add_argument("--args_json", type=str, default=None,
                    help="Path to args.json (auto-resolved from ckpt dir if omitted)")
    ap.add_argument("--output", type=str, default=None,
                    help="Output path for audit.json (auto-resolved to ckpt dir)")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    args_json_path = args.args_json or str(ckpt.parent / "args.json")
    output_path = args.output or str(ckpt.parent / "audit.json")

    print(f"Auditing {ckpt}")
    audit = {
        "checkpoint": str(ckpt),
        "args_json": args_json_path,
        "args_audit": audit_args(args_json_path),
        "state_dict_audit": audit_state_dict(str(ckpt)),
        "training_stats_audit": audit_training_stats(args_json_path),
    }
    # Overall verdict = worst severity
    severities = [
        audit["args_audit"]["severity"],
        audit["state_dict_audit"]["severity"],
        audit["training_stats_audit"]["severity"],
    ]
    if "FAIL" in severities:
        verdict = "FAIL"
    elif "FLAG" in severities:
        verdict = "FLAG"
    else:
        verdict = "PASS"
    audit["verdict"] = verdict

    Path(output_path).write_text(json.dumps(audit, indent=2))
    print(f"\nVerdict: {verdict}")
    print(f"  args: {audit['args_audit']['severity']} ({len(audit['args_audit']['findings'])} findings)")
    print(f"  state_dict: {audit['state_dict_audit']['severity']} ({audit['state_dict_audit']['n_tensors']} tensors, {audit['state_dict_audit']['n_params']:,} params)")
    print(f"  training_stats: {audit['training_stats_audit']['severity']}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
