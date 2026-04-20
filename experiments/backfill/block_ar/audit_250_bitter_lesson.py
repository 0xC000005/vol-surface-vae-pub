"""
Bitter-Lesson audit for 250-series Neural Factor Model.

Two-phase audit:
  (1) Source audit — grep neural_factor.py for hardcoded spatial dims (5/25/30)
      in the module body. Config defaults are OK; only flag hardcoded constants
      used inside forward-pass code.
  (2) Checkpoint audit — if a checkpoint is provided, confirm the config dict
      contains only hyperparameters (no per-cell data-derived lookup tables) and
      that no buffer on the model holds precomputed empirical statistics per (t, d).

Run as pre-audit (source only) or post-audit (source + checkpoint).
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tokenize
from io import StringIO
from pathlib import Path

import torch


FORBIDDEN_LITERALS_IN_FORWARD = {5, 25, 30}


SOURCE_FILE = Path("diffusion/block_ar/neural_factor.py")


def _strip_docstrings_and_comments(source: str) -> str:
    """Remove docstrings (module/class/function) and comments from source.

    Comments are stripped by tokenize; docstrings by AST-based removal.
    """
    # Strip comments and string contents from docstring positions.
    out: list[str] = []
    try:
        tokens = list(tokenize.generate_tokens(StringIO(source).readline))
    except tokenize.TokenizeError:
        return source
    # First pass: find docstring token positions via AST to blank them.
    tree = ast.parse(source)
    docstring_positions: set[tuple[int, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    docstring_positions.add((body[0].lineno, body[0].col_offset))
    # Rewrite: drop comments, replace docstring tokens with blank.
    result_tokens = []
    for tok in tokens:
        tok_type, tok_string, start, end, line = tok
        if tok_type == tokenize.COMMENT:
            continue
        if tok_type == tokenize.STRING and start in docstring_positions:
            # Replace with an empty string literal to preserve grammar.
            result_tokens.append((tok_type, '""', start, end, line))
            continue
        result_tokens.append(tok)
    return tokenize.untokenize(result_tokens)


def _is_dataclass_field_line(source_lines: list[str], lineno: int) -> bool:
    """Return True if the line is inside a @dataclass body."""
    # Walk upward to find the nearest @dataclass-decorated class.
    in_dc = False
    for i in range(lineno - 1, -1, -1):
        stripped = source_lines[i].strip()
        if stripped.startswith("class ") and "NeuralFactorConfig" in stripped:
            # Need the preceding line to be the decorator.
            if i > 0 and "@dataclass" in source_lines[i - 1]:
                return True
            return False
        if stripped.startswith("class "):
            return False
    return in_dc


def audit_source() -> tuple[bool, list[str]]:
    if not SOURCE_FILE.exists():
        return False, [f"SOURCE MISSING: {SOURCE_FILE}"]
    source = SOURCE_FILE.read_text()
    orig_lines = source.splitlines()
    stripped_source = _strip_docstrings_and_comments(source)
    # Parse the stripped source and walk numeric literals; flag any in forbidden set
    # that are NOT part of a dataclass-default expression.
    tree = ast.parse(stripped_source)
    findings: list[str] = []
    offenders = 0
    # Walk class definitions to find NeuralFactorConfig body positions to exclude.
    config_line_ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "NeuralFactorConfig":
            start = node.lineno
            end = node.end_lineno or start
            config_line_ranges.append((start, end))

    def in_config(lineno: int) -> bool:
        return any(a <= lineno <= b for a, b in config_line_ranges)

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            if node.value in FORBIDDEN_LITERALS_IN_FORWARD:
                ln = node.lineno
                if in_config(ln):
                    continue
                orig = orig_lines[ln - 1] if 0 <= ln - 1 < len(orig_lines) else ""
                findings.append(f"  line {ln}: {orig.strip()}  (literal `{node.value}` in code)")
                offenders += 1
    source_ok = offenders == 0
    return source_ok, findings


def audit_checkpoint(checkpoint_path: str) -> tuple[bool, list[str]]:
    if not Path(checkpoint_path).exists():
        return False, [f"checkpoint not found: {checkpoint_path}"]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    findings: list[str] = []
    # Forbidden keys: anything hinting per-cell data-derived quantiles
    forbidden_substrings = [
        "empirical_", "precomputed_", "quantile_map", "cell_median", "cell_iqr",
    ]
    offenders = 0
    for key, val in cfg.items():
        for bad in forbidden_substrings:
            if bad in key:
                findings.append(f"  config[{key}] = {val}")
                offenders += 1
                break
    # Buffer audit — any buffer named like an empirical quantile is flagged
    state = payload.get("model_state_dict", {})
    for k in state.keys():
        if any(b in k for b in forbidden_substrings):
            findings.append(f"  buffer {k} (shape {tuple(state[k].shape)})")
            offenders += 1
    ckpt_ok = offenders == 0
    return ckpt_ok, findings


def main() -> None:
    parser = argparse.ArgumentParser(description="250-series Bitter-Lesson audit")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    print("=" * 72)
    print("Bitter-Lesson audit — diffusion/block_ar/neural_factor.py")
    print("=" * 72)
    src_ok, src_findings = audit_source()
    if src_ok:
        print("  SOURCE: PASS — no per-factor-family literals in executable code.")
    else:
        print("  SOURCE: FAIL — literals in code body:")
        for f in src_findings:
            print(f)

    ckpt_ok = True
    if args.checkpoint is not None:
        ckpt_ok, ckpt_findings = audit_checkpoint(args.checkpoint)
        print(f"\nCheckpoint audit — {args.checkpoint}")
        if ckpt_ok:
            print("  CHECKPOINT: PASS — no per-cell empirical constants in config/state.")
        else:
            print("  CHECKPOINT: FAIL — offending keys:")
            for f in ckpt_findings:
                print(f)

    overall = src_ok and ckpt_ok
    print("=" * 72)
    print(f"Overall: {'PASS' if overall else 'FAIL'}")
    print("=" * 72)
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
