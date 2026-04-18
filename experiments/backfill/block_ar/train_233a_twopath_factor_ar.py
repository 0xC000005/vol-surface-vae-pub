#!/usr/bin/env python
"""
233a: Two-Path Factor AR — v1 + two ablation variants (v1-B, v1-C).

Spec: research/233a_twopath_v1/design.md

Variants (--variant flag):
  full  : two-path with learned slow state + Hawkes + FiLM + jump mixture
  B     : no slow path; aux-supervised cond_fast with parallel residual MLP
  C     : HAR-feature control (no learned state, 5 deterministic HAR features)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import namedtuple
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use existing 227a/226a infrastructure unchanged
from experiments.backfill.block_ar.train_227a_factor_ar import (
    FactorARModel as FactorARModel227a,
)

FastFeatures = namedtuple("FastFeatures", ["feature_vec", "local_scale"])


# --- stubs filled in by later tasks ---
class SlowPath(nn.Module):
    """Slow path: GRUSlow + hybrid (EWMA + Hawkes) analytic backbone + learned residuals."""
    pass


class FiLM(nn.Module):
    """Restricted FiLM coupling: (s, λ) -> (γ_Λ, β_Λ, γ_D, β_D, drift_bias, p_jump_logit)."""
    pass


class TwoPathFactorAR(nn.Module):
    """v1-full model. Subclasses FactorARModel227a and adds slow path + FiLM + jump mixture."""
    pass


if __name__ == "__main__":
    main()   # defined in later task
