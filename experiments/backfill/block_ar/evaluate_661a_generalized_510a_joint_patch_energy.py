#!/usr/bin/env python
"""661a: evaluate generalized 510a-style joint AR patch-energy checkpoints."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.evaluate_609a_unified_ar_transition_flow import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    main()
