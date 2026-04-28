#!/usr/bin/env python
"""658a: evaluate AR-652 native joint checkpoints with the 641a harness."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.evaluate_641a_state_conditioned_mixed_coordinate_flow import (
    main,
)


if __name__ == "__main__":
    main()
