#!/usr/bin/env python
"""658a: AR-652 native joint flow with typed IV/factor transition heads."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_641a_state_conditioned_mixed_coordinate_flow import (
    main,
)


def _default_multihead_args() -> None:
    if "--head_mode" not in sys.argv:
        sys.argv.extend(["--head_mode", "multihead"])
    if "--seed" not in sys.argv:
        sys.argv.extend(["--seed", "658"])


if __name__ == "__main__":
    _default_multihead_args()
    main()
