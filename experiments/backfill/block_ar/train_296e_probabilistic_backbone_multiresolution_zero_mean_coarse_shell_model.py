#!/usr/bin/env python
from __future__ import annotations

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_296c_probabilistic_backbone_profiled_zero_mean_coarse_shell_model import (
    main as run_296c_main,
)


def main() -> None:
    if "--knot_positions" not in sys.argv[1:]:
        sys.argv.extend(["--knot_positions", "1,2,3,5,10,15,22,30"])
    run_296c_main()


if __name__ == "__main__":
    main()
