#!/usr/bin/env python
"""
Fixed smoke-gate evaluation for 207b.

This reuses the 207a gate because the architecture and success criteria are unchanged;
only the family supervision changes.
"""

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.analyze_207a_smoke_gate import main


if __name__ == "__main__":
    main()
