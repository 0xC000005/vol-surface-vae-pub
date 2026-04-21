#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOAL_PATH = ROOT / "autoresearch-session" / "goal_11x11.json"
STATE_PATH = ROOT / "autoresearch-session" / "state_11x11.json"


def main() -> int:
    goal = json.loads(GOAL_PATH.read_text(encoding="utf-8"))
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))

    stop_path = ROOT / goal.get("stop_file", "autoresearch-session/STOP")
    stop_requested = stop_path.exists()
    current_best = int(state.get("current_best_n_pass", 0))
    target = int(goal.get("target_n_pass", 11))
    goal_reached = bool(state.get("goal_reached", False)) or current_best >= target

    status = {
        "target_n_pass": target,
        "current_best_n_pass": current_best,
        "goal_reached": goal_reached,
        "stop_requested": stop_requested,
        "active_family": state.get("active_family"),
        "last_iteration_type": state.get("last_iteration_type"),
        "last_commit": state.get("last_commit")
    }
    print(json.dumps(status, indent=2))

    if stop_requested or goal_reached:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
