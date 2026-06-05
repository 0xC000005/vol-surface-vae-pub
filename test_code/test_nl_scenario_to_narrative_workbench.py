from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (
    ScenarioSidecarV1,
    normalize_factor_table_csv_text,
)


def test_factor_table_requires_numeric_start_and_end() -> None:
    csv_text = "factor,start,confidence\nSPX,1294,medium\n"

    with pytest.raises(ValueError, match="missing required column"):
        normalize_factor_table_csv_text(csv_text, scenario_id="demo_upload")


def test_factor_table_normalizes_numeric_rows() -> None:
    csv_text = (
        "factor,start,end,confidence\n"
        "SPX,1294.0,1311.0,medium\n"
        "DXY,90.3,87.2,high\n"
        "BBB_OAS,1.42,1.50,medium\n"
    )

    sidecar = normalize_factor_table_csv_text(csv_text, scenario_id="demo_upload")

    assert isinstance(sidecar, ScenarioSidecarV1)
    assert sidecar.scenario_id == "demo_upload"
    assert sidecar.scenario_type == "factor_table_partial"
    assert sidecar.horizon_days == 30
    rows = {row.factor: row for row in sidecar.factor_rows}
    assert rows["SPX"].delta == pytest.approx(17.0)
    assert rows["SPX"].direction == "up"
    assert rows["DXY"].direction == "down"
    assert rows["BBB_OAS"].direction == "wider"
    assert rows["SPX"].evidence == "start=1294; end=1311; delta=17"
