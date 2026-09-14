"""Literal report display regressions; no model construction or fitting."""

import copy
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def reporting():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/bambi-workflow/example_support.py"
    )
    spec = importlib.util.spec_from_file_location("bambi_reporting_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("missing", ["absent", "null"])
@pytest.mark.parametrize("ok", [True, False])
def test_missing_extrema_use_summary_without_changing_shared_flags(
    reporting, missing, ok
):
    convergence = {
        "rhat": {"ok": ok},
        "ess_bulk": {"ok": ok},
        "ess_tail": {"ok": ok},
        "divergences": {"ok": True, "count": 0},
    }
    if missing == "null":
        for key, field in (("rhat", "max"), ("ess_bulk", "min"), ("ess_tail", "min")):
            convergence[key][field] = None
    before = copy.deepcopy(convergence)
    summary = pd.DataFrame(
        {
            "r_hat": [1.001, 1.01001, 1.004],
            "ess_bulk": [941.1, 721.23456, 862.4],
            "ess_tail": [831.7, 974.2, 651.78912],
        }
    )
    rows = reporting._convergence_rows(convergence, summary)
    assert [row["Value"] for row in rows] == [1.01001, 721.23456, 651.78912, 0]
    assert [row["Status"] for row in rows] == ["pass" if ok else "flag"] * 3 + ["pass"]
    assert convergence == before
    table = reporting._table(pd.DataFrame(rows))
    assert "None" not in table and "Not returned" not in table
    assert "| Max R-hat | 1.01 |" in table


def test_native_extrema_take_precedence_over_selected_summary(reporting):
    convergence = {
        "rhat": {"ok": False, "max": 1.15},
        "ess_bulk": {"ok": False, "min": 87.5},
        "ess_tail": {"ok": False, "min": 98.7},
        "divergences": {"ok": False, "count": 12},
    }
    summary = pd.DataFrame({"r_hat": [1.001], "ess_bulk": [999], "ess_tail": [999]})
    rows = reporting._convergence_rows(convergence, summary)
    assert [row["Value"] for row in rows] == [1.15, 87.5, 98.7, 12]
    assert all(row["Status"] == "flag" for row in rows)
