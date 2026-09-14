"""Expose scalar marginal PPC views of an HSSM 0.5 flat DDM artifact.

The saved RT/choice pair remains authoritative. These views deliberately contain
no posterior or log_likelihood: a joint trial density is not either marginal
likelihood, and marginal PPC-PIT is not a joint or held-out calibration claim.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

RESPONSE = "rt,response"
OBSERVATION_DIM = "__obs__"
COMPONENT_DIMS = {"rt,response_dim", "rt,response_extra_dim_0"}


def response_components(
    array: xr.DataArray, *, predictive: bool
) -> tuple[xr.DataArray, xr.DataArray]:
    """Validate the narrow positive-RT, choices {-1,+1} release layout.

    Observed and predictive event dimensions may have different names. Both
    must explicitly label columns [0, 1], meaning [RT in seconds, response].
    The caller must establish units; a positive number alone does not prove
    that RTs were recorded in seconds.
    """
    base_dims = {OBSERVATION_DIM, "chain", "draw"} if predictive else {OBSERVATION_DIM}
    event_dims = set(array.dims) - base_dims
    if len(event_dims) != 1 or not event_dims <= COMPONENT_DIMS:
        raise ValueError(f"Unexpected RT/choice dimensions: {array.dims}")
    (event_dim,) = event_dims
    if set(array.dims) != base_dims | {event_dim}:
        raise ValueError(f"Missing trial/sample dimensions: {array.dims}")
    for dim in array.dims:
        if array.sizes[dim] == 0 or dim not in array.coords:
            raise ValueError(f"Empty or unlabeled dimension: {dim}")
        if not array.get_index(dim).is_unique:
            raise ValueError(f"Duplicate coordinate labels: {dim}")
    if array.sizes[event_dim] != 2 or list(array[event_dim].values) != [0, 1]:
        raise ValueError("Expected labeled RT/choice component columns [0, 1].")
    rt = array.sel({event_dim: 0}, drop=True).rename("rt")
    response = array.sel({event_dim: 1}, drop=True).rename("response")
    if not np.isfinite(rt.values).all() or not (rt.values > 0).all():
        raise ValueError(
            "Only finite positive RTs are supported; no missing/deadline sentinels."
        )
    if not np.isin(response.values, [-1, 1]).all():
        raise ValueError("Only declared response labels {-1, +1} are supported.")
    order = ("chain", "draw", OBSERVATION_DIM) if predictive else (OBSERVATION_DIM,)
    return rt.transpose(*order), response.transpose(*order)


def scalar_ppc_views(dt: xr.DataTree) -> dict[str, xr.DataTree]:
    """Return isolated RT and +1-choice marginal views without altering ``dt``."""
    for group in ("observed_data", "posterior_predictive"):
        if group not in dt.children or RESPONSE not in dt[group].data_vars:
            raise ValueError(f"Missing {group}/{RESPONSE} in the fitted-data artifact.")
    observed_rt, observed_response = response_components(
        dt["observed_data"][RESPONSE], predictive=False
    )
    predicted_rt, predicted_response = response_components(
        dt["posterior_predictive"][RESPONSE], predictive=True
    )
    # This rejects reordered or missing rows rather than silently aligning or
    # trimming them. Prediction grids must not masquerade as fitted data.
    if not observed_rt.get_index(OBSERVATION_DIM).equals(
        predicted_rt.get_index(OBSERVATION_DIM)
    ):
        raise ValueError(
            "Observed and predictive trial coordinates must match exactly."
        )
    observed_choice = (observed_response == 1).astype("int8").rename("choice")
    predicted_choice = (predicted_response == 1).astype("int8").rename("choice")
    quantities = {
        "rt": (observed_rt, predicted_rt, "Marginal reaction time in seconds"),
        "choice": (observed_choice, predicted_choice, "Marginal event response == +1"),
    }
    views = {}
    for name, (observed, predicted, meaning) in quantities.items():
        views[name] = xr.DataTree.from_dict(
            {
                "observed_data": observed.to_dataset().copy(deep=True),
                "posterior_predictive": predicted.to_dataset().copy(deep=True),
            }
        )
        views[name].attrs.update(
            quantity=meaning,
            source_response=RESPONSE,
            applicability="Fitted-data marginal PPC-PIT only; not joint, conditional, or held-out calibration",
        )
    return views


def save_scalar_views(dt: xr.DataTree, output_dir: Path) -> dict:
    """Write two portable scalar artifacts and a description of their scope."""
    views = scalar_ppc_views(dt)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, view in views.items():
        directory = output_dir / name
        directory.mkdir(exist_ok=True)
        view.to_netcdf(directory / "inference_data.nc")
    manifest = {
        "source_response": RESPONSE,
        "component_order": ["rt_seconds", "response"],
        "choices": [-1, 1],
        "n_observations": dt["observed_data"][RESPONSE].sizes[OBSERVATION_DIM],
        "views": {
            "rt": {
                "artifact": "rt/inference_data.nc",
                "quantity": "Marginal RT in seconds",
            },
            "choice": {
                "artifact": "choice/inference_data.nc",
                "quantity": "Marginal event response == +1",
            },
        },
        "limitations": [
            "Fitted-data PPC-PIT reuses observations; it is not held-out validation.",
            "Two calibrated marginals do not establish the joint RT/choice distribution.",
            "Use choice proportions and conditional RT quantiles as additional domain checks.",
            "No marginal log-likelihood is supplied; do not run LOO-PIT on these views.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    with xr.open_datatree(args.idata) as dt:
        save_scalar_views(dt, args.output_dir)


if __name__ == "__main__":
    main()
