"""
Calibration assessment for Bayesian models.

Computes coverage calibration, PIT ECDFs, and generates calibration plots
using ArviZ 1.0+ (arviz_plots). Supports both PPC-PIT and LOO-PIT.

Usage:
    python calibration_check.py --idata path/to/inference_data.nc
    python calibration_check.py --idata path/to/inference_data.nc --var-name obs --save-plots
    python calibration_check.py --idata path/to/inference_data.nc --loo-pit --save-plots
    python calibration_check.py --idata path/to/inference_data.nc --save-plots --plot-dir plots/
"""

import argparse
import inspect
import json
import os
import sys
import warnings

import numpy as np
import xarray as xr

try:
    import arviz_plots as azp
    import arviz_stats as azs
    from arviz_base import convert_to_datatree
    from arviz_stats.ecdf_utils import ecdf_pit

except ImportError:
    print(
        json.dumps(
            {
                "error": (
                    "arviz_plots and arviz_base are required. "
                    "Install with: pip install arviz-plots arviz-base"
                )
            }
        )
    )
    sys.exit(1)

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    # ArviZ's native PPC-PIT preprocessing includes its Pareto tail refinement.
    from arviz_plots.plots.utils_ppc import get_ppc_pit
except ImportError:  # arviz-plots 1.0 uses empirical PIT and envelopes
    get_ppc_pit = None


def _has_modern_pit() -> bool:
    plot = getattr(azp, "plot_ecdf_pit", None)
    return (
        callable(get_ppc_pit)
        and callable(plot)
        and "method" in inspect.signature(plot).parameters
        and hasattr(xr.Dataset().azstats, "uniformity_test")
    )


def _resolve_method(method: str) -> str:
    if method not in {"auto", "pot_c", "envelope"}:
        raise ValueError(f"Unknown uniformity method: {method!r}.")
    modern = _has_modern_pit()
    if method == "auto":
        return "pot_c" if modern else "envelope"
    if method == "pot_c" and not modern:
        raise ValueError("pot_c requires modern ArviZ PIT and uniformity-test APIs.")
    return method


def prepare_pit_values(dt, var_name, use_loo=False, method="auto") -> xr.Dataset:
    """Prepare one labeled raw PIT dataset, shared by assessment and plotting."""
    method = _resolve_method(method)
    predictive = dt["posterior_predictive"][var_name]
    observed = dt["observed_data"][var_name]
    if set(predictive.dims) != set(observed.dims) | {"chain", "draw"}:
        raise ValueError(
            "Predictive dimensions must match observed dimensions plus chain/draw."
        )
    if "chain" in observed.dims or "draw" in observed.dims:
        raise ValueError("Observed data cannot have chain/draw dimensions.")
    xr.align(predictive, observed, join="exact")
    for label, array in (("predictive", predictive), ("observed", observed)):
        if not array.size or not np.isfinite(array.values).all():
            raise ValueError(f"{label} data must be nonempty and finite.")
    if use_loo:
        kwargs = {"var_names": [var_name]}
        if "pareto_pit" in inspect.signature(azs.loo_pit).parameters:
            kwargs["pareto_pit"] = method == "pot_c"
        pit = azs.loo_pit(dt, **kwargs)[[var_name]]
    elif method == "pot_c":
        pit = get_ppc_pit(
            dt["posterior_predictive"].ds[[var_name]],
            dt["observed_data"].ds[[var_name]],
            ["chain", "draw"],
            coverage=False,
            method=method,
        )["ecdf_pit"].ds
    else:
        values = (predictive <= observed).mean(("chain", "draw"))
        if predictive.dtype.kind in "biu" or observed.dtype.kind in "biu":
            less = (predictive < observed).mean(("chain", "draw"))
            uniforms = np.random.default_rng(214).uniform(size=values.shape)
            # Preserve the legacy ArviZ 1.0 tie-breaking convention exactly.
            values = uniforms * less + (1 - uniforms) * values
        pit = values.rename(var_name).to_dataset()
    values = pit[var_name].values
    if (
        values.size == 0
        or not np.isfinite(values).all()
        or not ((values >= 0) & (values <= 1)).all()
    ):
        raise ValueError("PIT values must be finite, nonempty and lie in [0, 1].")
    return pit


def _evaluate_pit(pit_values, var_name, method, ci_prob):
    """Return the exact curve and uniformity evidence used by assessment and plots."""
    values = pit_values[var_name].values.ravel()
    if (
        values.size == 0
        or not np.isfinite(values).all()
        or not ((values >= 0) & (values <= 1)).all()
    ):
        raise ValueError("PIT values must be finite, nonempty and lie in [0, 1].")
    if not 0 < ci_prob < 1:
        raise ValueError("ci_prob must lie strictly between 0 and 1.")
    if method == "envelope":
        x, _, lower, upper = ecdf_pit(values, ci_prob, n_simulations=1000)
        # Count real PIT atoms at zero in the curve. Native ecdf_pit pads its
        # endpoints with zeros; those are not part of its envelope test.
        delta = np.searchsorted(np.sort(values), x, side="right") / values.size - x
        lower, upper = lower - x, upper - x
        passed = bool(
            ((delta[1:-1] >= lower[1:-1]) & (delta[1:-1] <= upper[1:-1])).all()
        )
        lower[[0, -1]] = upper[[0, -1]] = np.nan
        p_value = None
    else:
        # Use actual probability coordinates. ArviZ's plotting ECDF currently
        # rescales its grid by max(PIT), which can shift a genuine PIT curve.
        x = np.linspace(0, 1, values.size + 1)
        delta = np.searchsorted(np.sort(values), x, side="right") / values.size - x
        p_values, _, _ = pit_values.azstats.uniformity_test(
            dim=list(pit_values[var_name].dims), method=method
        )
        p_value = float(p_values[var_name].item())
        if not np.isfinite(p_value) or not 0 <= p_value <= 1:
            raise ValueError("ArviZ returned an invalid uniformity p-value.")
        passed = p_value >= 1 - ci_prob
        lower = upper = None
    return {
        "x": x,
        "delta": delta,
        "lower": lower,
        "upper": upper,
        "p_value": p_value,
        "passed": passed,
        "mean_delta": round(float(np.mean(delta)), 4),
    }


def assess_calibration(
    dt, var_name, use_loo, ci_prob=0.99, *, method="auto", pit_values=None
):
    """Assess raw PIT and once-transformed central coverage with one named method.

    Modern ArviZ uses its native ``pot_c`` uniformity test. Older stacks use
    simultaneous ECDF envelopes. The mean coverage deviation is a descriptive
    direction statistic; a small mean does not override a failed shape test.
    Pass the same prepared ``pit_values`` to plotting to share discrete tie draws.
    """
    method = _resolve_method(method)
    if pit_values is None:
        pit_values = prepare_pit_values(dt, var_name, use_loo, method)
    pit = _evaluate_pit(pit_values, var_name, method, ci_prob)
    coverage = _evaluate_pit(2 * np.abs(pit_values - 0.5), var_name, method, ci_prob)
    mean_cov_delta = coverage["mean_delta"]
    well_calibrated = pit["passed"] and coverage["passed"]
    if mean_cov_delta > 0.02:
        diagnosis = "under-confident (predictions too uncertain)"
    elif mean_cov_delta < -0.02:
        diagnosis = "over-confident (predictions too certain)"
    elif well_calibrated:
        diagnosis = "well-calibrated"
    else:
        diagnosis = "non-uniform predictive PIT; no dominant mean coverage direction"
    return {
        "uniformity_method": method,
        "significance_level": 1 - ci_prob,
        "pit_p_value": pit["p_value"],
        "coverage_p_value": coverage["p_value"],
        "pit_uniformity_passed": pit["passed"],
        "coverage_uniformity_passed": coverage["passed"],
        # No confidence bands are computed by pot_c: null is not a passing band.
        "pit_ecdf_inside_bands": pit["passed"] if method == "envelope" else None,
        "coverage_ecdf_inside_bands": coverage["passed"]
        if method == "envelope"
        else None,
        "well_calibrated": well_calibrated,
        "mean_coverage_deviation": mean_cov_delta,
        "calibration_diagnosis": diagnosis,
    }


def save_pit_plot(
    dt,
    var_name,
    output_path,
    *,
    use_loo=False,
    coverage=False,
    ci_prob=0.99,
    method="auto",
    pit_values=None,
):
    """Plot the exact tested ECDF on its unscaled probability axis.

    Rendering the native statistics directly avoids the PPC wrapper's repeated
    coverage transform and plotting ECDF's maximum normalization. Only legacy
    envelope mode has simultaneous confidence bands; pot_c displays its p-value.
    """
    import matplotlib.pyplot as plt

    method = _resolve_method(method)
    if pit_values is None:
        pit_values = prepare_pit_values(dt, var_name, use_loo, method)
    if coverage:
        pit_values = 2 * np.abs(pit_values - 0.5)
    evidence = _evaluate_pit(pit_values, var_name, method, ci_prob)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.axhline(0, color="0.4", linewidth=1)
    if method == "envelope":
        ax.fill_between(
            evidence["x"],
            evidence["lower"],
            evidence["upper"],
            color="C0",
            alpha=0.2,
            label=f"{ci_prob:.0%} simultaneous envelope",
        )
        annotation = f"envelope: {'pass' if evidence['passed'] else 'fail'}"
    else:
        annotation = f"pot_c: p={evidence['p_value']:.3g}, α={1 - ci_prob:.3g}"
    ax.step(evidence["x"], evidence["delta"], where="post", color="C0")
    ax.text(0.02, 0.98, annotation, transform=ax.transAxes, va="top")
    ax.set(
        xlim=(0, 1),
        ylabel="ΔECDF",
        title=f"{var_name}: {'LOO-PIT' if use_loo else 'PPC-PIT'}",
    )
    if coverage:
        ax.set_xlabel("Nominal central predictive coverage (%)")
        ax.set_xticks(np.linspace(0, 1, 5), ["0", "25", "50", "75", "100"])
    else:
        ax.set_xlabel("PIT")
    fig.tight_layout()
    try:
        fig.savefig(output_path)
    finally:
        plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Bayesian model calibration check")
    parser.add_argument(
        "--idata", required=True, help="Path to InferenceData (.nc file)"
    )
    parser.add_argument(
        "--var-name",
        default=None,
        help="Name of the observed variable (auto-detected if not specified)",
    )
    parser.add_argument("--output", default=None, help="Path to save JSON report")
    parser.add_argument(
        "--save-plots", action="store_true", help="Save calibration plots"
    )
    parser.add_argument(
        "--loo-pit",
        action="store_true",
        help="Use LOO-PIT instead of PPC-PIT (requires log_likelihood group)",
    )
    parser.add_argument(
        "--plot-dir", default=".", help="Directory for saved plots (default: .)"
    )
    parser.add_argument(
        "--ci-prob",
        type=float,
        default=0.99,
        help="One minus the uniformity significance level (default: 0.99)",
    )
    parser.add_argument(
        "--uniformity-method",
        choices=("auto", "pot_c", "envelope"),
        default="auto",
        help="auto selects native pot_c when supported, otherwise legacy envelopes",
    )
    args = parser.parse_args()

    try:
        dt = convert_to_datatree(args.idata)
    except Exception as e:  # noqa: BLE001 -- CLI boundary reports loader failures
        print(json.dumps({"error": f"Could not load InferenceData: {e}"}))
        sys.exit(1)

    # Validate data availability
    if "posterior_predictive" not in dt.children:
        print(
            json.dumps(
                {
                    "error": "No posterior_predictive group. Run pm.sample_posterior_predictive() first."
                }
            )
        )
        sys.exit(1)

    if "observed_data" not in dt.children:
        print(
            json.dumps({"error": "No observed_data group. Cannot compute calibration."})
        )
        sys.exit(1)

    # Auto-detect var_name if not specified
    var_name = args.var_name
    if var_name is None:
        pp_vars = set(dt["posterior_predictive"].data_vars)
        obs_vars = set(dt["observed_data"].data_vars)
        common = sorted(pp_vars & obs_vars)
        if not common:
            print(
                json.dumps(
                    {
                        "error": (
                            f"No common variables between posterior_predictive {sorted(pp_vars)} "
                            f"and observed_data {sorted(obs_vars)}."
                        )
                    }
                )
            )
            sys.exit(1)
        var_name = common[0]
        if len(common) > 1:
            print(
                f"Warning: multiple common variables found: {common}. Using '{var_name}'.",
                file=sys.stderr,
            )

    if var_name not in dt["posterior_predictive"].data_vars:
        available = list(dt["posterior_predictive"].data_vars)
        print(
            json.dumps(
                {
                    "error": f"Variable '{var_name}' not found in posterior_predictive. Available: {available}"
                }
            )
        )
        sys.exit(1)

    if var_name not in dt["observed_data"].data_vars:
        print(
            json.dumps(
                {
                    "error": f"No observed data for '{var_name}'. Cannot compute calibration."
                }
            )
        )
        sys.exit(1)

    # Validate LOO-PIT requirements
    if args.loo_pit and "log_likelihood" not in dt.children:
        print(
            json.dumps(
                {
                    "error": (
                        "LOO-PIT requires a log_likelihood group in the InferenceData. "
                        "Re-run sampling with idata_kwargs={'log_likelihood': True}, or "
                        "run `pm.compute_log_likelihood(idata, model_instance)."
                    )
                }
            )
        )
        sys.exit(1)

    ci_prob = args.ci_prob
    method = _resolve_method(args.uniformity_method)
    pit_values = prepare_pit_values(dt, var_name, args.loo_pit, method)
    assessment = assess_calibration(
        dt,
        var_name,
        use_loo=args.loo_pit,
        ci_prob=ci_prob,
        method=method,
        pit_values=pit_values,
    )

    n_obs = len(dt["observed_data"][var_name].values)
    report = {
        "variable": var_name,
        "n_observations": n_obs,
        "pit_method": "loo_pit" if args.loo_pit else "ppc_pit",
        "uniformity_method": method,
        "coverage_transform": "2 * abs(PIT - 0.5), applied once",
        "assessment": assessment,
    }

    # Save plots using arviz_plots
    if args.save_plots:
        os.makedirs(args.plot_dir, exist_ok=True)
        prefix = "loo_pit" if args.loo_pit else "pit"
        report["plots"] = {
            "pit_ecdf": save_pit_plot(
                dt,
                var_name,
                os.path.join(args.plot_dir, f"{prefix}_ecdf.png"),
                use_loo=args.loo_pit,
                ci_prob=ci_prob,
                method=method,
                pit_values=pit_values,
            ),
            "coverage": save_pit_plot(
                dt,
                var_name,
                os.path.join(args.plot_dir, f"{prefix}_coverage.png"),
                use_loo=args.loo_pit,
                coverage=True,
                ci_prob=ci_prob,
                method=method,
                pit_values=pit_values,
            ),
        }

    output = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
