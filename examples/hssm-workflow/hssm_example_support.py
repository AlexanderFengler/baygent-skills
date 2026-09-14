"""Example-only plots and canonical report using the installed HSSM adapter.

HSSM construction, inference, density computation and domain summaries stay
visible in the notebook. Diagnostic ratings come only from bayesian-workflow.
This module is not required by the standalone HSSM skill.
"""

import json
import os
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import arviz as az
import arviz_plots as azp
import arviz_stats as azs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BAYESIAN = REPO / "bayesian-workflow"
sys.path.insert(0, str(REPO / "hssm-workflow" / "scripts"))
from prepare_rt_choice import save_scalar_views


def result_directory(slug: str) -> Path:
    """Create the notebook output directory after the explicit run gate."""
    root = Path(
        os.environ.get("BAYGENT_OUTPUT_ROOT", Path(__file__).parent / "results")
    )
    directory = root / slug
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _dump(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n"
    )


def _table(frame: pd.DataFrame) -> str:
    """Render a small table without adding a markdown dependency."""

    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.4g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = [list(frame.columns), ["---"] * len(frame.columns)]
    rows.extend(frame.itertuples(index=False, name=None))
    return "\n".join("| " + " | ".join(map(cell, row)) + " |" for row in rows)


def _run_helper(name: str, arguments: list[str], output: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, str(BAYESIAN / "scripts" / name), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    if completed.returncode:
        raise RuntimeError(
            f"{name} failed ({completed.returncode}):\n{completed.stderr}\n{completed.stdout}"
        )
    return json.loads(output.read_text())


def _canonical_report(sections: dict[str, tuple[str, str | None]], title: str) -> str:
    """Fill the dependency's canonical template, retaining its explanatory prose.

    Each supported section has one narrative placeholder; dynamic tables are
    replaced as units. Unexpected template changes fail visibly rather than
    leaving a seemingly complete report with unfilled fields.
    """
    source = (BAYESIAN / "references" / "reporting.md").read_text()
    template = source.split("````markdown\n", 1)[1].split("````", 1)[0]
    # Modern ArviZ plot_trace shows draw order, not the template's legacy rank view.
    template = template.replace(
        "Rank-vline trace plots check chain mixing. Well-mixed chains show overlapping rank distributions across chains — the vertical lines (one per chain) sit close to the uniform expectation.",
        "Trace plots show parameter draws in sampling order for each chain. Well-mixed chains explore similar ranges without persistent drift or sticking.",
    )
    template = template.replace(
        "Lines outside the bands above the diagonal indicate under-confident predictions (intervals wider than they should be); lines below indicate over-confident predictions (intervals too narrow).",
        "Departures from uniform PIT values can reflect location bias or dispersion errors; their shape matters. Use the separate coverage curve to assess interval coverage. These are fitted-data PPC-PIT checks, not held-out validation.",
    )
    parts = re.split(r"(?m)^## (.+)\n", template)
    output = [f"# {title} — Bayesian Analysis Report\n"]
    for heading, body in zip(parts[1::2], parts[2::2], strict=True):
        if heading.startswith("[IF MODEL_COMPARISON]"):
            continue  # These examples fit one model, not a model-comparison exercise.
        heading = heading.removeprefix("[IF SENSITIVITY] ")
        narrative, table = sections[heading]
        if table is not None:
            body = re.sub(
                r"(?m)^\|[^\n]*(?:\n\|[^\n]*)*",
                lambda _, replacement=table: replacement,
                body,
            )
        # The limits and next-steps blocks include placeholder numbered items.
        if heading in {"Limitations and Threats", "Suggested Next Steps"}:
            body = re.sub(r"(?m)^\d+\. <[^\n]+>\n?", "", body)
        placeholders = re.findall(r"<[^\n]+>", body)
        if heading == "Limitations and Threats":
            body = body.rstrip() + "\n\n" + narrative + "\n"
        elif len(placeholders) == 1:
            body = body.replace(placeholders[0], narrative)
        elif len(placeholders) == 0 and heading == "Suggested Next Steps":
            body = narrative + "\n"
        else:
            raise ValueError(f"Review canonical template placeholders in {heading}")
        output.append(f"## {heading}\n{body.rstrip()}\n")
    return "\n".join(output)


def write_report(
    model,
    prior,
    posterior,
    output_dir: Path,
    *,
    details: dict,
    domain_checks: pd.DataFrame,
) -> dict:
    """Save original joint output, assess two margins, and assemble the report."""
    output_dir = Path(output_dir)
    required_figures = (
        "model_graph.png",
        "prior_predictive.png",
        "posterior_predictive.png",
        "quantile_probability.png",
    )
    for name in required_figures:
        if not (output_dir / name).is_file():
            raise ValueError(f"Notebook has not produced required figure: {name}")
    if details.get("prior_density_verified") is not True:
        raise ValueError(
            "Verify the effective scalar prior densities before sensitivity assessment."
        )
    free_names = {rv.name for rv in model.pymc_model.free_RVs}
    if free_names != set(posterior["log_prior"].data_vars):
        raise ValueError(
            "The original joint artifact must contain every free-variable prior density."
        )
    if any(
        not np.isfinite(array.values).all()
        for array in posterior["log_prior"].data_vars.values()
    ):
        raise ValueError("Non-finite prior densities require investigation.")
    likelihood = posterior["log_likelihood"]["rt,response"]
    if set(likelihood.dims) != {"chain", "draw", "__obs__"}:
        raise ValueError("Expected one joint RT/choice log likelihood per trial.")
    if not np.isfinite(likelihood.values).all():
        raise ValueError(
            "Non-finite trial likelihoods require investigation before reporting."
        )
    posterior.to_netcdf(output_dir / "inference_data.nc")
    prior.to_netcdf(output_dir / "prior_data.nc")
    model.data.to_csv(output_dir / "data.csv", index=False)
    domain_checks.to_csv(output_dir / "domain_checks.csv", index=False)
    summary = az.summary(
        posterior, var_names=["v", "a", "z", "t"], ci_prob=0.94, ci_kind="hdi"
    )
    summary.to_csv(output_dir / "summary.csv")
    psense = azs.psense_summary(posterior, var_names=["v", "a", "z", "t"])
    _dump(output_dir / "psense.json", psense.to_dict(orient="index"))
    figure_kwargs = {"figsize": (10, 10), "layout": "constrained"}
    az.plot_trace(
        posterior, var_names=["v", "a", "z", "t"], figure_kwargs=figure_kwargs
    ).savefig(output_dir / "trace.png")
    az.plot_forest(
        posterior,
        var_names=["v", "a", "z", "t"],
        combined=True,
        ci_probs=[0.5, 0.94],
        ci_kind="hdi",
        point_estimate="median",
    ).savefig(output_dir / "forest.png")
    azp.plot_psense_dist(
        posterior, var_names=["v", "a", "z", "t"], figure_kwargs=figure_kwargs
    ).savefig(output_dir / "psense.png")
    plt.close("all")

    # LOO sees the original joint density, never either scalar marginal view.
    diagnostic_path = output_dir / "diagnostics.json"
    diagnostics = _run_helper(
        "diagnose_model.py",
        [
            "--idata",
            str(output_dir / "inference_data.nc"),
            "--output",
            str(diagnostic_path),
        ],
        diagnostic_path,
    )
    if not diagnostics.get("convergence"):
        raise ValueError("The shared helper did not return convergence evidence.")
    check_path = output_dir / "check_report.json"
    checks = _run_helper(
        "check_diagnostics.py",
        [
            "--diagnostics",
            str(diagnostic_path),
            "--psense",
            str(output_dir / "psense.json"),
            "--output",
            str(check_path),
        ],
        check_path,
    )

    manifest = save_scalar_views(posterior, output_dir / "calibration")
    margins = {}
    for name in ("rt", "choice"):
        directory = output_dir / "calibration" / name
        cal_path = directory / "calibration.json"
        calibration = _run_helper(
            "calibration_check.py",
            [
                "--idata",
                str(directory / "inference_data.nc"),
                "--var-name",
                name,
                "--output",
                str(cal_path),
                "--save-plots",
                "--plot-dir",
                str(directory),
            ],
            cal_path,
        )
        if (
            not calibration.get("assessment")
            or calibration.get("pit_method") != "ppc_pit"
        ):
            raise ValueError(f"Missing marginal PPC-PIT assessment for {name}.")
        calibration["plots"] = {
            key: Path(value).name for key, value in calibration["plots"].items()
        }
        _dump(cal_path, calibration)
        margin_check_path = directory / "check_report.json"
        margins[name] = _run_helper(
            "check_diagnostics.py",
            [
                "--diagnostics",
                str(diagnostic_path),
                "--calibration",
                str(cal_path),
                "--output",
                str(margin_check_path),
            ],
            margin_check_path,
        )

    versions = {
        name: version(name)
        for name in (
            "hssm",
            "bambi",
            "pymc",
            "pytensor",
            "ssm-simulators",
            "arviz",
            "arviz-stats",
            "arviz-plots",
            "numpy",
            "marimo",
        )
    }
    versions["python"] = sys.version.split()[0]
    _dump(output_dir / "versions.json", versions)
    _dump(output_dir / "analysis.json", details)
    prior_rows = [
        {"Parameter": name, "Resolved prior": str(parameter.prior)}
        for name, parameter in model.params.items()
    ]
    conv = diagnostics["convergence"]
    diagnostic_rows = [
        {
            "Diagnostic": label,
            "Value": conv[key].get(value, "Not returned by shared helper"),
            "Status": "pass" if conv[key]["ok"] else "flag",
        }
        for label, key, value in [
            ("Max R-hat", "rhat", "max"),
            ("Min ESS (bulk)", "ess_bulk", "min"),
            ("Min ESS (tail)", "ess_tail", "min"),
            ("Divergences", "divergences", "count"),
        ]
    ]
    limits = details["limitations"]
    if isinstance(limits, str):
        limits = [limits]
    limits = [*limits, *manifest["limitations"]]
    calibration_text = (
        "**Marginal RT:** "
        + margins["rt"]["summary"]["calibration"]
        + "\n\n**Marginal choice (+1):** "
        + margins["choice"]["summary"]["calibration"]
        + "\n\n![Choice PIT ECDF](calibration/choice/pit_ecdf.png)"
        + "\n\n![Choice coverage](calibration/choice/pit_coverage.png)"
        + "\n\nThese use separate scalar PPC views. Passing both does not establish joint RT/choice, conditional, or held-out calibration."
    )
    # Preserve raw shared suggestions in JSON; render DDM-specific actions
    # from those same ratings without introducing diagnostic thresholds.
    next_steps = []
    if checks["convergence"]["rating"] in {"fair", "poor"}:
        next_steps.append(
            "Inspect the HSSM trace and parameter geometry; verify RT units, response coding and t support. Adjust priors or the model where justified, and use HSSM.sample for any longer or revised fit before interpretation."
        )
    if checks["loo"]["rating"] in {"fair", "poor"}:
        next_steps.append(
            "Inspect influential joint RT/choice trials and their Pareto-k values. Check measurement validity and task-model assumptions; justify any alternative SSM or lapse treatment before refitting."
        )
    if checks["loo"]["rating"] == "not computed":
        next_steps.append(
            "Inspect diagnostics.json for the unavailable joint-trial LOO calculation; do not substitute marginal LOO-PIT from the scalar views."
        )
    if checks["psense"]["rating"] in {"moderate sensitivity", "strong sensitivity"}:
        next_steps.append(
            "Refit justified alternative priors on the flagged DDM parameters and compare domain predictions. Separately assess the data-informed t upper bound; power sensitivity cannot assess a different support."
        )
    for name, assessment in margins.items():
        if assessment["calibration"]["rating"] in {"fair", "poor"}:
            next_steps.append(
                f"Investigate the {name} marginal PPC-PIT discrepancy alongside choice proportions and conditional RT tails. Revisit RT units, t support, priors and the DDM task assumptions before considering a separately validated model extension."
            )
    next_steps.append(
        "Review choice proportions and conditional RT quantiles together, including absent-choice replicate counts. Interpret parameters only when the available convergence and domain checks support the intended claim."
    )
    sections = {
        "Executive Summary": (
            details["interpretation"]
            + " "
            + checks["summary"]["convergence"]
            + " "
            + checks["summary"]["psense"]
            + " This is a synthetic teaching analysis, not a real-data conclusion.",
            None,
        ),
        "Data and Question": (
            "RT is in seconds; responses -1/+1 refer to declared boundaries. No missing trials, deadlines, or lapse mixture are modeled.",
            _table(
                pd.DataFrame(
                    {
                        "Field": ["Source", "N", "Question"],
                        "Value": [
                            "Synthetic DDM teaching data",
                            len(model.data),
                            details["question"],
                        ],
                    }
                )
            ),
        ),
        "Model Specification": (
            details["generative_story"]
            + "\n\n"
            + str(details["prior_policy"])
            + "\n\n```text\n"
            + str(model)
            + "\n```",
            _table(pd.DataFrame(prior_rows)),
        ),
        "Prior Predictive Check": (
            details["prior_assessment"]
            + " Conditional quantiles omit replicates without that choice; their counts are reported below.\n\n"
            + _table(domain_checks.loc[domain_checks.stage == "prior_predictive"]),
            None,
        ),
        "Sampling and Convergence": (
            checks["summary"]["convergence"]
            + " Settings: `"
            + str(details["sampler_settings"])
            + "`.",
            _table(pd.DataFrame(diagnostic_rows)),
        ),
        "Posterior": (
            details["interpretation"]
            + " The table reports parameter means and 94% HDIs; these do not identify a unique psychological explanation or a causal effect.",
            _table(summary.reset_index(names="Parameter")),
        ),
        "Posterior Predictive Check": (
            "Compare both choices and their conditional RT distributions. The following are replicated-statistic intervals, not parameter HDIs.\n\n"
            + _table(domain_checks.loc[domain_checks.stage == "posterior_predictive"])
            + "\n\n![RT quantiles by choice](quantile_probability.png)",
            None,
        ),
        "Calibration": (calibration_text, None),
        "Prior Sensitivity": (
            checks["summary"]["psense"],
            _table(psense.reset_index(names="Parameter")),
        ),
        "Limitations and Threats": (
            "\n".join(f"{index}. {value}" for index, value in enumerate(limits, 1)),
            None,
        ),
        "Suggested Next Steps": (
            "\n".join(f"{index}. {value}" for index, value in enumerate(next_steps, 1)),
            None,
        ),
        "Appendix": (
            "Joint-trial LOO: "
            + checks["summary"]["loo"]
            + f"\n\nDescriptive seed: `{details['seed']}`. Predictive RNG settings: `{details['predictive_settings']}`."
            + "\n\nArtifacts: `inference_data.nc` (joint fit), `prior_data.nc`, `domain_checks.csv`, `summary.csv`, and the scoped `calibration/` outputs."
            + "\n\n"
            + _table(pd.DataFrame(versions.items(), columns=["Package", "Version"])),
            None,
        ),
    }
    report = _canonical_report(sections, details["title"])
    report = report.replace("(pit_ecdf.png)", "(calibration/rt/pit_ecdf.png)").replace(
        "(pit_coverage.png)", "(calibration/rt/pit_coverage.png)"
    )
    (output_dir / "report.md").write_text(report)
    return {
        "checks": checks,
        "marginal_checks": margins,
        "next_steps": next_steps,
        "summary": summary,
        "domain_checks": domain_checks,
    }
