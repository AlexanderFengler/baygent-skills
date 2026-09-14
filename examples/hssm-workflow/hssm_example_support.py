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
    (directory / "report.md").unlink(missing_ok=True)
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
    # A successful exit without a newly written output must never reuse an
    # earlier run's healthy assessment.
    output.unlink(missing_ok=True)
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
    try:
        result = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"{name} did not produce valid JSON evidence: {output}"
        ) from error
    if not isinstance(result, dict) or not result or "error" in result:
        raise ValueError(f"{name} returned missing or failed evidence: {result!r}")
    return result


def _validate_diagnostics(diagnostics: dict) -> None:
    """Reject absent convergence evidence before the shared defaults hide it."""
    convergence = diagnostics.get("convergence", {})
    if not isinstance(convergence, dict) or type(convergence.get("all_ok")) is not bool:
        raise ValueError("Missing convergence evidence from the shared helper.")
    for name in ("rhat", "ess_bulk", "ess_tail", "divergences"):
        item = convergence.get(name, {})
        if not isinstance(item, dict) or type(item.get("ok")) is not bool:
            raise ValueError(f"Missing convergence evidence for {name}.")
        if convergence["all_ok"] and not item["ok"]:
            raise ValueError("Contradictory shared convergence evidence.")
    loo = diagnostics.get("loo", {})
    if not isinstance(loo, dict) or type(loo.get("computed")) is not bool:
        raise ValueError("Missing joint-trial LOO status.")
    if not loo["computed"] and not loo.get("error"):
        raise ValueError("Unavailable joint-trial LOO needs an explicit reason.")
    if loo["computed"] and not loo.get("pareto_k"):
        raise ValueError("Computed joint-trial LOO needs Pareto-k evidence.")


def _validate_assessments(checks: dict, margins: dict) -> None:
    allowed = {
        "convergence": {"excellent", "good", "fair", "poor"},
        "loo": {"excellent", "fair", "poor", "not computed"},
        "psense": {
            "low sensitivity",
            "moderate sensitivity",
            "strong sensitivity",
            "not computed",
        },
        "calibration": {"excellent", "fair", "poor", "not computed"},
    }
    if set(margins) != {"rt", "choice"}:
        raise ValueError("Both separate RT and choice assessments are required.")
    for assessment, sections in (
        (checks, ("convergence", "loo", "psense")),
        (margins["rt"], ("calibration",)),
        (margins["choice"], ("calibration",)),
    ):
        if not isinstance(assessment, dict):
            raise TypeError("Malformed shared assessment.")
        summaries = assessment.get("summary", {})
        if not isinstance(summaries, dict):
            raise TypeError("Malformed shared assessment summaries.")
        for section in sections:
            item = assessment.get(section, {})
            summary = summaries.get(section)
            if (
                not isinstance(item, dict)
                or item.get("rating") not in allowed[section]
                or not isinstance(summary, str)
                or not summary.strip()
            ):
                raise ValueError(f"Missing or malformed shared {section} assessment.")


def _validate_domain_checks(domain_checks: pd.DataFrame) -> None:
    required = {
        "stage",
        "choice",
        "quantity",
        "observed",
        "predicted_mean",
        "lower_94",
        "upper_94",
    }
    if domain_checks.empty or not required.issubset(domain_checks.columns):
        raise ValueError("Missing choice/RT domain evidence.")
    if set(domain_checks.stage) != {"prior_predictive", "posterior_predictive"}:
        raise ValueError("Both prior and posterior choice/RT evidence are required.")
    for _, group in domain_checks.groupby("stage"):
        if set(group.choice) != {-1, 1}:
            raise ValueError("Choice/RT evidence must retain both response boundaries.")
    if not np.isfinite(
        domain_checks[["observed", "predicted_mean", "lower_94", "upper_94"]].to_numpy()
    ).all():
        raise ValueError("Non-finite choice/RT summaries require investigation.")


def _assess_saved_margin(diagnostic_path: Path, cal_path: Path, name: str) -> dict:
    """Assess one scalar margin with the real shared CLI and joint diagnostics."""
    check_path = cal_path.with_name("check_report.json")
    check_path.unlink(missing_ok=True)
    calibration = json.loads(cal_path.read_text())
    assessment = calibration.get("assessment", {})
    deviation = assessment.get("mean_coverage_deviation")
    if (
        calibration.get("variable") != name
        or calibration.get("pit_method") != "ppc_pit"
        or type(assessment.get("well_calibrated")) is not bool
        or not isinstance(deviation, (int, float))
        or not np.isfinite(deviation)
        or not isinstance(assessment.get("calibration_diagnosis"), str)
    ):
        raise ValueError(f"Missing or malformed marginal PPC-PIT evidence for {name}.")
    return _run_helper(
        "check_diagnostics.py",
        [
            "--diagnostics",
            str(diagnostic_path),
            "--calibration",
            str(cal_path),
            "--output",
            str(check_path),
        ],
        check_path,
    )


def _save_report(report: str, output_dir: Path) -> None:
    """Publish a complete report only after every linked figure exists."""
    destination = output_dir / "report.md"
    destination.unlink(missing_ok=True)
    for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report):
        path = output_dir / link
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing report figure: {link}")
    temporary = output_dir / "report.md.tmp"
    temporary.write_text(report)
    temporary.replace(destination)


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
        "The prior predictive distribution shows the data the model would generate before seeing any observations, using only the priors.",
        "The prior predictive samples come from the specified priors before posterior fitting. Here the upper bound for t is set from the observed minimum RT, so this check uses a data-informed prior and is not independent of the observations.",
    )
    template = template.replace(
        "The forest plot shows posterior medians (points) and credible intervals (lines) for the parameters of interest. Wide intervals indicate parameters the data are only weakly informative for; narrow intervals concentrated away from zero indicate strong evidence in a direction.",
        "The forest plot shows posterior medians (points) and 50% and 94% HDIs (lines). Interval width describes posterior uncertainty on each parameter's own scale; compare priors and posteriors before attributing precision to the data. Interpret drift v relative to 0 and normalized starting point z relative to 0.5. For a and t, excluding zero reflects their positive support and is not itself evidence of a directional effect.",
    )
    template = template.replace(
        "The PIT-ECDF plot tests whether the model's predictive distribution is calibrated — that is, whether stated credible levels match empirical coverage. The empirical CDF of probability integral transform values should fall within the simultaneous confidence bands. Lines outside the bands above the diagonal indicate under-confident predictions (intervals wider than they should be); lines below indicate over-confident predictions (intervals too narrow).",
        "The PIT plot shows the empirical CDF of predictive PIT values minus the uniform CDF (ΔECDF). The dashed horizontal zero line is the uniform reference. Departures can reflect location bias or dispersion errors; their shape matters. The p-value annotation reports a test against that reference at the displayed significance level. These are fitted-data marginal PPC-PIT checks, not held-out validation.",
    )
    template = template.replace(
        "The coverage plot tests the same idea in coverage units: it asks whether nominal central credible intervals (50%, 80%, 95%) actually contain the stated fraction of the observed data. A well-calibrated model lies on the diagonal.",
        "The coverage plot applies the same ΔECDF display to coverage-transformed PIT values. Its x-axis gives nominal central predictive coverage in percent; its y-axis gives the ECDF difference, with agreement represented by the horizontal zero line. Use its p-value annotation alongside the PIT check; passing both checks does not establish joint or held-out calibration.",
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


def hssm_next_steps(checks: dict, margins: dict) -> list[str]:
    """Translate shared ratings into actions for the supported flat DDM."""
    _validate_assessments(checks, margins)
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
    if checks["psense"]["rating"] == "not computed":
        next_steps.append(
            "Prior sensitivity is unavailable; inspect its evidence and verify complete effective DDM prior densities before drawing a robustness conclusion."
        )
    for name, assessment in margins.items():
        if assessment["calibration"]["rating"] in {"fair", "poor"}:
            next_steps.append(
                f"Investigate the {name} marginal PPC-PIT discrepancy alongside choice proportions and conditional RT tails. Revisit RT units, t support, priors and the DDM task assumptions before considering a separately validated model extension."
            )
        if assessment["calibration"]["rating"] == "not computed":
            next_steps.append(
                f"The {name} marginal check is unavailable; inspect its scalar artifact and helper output, and leave its calibration claim unassessed."
            )
    next_steps.append(
        "Review choice proportions and conditional RT quantiles together, including absent-choice replicate counts. Interpret parameters only when the available convergence and domain checks support the intended claim."
    )
    return next_steps


def assemble_report(
    model,
    *,
    details: dict,
    domain_checks: pd.DataFrame,
    summary: pd.DataFrame,
    psense: pd.DataFrame,
    diagnostics: dict,
    checks: dict,
    margins: dict,
    versions: dict,
    manifest: dict,
) -> tuple[str, list[str]]:
    """Render existing evidence; never fit, simulate, plot, or assign ratings."""
    _validate_diagnostics(diagnostics)
    _validate_assessments(checks, margins)
    _validate_domain_checks(domain_checks)
    prior_rows = [
        {"Parameter": name, "Resolved prior": str(parameter.prior)}
        for name, parameter in model.params.items()
    ]
    conv = diagnostics["convergence"]
    # Modern shared diagnose output may retain only flags, with no extrema.
    # Display extrema from the already computed unrounded parameter summary;
    # the original shared evidence remains the sole source of statuses/ratings.
    summary_extrema = {
        "rhat": summary["r_hat"].max(skipna=False),
        "ess_bulk": summary["ess_bulk"].min(skipna=False),
        "ess_tail": summary["ess_tail"].min(skipna=False),
    }
    diagnostic_rows = [
        {
            "Diagnostic": label,
            "Value": conv[key][value]
            if conv[key].get(value) is not None
            else summary_extrema.get(key, "Not returned by shared helper"),
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
    next_steps = hssm_next_steps(checks, margins)
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
            + "`. Missing extrema in the shared diagnostic output are taken from summary.csv; pass/flag statuses retain the shared assessment.",
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
            + "\n\n![RT quantiles by choice](quantile_probability.png)"
            + "\n\nIn this quantile-probability plot, the left cluster represents response -1 and the right cluster response +1. Crosses joined by lines show observed quantiles; dots show quantiles from predictive replicates. Colors identify the 0.1, 0.5 and 0.9 RT quantiles, and the horizontal position shows each choice's proportion.",
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
            + (
                " Reason: " + diagnostics["loo"]["error"]
                if not diagnostics["loo"]["computed"]
                else ""
            )
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
    return report, next_steps


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
    (output_dir / "report.md").unlink(missing_ok=True)
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
    _validate_domain_checks(domain_checks)
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
        posterior,
        var_names=["v", "a", "z", "t"],
        ci_prob=0.94,
        ci_kind="hdi",
        round_to="none",
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
    _validate_diagnostics(diagnostics)
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
        margins[name] = _assess_saved_margin(diagnostic_path, cal_path, name)

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
    report, next_steps = assemble_report(
        model,
        details=details,
        domain_checks=domain_checks,
        summary=summary,
        psense=psense,
        diagnostics=diagnostics,
        checks=checks,
        margins=margins,
        versions=versions,
        manifest=manifest,
    )
    _save_report(report, output_dir)
    return {
        "checks": checks,
        "marginal_checks": margins,
        "next_steps": next_steps,
        "summary": summary,
        "domain_checks": domain_checks,
    }
