"""Plotting and report assembly for the two repository examples.

Model construction, sampling and Bambi log-density/prediction calls stay in the
notebooks. Diagnostic calculations and assessments stay in bayesian-workflow.
This example-only module is not a dependency of the installed Bambi skill.
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


def result_directory(slug: str) -> Path:
    """Create the notebook's result folder; tests may select another root."""
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


def _convergence_rows(convergence: dict, summary: pd.DataFrame) -> list[dict]:
    """Display available extrema without changing the shared diagnostic verdicts."""
    extrema = {
        "rhat": summary["r_hat"].max(skipna=False),
        "ess_bulk": summary["ess_bulk"].min(skipna=False),
        "ess_tail": summary["ess_tail"].min(skipna=False),
    }
    return [
        {
            "Diagnostic": name,
            "Value": convergence[key][value]
            if convergence[key].get(value) is not None
            else extrema.get(key, "Not returned by shared helper"),
            "Status": "pass" if convergence[key]["ok"] else "flag",
        }
        for name, key, value in [
            ("Max R-hat", "rhat", "max"),
            ("Min ESS (bulk)", "ess_bulk", "min"),
            ("Min ESS (tail)", "ess_tail", "min"),
            ("Divergences", "divergences", "count"),
        ]
    ]


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
    # ArviZ 1.3 plot_trace shows draw order, not the template's legacy rank view.
    template = template.replace(
        "Rank-vline trace plots check chain mixing. Well-mixed chains show overlapping rank distributions across chains — the vertical lines (one per chain) sit close to the uniform expectation.",
        "Trace plots show parameter draws in sampling order for each chain. Well-mixed chains explore similar ranges without persistent drift or sticking.",
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
    model, prior, idata, result, output_dir, *, outcome, var_names, details
):
    """Save figures, run shared assessments, and fill the canonical report."""
    output_dir = Path(output_dir)
    # The notebooks save immediately after sampling. Preserve their completed
    # diagnostic artifact rather than using Result.draws (a prediction grid).
    idata.to_netcdf(output_dir / "inference_data.nc")
    prior.to_netcdf(output_dir / "prior_data.nc")
    model.data.to_csv(output_dir / "data.csv", index=False)
    prediction_summary = result.summary.copy()
    prediction_summary.to_csv(output_dir / "predictions.csv", index=False)
    summary = az.summary(
        idata, var_names=var_names, ci_prob=0.94, ci_kind="hdi", round_to="none"
    )
    summary.to_csv(output_dir / "summary.csv")
    psense = azs.psense_summary(idata, var_names=var_names)
    # Shared checker expects parameter -> {prior, likelihood}, not column -> rows.
    _dump(output_dir / "psense.json", psense.to_dict(orient="index"))

    model.graph().render(str(output_dir / "model_graph"), format="png", cleanup=True)
    if model.family.name == "bernoulli":
        # A density over pooled binary values is nearly flat and hides group
        # discrepancies. Plot replicated proportions and regional uncertainty.
        prior_rates = (
            prior["prior_predictive"][outcome].mean(dim="__obs__").values.ravel()
        )
        fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
        ax.hist(prior_rates, bins=25, density=True, alpha=0.6)
        ax.axvline(
            model.data[outcome].mean(), color="black", label="Observed proportion"
        )
        ax.set(
            xlabel="Replicated overall support proportion",
            ylabel="Density",
            xlim=(0, 1),
        )
        ax.legend()
        fig.savefig(output_dir / "prior_predictive.png")
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        labels = list(model.data.region.cat.categories)
        replicated = idata["posterior_predictive"][outcome].values
        for index, region in enumerate(labels):
            mask = model.data.region == region
            rates = replicated[..., mask].mean(axis=-1)
            low, high = np.quantile(rates, [0.03, 0.97])
            ax.plot(
                [index, index],
                [low, high],
                color="C0",
                lw=3,
                label="94% replicated-proportion interval" if index == 0 else None,
            )
            ax.plot(index, rates.mean(), "o", color="C0")
            ax.plot(
                index,
                model.data.loc[mask, outcome].mean(),
                "x",
                color="black",
                label="Observed proportion" if index == 0 else None,
            )
        ax.set(
            xticks=range(len(labels)),
            xticklabels=labels,
            ylim=(0, 1),
            ylabel="Support proportion",
            title="Known-region posterior predictive check",
        )
        ax.legend()
        fig.savefig(output_dir / "posterior_predictive.png")
    else:
        azp.plot_ppc_dist(prior, group="prior_predictive", var_names=[outcome]).savefig(
            output_dir / "prior_predictive.png"
        )
        azp.plot_ppc_dist(idata, var_names=[outcome]).savefig(
            output_dir / "posterior_predictive.png"
        )
    figure_kwargs = {
        "figsize": (10, max(4, 2.5 * len(summary))),
        "layout": "constrained",
    }
    az.plot_trace(idata, var_names=var_names, figure_kwargs=figure_kwargs).savefig(
        output_dir / "trace.png"
    )
    az.plot_forest(
        idata,
        var_names=var_names,
        combined=True,
        ci_probs=[0.5, 0.94],
        ci_kind="hdi",
        point_estimate="median",
    ).savefig(output_dir / "forest.png")
    azp.plot_psense_dist(
        idata, var_names=var_names, figure_kwargs=figure_kwargs
    ).savefig(output_dir / "psense.png")
    plt.close("all")

    artifact = str(output_dir / "inference_data.nc")
    diag_path = output_dir / "diagnostics.json"
    cal_path = output_dir / "calibration.json"
    check_path = output_dir / "check_report.json"
    diagnostics = _run_helper(
        "diagnose_model.py",
        ["--idata", artifact, "--output", str(diag_path)],
        diag_path,
    )
    calibration = _run_helper(
        "calibration_check.py",
        [
            "--idata",
            artifact,
            "--var-name",
            outcome,
            "--output",
            str(cal_path),
            "--save-plots",
            "--plot-dir",
            str(output_dir),
        ],
        cal_path,
    )
    # Portable links in the stored JSON; no change to computed calibration values.
    calibration["plots"] = {
        key: Path(value).name for key, value in calibration["plots"].items()
    }
    _dump(cal_path, calibration)
    checks = _run_helper(
        "check_diagnostics.py",
        [
            "--diagnostics",
            str(diag_path),
            "--calibration",
            str(cal_path),
            "--psense",
            str(output_dir / "psense.json"),
            "--output",
            str(check_path),
        ],
        check_path,
    )
    versions = {
        name: version(name)
        for name in (
            "bambi",
            "pymc",
            "pytensor",
            "formulae",
            "arviz",
            "arviz-stats",
            "arviz-plots",
            "nutpie",
            "marimo",
            "numpy",
            "pandas",
        )
    }
    versions["python"] = sys.version.split()[0]
    _dump(output_dir / "versions.json", versions)
    details["sampling"] = {
        **details["sampling"],
        "chains": idata["posterior"].sizes["chain"],
        "retained_draws_per_chain": idata["posterior"].sizes["draw"],
        "backend": "nutpie",
        "omit_offsets": False,
        "center_predictors": model.center_predictors,
    }
    _dump(output_dir / "analysis.json", details)

    prior_rows = [
        {
            "Parameter": f"{name}:{term_name}",
            "Prior": str(term.prior),
            "Justification": details["prior_rationale"],
        }
        for name, component in model.conditional_parameters.items()
        for term_name, term in component.terms.items()
    ]
    prior_rows.extend(
        {
            "Parameter": name,
            "Prior": str(parameter.prior),
            "Justification": details["prior_rationale"],
        }
        for name, parameter in model.marginal_parameters.items()
    )
    diagnostic_rows = _convergence_rows(diagnostics["convergence"], summary)
    data_table = pd.DataFrame(
        {
            "Field": ["Source", "Sample size", "Key variables", "Question"],
            "Value": [
                "Generated synthetic teaching data",
                len(model.data),
                details["encoding"],
                details["question"],
            ],
        }
    )
    limitations = (
        [details["limitations"]]
        if isinstance(details["limitations"], str)
        else list(details["limitations"])
    )
    if details["smoke"]:
        limitations.insert(
            0,
            "Smoke-sized run: validates the software path, not substantive posterior adequacy.",
        )
    limitations.append(
        "PPC-PIT reuses the fitted observations; it is not evidence of out-of-sample calibration or causal identification."
    )
    # These examples have one explicitly defined response-scale target each.
    target = prediction_summary.iloc[0]
    lower = next(name for name in target.index if name.startswith("lower_"))
    upper = next(name for name in target.index if name.startswith("upper_"))
    answer = (
        f"{details['prediction_target']} Posterior mean: {target['estimate']:.3f}; "
        f"94% HDI [{target[lower]:.3f}, {target[upper]:.3f}]."
    )
    sections = {
        "Executive Summary": (
            answer
            + " "
            + details["interpretation"]
            + " "
            + checks["summary"]["convergence"]
            + " "
            + checks["summary"]["psense"]
            + (" This is an execution smoke run." if details["smoke"] else ""),
            None,
        ),
        "Data and Question": (details["encoding"], _table(data_table)),
        "Model Specification": (
            details["generative_story"]
            + f" Formula: `{details['formula']}`; family/link: {details['family']}/{details['link']}.",
            _table(pd.DataFrame(prior_rows)),
        ),
        "Prior Predictive Check": (details["prior_assessment"], None),
        "Sampling and Convergence": (
            checks["summary"]["convergence"]
            + f" Sampling settings: `{details['sampling']}`."
            + " Missing extrema in the shared output are taken from the selected parameter summary.csv; pass/flag statuses retain the shared assessment.",
            _table(pd.DataFrame(diagnostic_rows)),
        ),
        "Posterior": (
            details["interpretation"]
            + "\n\nTarget: "
            + details["prediction_target"]
            + "\n\n"
            + _table(prediction_summary),
            _table(summary.reset_index(names="Parameter")),
        ),
        "Posterior Predictive Check": (details["ppc_assessment"], None),
        "Calibration": (
            checks["summary"]["calibration"]
            + " This is a PPC-PIT assessment on the fitted data.",
            None,
        ),
        "Prior Sensitivity": (
            checks["summary"]["psense"],
            _table(psense.reset_index(names="Parameter")),
        ),
        "Limitations and Threats": (
            "\n".join(f"{i}. {text}" for i, text in enumerate(limitations, 1)),
            None,
        ),
        "Suggested Next Steps": (
            "\n".join(f"{i}. {text}" for i, text in enumerate(checks["next_steps"], 1)),
            None,
        ),
        "Appendix": (
            f"Descriptive seed: `{details['seed']}`. Resolved versions: `versions.json`. Full posterior summary: `summary.csv`. Native contrast/prediction results: `predictions.csv`.\n\n"
            + _table(pd.DataFrame(versions.items(), columns=["Package", "Version"])),
            None,
        ),
    }
    (output_dir / "report.md").write_text(_canonical_report(sections, details["title"]))
    return {
        "diagnostics": diagnostics,
        "calibration": calibration,
        "checks": checks,
        "psense": psense,
        "prediction_summary": prediction_summary,
    }
