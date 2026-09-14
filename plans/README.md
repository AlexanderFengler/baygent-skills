# Bambi and HSSM workflow skills

**Status:** Bambi M1 implementation and runtime validation recorded on 2026-09-14;
paired agent/trigger evaluations pending. HSSM M2 remains a plan. See the
[validation record](../evals/bambi-workflow/iteration-1/README.md).

## Branch and upstream assessment

The contribution branch is `AlexanderFengler/baygent-skills:hssm-bambi-dev`.
At review, its published head was `73de41c0c6df7b94747dc29ae34b9b394d8f57b2`,
and upstream `main` was `18ab0de9b63b5bb03caa96146235e8f1534d45d4`.
The branch contained all upstream commits and two additional planning commits.
There was no upstream merge or rebase to perform, and no shipped Bambi or HSSM
skill had superseded these proposals.

Several draft details were already superseded by upstream conventions. The
refresh below replaces those details; original drafts remain in Git history at
`ee52405` and `73de41c`.

| Upstream work already present in the branch | Consequence for the new skills |
|---|---|
| [Portable reporting harness, #13](https://github.com/Learning-Bayesian-Statistics/baygent-skills/pull/13) | Reuse `diagnose_model.py` → `calibration_check.py` → `check_diagnostics.py` and the canonical `<slug>/report.md`; retire the draft's `analysis_notes.md` deliverable and bespoke diagnostic ratings. |
| [PyMC 5/6 compatibility, #14](https://github.com/Learning-Bayesian-Statistics/baygent-skills/pull/14) | Shared helpers handle both InferenceData and DataTree. Each new package workflow must still demonstrate its own supported stack; an InferenceData-only handoff is insufficient. |
| [Calibration and model-comparison fixes, #15](https://github.com/Learning-Bayesian-Statistics/baygent-skills/pull/15) | Retire `plot_ppc_pit(..., loo_pit=True)`, use current shared calibration guidance, and do not add a fixed ELPD-difference cutoff. Quote Markdown author values in YAML. |
| [Amortized workflow v2, #11](https://github.com/Learning-Bayesian-Statistics/baygent-skills/pull/11) | BayesFlow training remains owned by `amortized-workflow`; fitting SSM likelihoods with HSSM is a distinct task. |

The proposals retain a useful role: Bambi-specific regression construction and
interpretation, and HSSM-specific choice/RT modeling. Their value is the package
workflow and its verified boundaries, rather than another general Bayesian
workflow or reporting implementation.

## Architecture and next milestone

Use ordinary sibling skill folders, like the three shipped skills. Keep a
concise `SKILL.md`, optional detailed `references/`, and executable `scripts/`
only when a demonstrated repeated task needs one. Evaluation lives in root
`evals/`; developer environments and plans are not installed skill resources.
No new plugin, registry, shared runtime, or packaging framework is needed.

Both skills depend **directly** on `bayesian-workflow`. Bambi owns its formulas,
families, priors, fit/predict APIs, and response-scale interpretation. HSSM owns
its parameter priors, links, likelihoods, response coding, and predictive APIs.
A precise formula-syntax reference may be reused; Bambi auto-priors do not
transfer to HSSM.

1. **M1: an installable, evaluated Bambi workflow.** Demonstrate Gaussian
   regression and a hierarchical Bernoulli survey analysis on Bambi 0.21.0,
   from model specification through saved predictions, shared diagnostics,
   and a canonical report. Include real execution smoke checks and paired
   agent evaluations. See [architecture](bambi-workflow-skill.md) and
   [implementation/acceptance plan](bambi-workflow-skill-iter2.md).
2. **M2: an HSSM analytical DDM workflow.** Start with a flat, clean two-choice
   dataset and HSSM-specific prior/posterior predictive checks. Establish how
   RT/choice outputs connect to shared diagnostics before broadening to
   hierarchical models. See [HSSM plan](hssm-workflow-skill.md).

The installed Bambi folder, two marimo examples and 13 integration checks now
exist. Both full examples executed against Bambi 0.21.0/PyMC 6.3.2 and produced
canonical reports. M1's remaining acceptance gate is the independent paired
behavior and trigger evaluation; no benchmark improvement is claimed yet.

Next, finish that gate and compare a justified alternative region-scale prior
in the survey example, whose sensitivity assessment flags that parameter.
Broader Bambi families and HSSM M2 require their own verified increments.
