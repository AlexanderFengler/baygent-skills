# Bambi and HSSM workflow skills

**Status (2026-09-14):** Bambi M1 implementation and runtime validation recorded;
paired agent/trigger evaluations pending. HSSM M2 skill and analytical-DDM
notebook are implemented and statically checked. HSSM numerical runs and all
new agent evaluations are deferred by request. See the
[Bambi validation record](../evals/bambi-workflow/iteration-1/README.md) and
[HSSM validation record](../evals/hssm-workflow/iteration-1/README.md).

The [next validation round](validation-round-1.md) specifies the environment,
deterministic contracts, independent density checks, full notebook run and
separate agent evaluations, with explicit evidence and acceptance criteria.

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

Use ordinary sibling skill folders, like the existing skills. Keep a
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

The HSSM implementation now includes a standalone skill, three focused
references, a strict RT/choice adapter, one gated marimo notebook and authored
tests/evaluation scenarios. It uses the same direct Bayesian dependency and
canonical report. The candidate environment and statistical workflow have not
been executed; source inspection and static checks do not complete M2 acceptance.

Next, resolve the HSSM candidate environment and perform the deferred adapter,
full notebook and paired agent checks described in its validation record. For
Bambi, complete the agent evaluation gate and compare a justified alternative
region-scale prior in the survey example, whose sensitivity assessment flags
that parameter. Broader models require their own verified increments.
