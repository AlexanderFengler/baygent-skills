---
name: hssm-workflow
description: >
  Fit and assess choice/reaction-time models with HSSM. Use when the user
  requests HSSM, a drift-diffusion model, DDM parameter inference, or an
  HSSM choice/RT predictive check. Guides response coding, RT units,
  likelihood selection, parameter support and native HSSM prediction;
  delegates shared diagnostics and reporting to bayesian-workflow.
  The initial implementation covers a flat analytical DDM. Preserve explicit
  requests for ordinary RT regression, raw PyMC custom likelihoods, or
  BayesFlow/LAN training.
license: MIT
metadata:
  author: "[Alexander Fengler](https://www.alexanderfengler.com)"
  version: "0.1"
---

# HSSM Workflow

Use HSSM to connect a choice/RT question to an explicit sequential-sampling
model, predictive evidence in that domain, and a reproducible analysis report.

## Dependencies

Requires **bayesian-workflow** directly for convergence assessments, applicable
model-checking guidance, diagnostic utilities, and the canonical report.
Locate it through the agent's skill catalog; a sibling
`bayesian-workflow/SKILL.md` is another candidate, not a required installation
layout. Read the dependency and resolve its absolute directory before using
its references or scripts. `BAYESIAN_SKILL_DIR` below denotes that verified
location; `HSSM_SKILL_DIR` denotes this installed skill's directory.

If the dependency is absent, explain which checks are unavailable and install
the two folders into the user's selected skill location when authorized. Their
source is [baygent-skills](https://github.com/Learning-Bayesian-Statistics/baygent-skills).
The installed workflow does not require this repository's examples or plans.

The **Bambi skill is optional**. A future parameter-regression request can use
its formula-notation reference, while HSSM continues to own parameter priors,
links, sampling and prediction. Do not apply Bambi's `fit`, `interpret`, or
prior-scaling instructions to an HSSM object.

## Initial scope and validation status

This implementation targets released **HSSM 0.5.0** and a **flat analytical
DDM**, with complete positive RTs in seconds, responses `-1/+1`, and an explicit
no-lapse assumption. Deterministic contracts, independent numerical density
checks and one full synthetic notebook/report run passed on HSSM 0.5.0,
Bambi 0.19.0, PyMC 6.1.0, ArviZ 1.2.0 and NumPy 2.4.6, installed with uv on
Python 3.12.13/macOS arm64. This validates that scoped runtime workflow;
agent behavior evaluations, multi-dataset recovery and broader compatibility
remain untested. Marginal fitted-data predictive checks do not establish joint
or held-out calibration. Record installed versions for each analysis and do
not substitute the separate Bambi skill's version set.

Hierarchical effects, LANs, RLSSMs, alternate SSMs, missing/deadline trials,
lapse mixtures and regressions are later extensions. HSSM itself supports more
than this initial scope. For those requests, identify the extension and verify
its current package contract instead of forcing it through these examples.

## Workflow overview

For focused questions or an existing fit, start at the applicable step. For
a full analysis, complete this sequence within the user's execution scope.

1. **Define the question.** Establish the task's choices, RT units and what
   DDM parameters would answer the question. Preserve supplied decisions;
   resolve only missing assumptions that affect the model.
2. **Validate the data.** Check finite positive RTs, exact response labels,
   retained rows and whether the data contain timeouts, exclusions or lapses.
   Read [references/data-and-ddm.md](references/data-and-ddm.md).
3. **Choose the likelihood.** Set `model="ddm", loglik_kind="analytical"`
   explicitly and inspect `hssm.show_defaults("ddm", "analytical")`.
   Set `p_outlier=None` only for the intended clean no-lapse analysis.
4. **Specify priors and support.** Use `include` with `hssm.Param` or dictionary
   specifications. Keep priors on the likelihood-parameter scale for this flat
   model; ensure the configured upper support of `t` is below the smallest
   retained RT. Read [references/priors-and-links.md](references/priors-and-links.md).
5. **Check prior predictions.** When execution is in scope, call
   `model.sample_prior_predictive(...)` and inspect RT ranges, tails and choice
   proportions using HSSM's predictive surfaces before fitting.
6. **Fit and save.** Use `model.sample(sampler="pymc", ...)` for the explicit
   analytical path shown here; record sampler settings and a descriptive seed.
   Save its returned DataTree before subsequent processing.
7. **Check posterior predictions.** Use
   `model.sample_posterior_predictive(dt=..., inplace=False)` on the saved-fit
   data. Compare choice proportions and conditional RT distributions. Preserve
   the original posterior artifact and report any unsupported predictions.
8. **Assess and report.** Use the installed adapter and shared Bayesian helpers
   only for quantities whose input assumptions hold. Fill `<slug>/report.md`
   with convergence ratings, RT/choice evidence, the model/likelihood and
   coding, and unassessed checks with their reasons. Read
   [references/prediction-and-reporting.md](references/prediction-and-reporting.md).

## Critical rules

- **Preserve the meaning of the data.** Positive RTs and valid response labels
  do not establish correct units, accuracy coding, independence, or absence of
  timeouts. Record conversions and exclusions; do not silently take absolute
  RTs or recode choices based on their ordering.
- **Constrain non-decision time for this no-lapse model.** Choose and justify
  `t`'s prior support, verify `t_upper < min(rt)`, and inspect the effective
  parameter. The analytical implementation can return a finite numerical
  floor for invalid `rt - t`; finite likelihood values alone do not validate
  support. Do not drop fast observations merely to satisfy a chosen prior.
- **Separate bounds, priors and links.** A flat parameter has no regression
  link to configure. HSSM's `link_settings` and safe regression-prior presets
  do not change that fact. Follow HSSM's parameter policy when extending to
  hierarchies; there is no universal non-centering rule here.
- **Keep native HSSM method names and return semantics.** `sample` returns the
  fitted DataTree; `sample_posterior_predictive` and `log_likelihood` mutate by
  default. Bambi examples are not alternate methods on the HSSM object.
- **Treat RT and choice as one trial's outcome.** Do not flatten their component
  axis into extra observations or apply scalar PIT to the combined variable.
  [scripts/prepare_rt_choice.py](scripts/prepare_rt_choice.py) produces separate
  fitted-data marginal RT and choice PPC views without log likelihood. Their
  assessments cannot establish joint, conditional or held-out calibration.
- **Keep execution evidence explicit.** Plotting can trigger predictive
  generation if the required group is absent. Verify the group first, and do
  not use plots as a back door to deferred simulation. A written script or
  successfully saved JSON is not an executed, healthy analysis.

## When things go wrong

| Symptom | Check and next action |
|---|---|
| Response validation error | Check the declared `-1/+1` mapping and its scientific meaning. |
| Implausibly fast or slow predictions | Recheck seconds versus milliseconds, `t` support, and priors on boundary/drift. |
| Finite likelihood despite `t >= rt` | The analytical floor is not valid support; revise the configured prior or model assumptions. |
| Divergences or weak identification | Inspect joint parameter uncertainty, prior predictions and Bayesian diagnostics before interpretation. |
| Predictive call returns `None` | It used its default in-place mutation; request `inplace=False` for a separate result. |
| Calibration helper sees a length-two outcome | Use the documented RT/choice adaptation and preserve its scope limits. |
| A plot starts generating samples | Supply an existing predictive group or leave the plot pending when simulation is deferred. |
