# HSSM evaluation inputs — iteration 2

**Status: prepared, not executed.** These three scenarios define a
runtime task and two bounded artifact-review tasks. No agent run, trigger trial,
model-service request, grading result, or benchmark score has been produced by
preparing them. The existing
[deferred-execution scenario](../iteration-1/ddm-flat/eval_metadata.json) remains
unchanged and serves a different execution-permission test.

## Scenarios and input allowlist

For each arm, send the `prompt` string from its metadata and **only** the files
listed below, placed at the workspace root. `eval_metadata.json`, its assertions,
this README, repository notebooks, previous reports and reference outputs belong
to the evaluator and must not enter either agent workspace.

| Scenario | Inputs supplied to each arm | Execution scope |
|---|---|---|
| [ddm-runtime](ddm-runtime/eval_metadata.json) | [data.csv](ddm-runtime/data.csv), [dataset_metadata.json](ddm-runtime/dataset_metadata.json) | Analyze the frozen observations; 200 prior draws, two chains of 1,000 tuning plus 1,000 retained draws, one process. No replacement dataset or automatic budget/seed changes. |
| [predictive-misalignment](predictive-misalignment/eval_metadata.json) | [predictive_fixture.json](predictive-misalignment/predictive_fixture.json) | Inspect or reconstruct the supplied literal arrays; no fitting, simulation or predictive generation. |
| [diagnostic-review](diagnostic-review/eval_metadata.json) | [fixture_metadata.json](diagnostic-review/fixture_metadata.json), [diagnostics.json](diagnostic-review/diagnostics.json), [psense.json](diagnostic-review/psense.json), [rt_calibration.json](diagnostic-review/rt_calibration.json), [choice_calibration.json](diagnostic-review/choice_calibration.json), [domain_checks.csv](diagnostic-review/domain_checks.csv) | Interpret the supplied literal summaries; deterministic shared assessment is allowed, new inference and invented evidence are not. |

The runtime CSV is copied byte-for-byte from the actual notebook's generated
input at the [archived notebook input](https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/hssm-workflow/iteration-1/runtime/analytical-ddm/data.csv).
Its metadata records the hash, generation settings and source baseline. It
contains no fitted draws, estimates, diagnostics, report text or grading output.
The two review scenarios are explicitly hand-authored fixtures; their arrays
and numbers are not outputs from a sampled model.

## Paired-run setup

Use fresh sessions and directories with the same frozen inputs, package stack,
model settings, tools and resource limits for each pair. The baseline receives
`bayesian-workflow`; the treatment adds `hssm-workflow`, including its installed
adapter. Neither requires the Bambi skill. Exclude private Spine context,
unrelated repositories and the other arm's output.

For each evaluation, record model/version, tool access, resource limits and
resolved environment before running. Preserve timeouts, tool failures and
statistical findings separately; grade per-assertion evidence with the skill
condition hidden from the evaluator. Do not create placeholder result files
or interpret prepared metadata as an executed benchmark.

The runtime task must execute its authorized analysis rather than return only a
plan. Review tasks must retain their narrower execution boundary. Poor numerical
health alone is not a software or agent failure: hiding it, changing the target,
corrupting coordinates or fabricating evidence is.

## Contracts behind the assertions

- [HSSM scope and native workflow](../../../hssm-workflow/SKILL.md),
  [parameter support](../../../hssm-workflow/references/priors-and-links.md),
  and [reporting handoff](../../../hssm-workflow/references/prediction-and-reporting.md).
- [Installed paired-response adapter](../../../hssm-workflow/scripts/prepare_rt_choice.py):
  component order, exact trial-label matching, separate scalar marginals and no
  marginal copy of the joint likelihood.
- [Shared assessment implementation](../../../bayesian-workflow/scripts/check_diagnostics.py)
  and [canonical reporting guidance](../../../bayesian-workflow/references/reporting.md):
  ratings remain shared, with HSSM-specific interpretation and scope limits.
- Released [HSSM 0.5.0 native APIs](https://github.com/lnccbrown/HSSM/blob/fefed57d2142637af503b0e92cefe799715c0f46/src/hssm/base.py)
  and [analytical DDM conventions](https://github.com/lnccbrown/HSSM/blob/fefed57d2142637af503b0e92cefe799715c0f46/src/hssm/likelihoods/analytical.py).

Preserve these input files and their metadata when evaluating revisions. A
changed prompt, fixture or skill needs a new source/run identifier; do not
overwrite failed evidence or interpret a small pilot as a population success
rate or a formal recovery/calibration experiment.
