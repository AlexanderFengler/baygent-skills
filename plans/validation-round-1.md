# Validation round 1: HSSM runtime and skill behavior

**Status (2026-09-14): HSSM runtime validated; skill behavior pending.**
Environment, deterministic-contract, independent-density and full-export gates
have passed. See the [runtime evidence](../evals/hssm-workflow/iteration-1/runtime/README.md)
for exact commands, resolved dependencies, reports and limitations. Prepared
model-service scenarios have not executed; no agent or trigger grades exist.

Original implementation baseline: `54bf1f77e801a2dcb1a539f4cbb4a37b8ab0c6c2`
on `hssm-bambi-dev`. HSSM source baseline: released 0.5.0 (`fefed57d`).
The procedures below retain the planned acceptance criteria; sections 1–4 now
have executed evidence, while section 5 remains open.

- ~~Establish and record one clean HSSM environment.~~
- ~~Exercise deterministic adapter/CLI, domain, execution-gate and report contracts.~~
- ~~Verify model-level priors and analytical likelihood against independent references.~~
- ~~Export the full notebook, fix observed defects, and inspect saved evidence.~~
- Execute paired model-service and trigger evaluations with authorized payloads.

The runtime milestone is one execution-validated flat analytical DDM example.
The next milestone is evidence that the Bambi and HSSM skills guide agents correctly.
Software correctness, statistical adequacy, and agent behavior have separate
acceptance criteria. A successful fit alone does not establish any of them.

## Starting evidence at the original plan baseline

- HSSM: static checks passed; 43 literal-array adapter cases and one opt-in full
  notebook integration have been collected but never executed.
- The integration checks joint likelihood shape and finiteness, but currently
  lacks an independent numerical check of its values. This is the highest-risk gap.
- The notebook and integration already contain independent flat-prior formulas;
  neither has executed. Domain summaries, preview gates, report failure handling,
  and the installed adapter CLI need focused coverage beyond the current tests.
- Bambi: 13 integration checks and two full notebook exports already passed on
  its recorded stack. Do not repeat these fits unless this round changes their
  code/dependencies or exposes a shared compatibility issue.
- All paired agent and trigger evaluations remain pending. The existing HSSM
  `ddm-flat` prompt explicitly forbids execution: retain it as a scope-respect
  scenario and add a distinct authorized-runtime scenario.

## 1. Establish one reproducible HSSM environment

Create a fresh Python 3.12 environment under Spine's
`_local/venvs/baygent-hssm-m2`; do not reuse or modify the Bambi environment.
Resolve the pins in [environment-hssm.yml](../environment-hssm.yml), including
Graphviz, plotting and netCDF support. Use released packages, checking import
paths for accidental editable sibling packages or `PYTHONPATH` overrides.

Record the Python/platform, resolved package versions, package origins, `dot`
version and dependency consistency check. Exercise imports, a tiny labeled
DataTree file roundtrip and graph rendering before any sampling. If the
candidate pins cannot resolve, document and review the smallest justified pin
change; retain the failed resolution log. Do not silently upgrade the stack.

**Gate:** a reproducible environment on the current host, with required native
tools and serialization working. This establishes one platform, not a support
matrix. Preserve exact resolved requirements with the later runtime evidence.

## 2. Run deterministic contracts and close focused gaps

Begin with the existing suite in that environment, with execution flags unset:

```bash
env -u BAYGENT_TEST_HSSM -u BAYGENT_RUN_HSSM \
  python -m pytest -q evals/smoke/test_hssm_workflow.py
```

The original selection was **43 adapter cases and one opt-in full integration**.
The completed round added the focused contracts below; exact results are in
the runtime record. The duplicate full-fit integration remains intentionally
skipped because the successful HTML export and saved-artifact suite exercise it.

| Check | Required evidence |
|---|---|
| Paired response adapter | Exact RT values and binary +1 coding; named sample/trial axes and coordinates preserved; reject malformed labels, invalid values and mismatched rows; source remains unchanged. |
| Installed CLI | Run the actual CLI against a saved literal fixture from a temporary installation containing only HSSM and Bayesian skills. Reload both outputs, compare exact quantities and verify the manifest; malformed input fails visibly. |
| Domain summaries | Hand-computed choice proportions and conditional RT quantiles agree with the reported table. Replicates without a choice contribute to its probability but not its conditional RT quantiles; counts and the all-absent case remain explicit. |
| Execution gates | Default preview never calls simulation or sampling and creates no result directory. With only prior generation enabled, fitting still waits for its own button. Test control flow with fail-on-call sentinels, not an accidental expensive run. |
| Report failure paths | Shared fair/poor/unavailable diagnostics remain visible; HSSM actions match the unchanged ratings. Missing evidence or failed density verification prevents an apparently complete report, including a stale success report from an earlier run in the same directory. Scalar margins never acquire joint log likelihood or a joint-calibration claim. |

Separate pure metric/report verification from fitting only where needed to test
these contracts. Keep the example workflow visible in marimo; do not introduce
a general reporting framework. Use the existing shared checker for diagnostic
fixture ratings, rather than recreating its thresholds in tests.

**Gate:** all deterministic contracts pass with no simulation or MCMC. Fix a
failure and add a focused regression before starting the full notebook. Measure
coverage of the new installed adapter to identify unexercised contracts; do not
expand this into a coverage exercise for HSSM's library code.

## 3. Verify numerical densities before expensive inference

Use a small deterministic parameter/trial grid; this needs no fitted posterior.

1. Verify the effective Normal, LogNormal, Beta and Uniform prior densities in
   the constructed HSSM model, including support and the LogNormal Jacobian.
   Check the complete free-variable set and behavior outside support; sampled
   interior values alone cannot prove that the effective bounds are enforced.
   Cover both branches of `t_upper=min(0.5,0.95*min_rt)` and each distribution's
   actual endpoint convention. Retain the existing draw-wise checks for the
   subsequent real fit.
2. Compare joint per-trial DDM log densities against the separate compiled
   `hddm_wfpt.wfpt.wiener_logp_array` implementation, subject to confirming its
   availability and pinning it in the test environment. Source inspection of
   HSSM's release [blackbox bridge](https://github.com/lnccbrown/HSSM/blob/fefed57d2142637af503b0e92cefe799715c0f46/src/hssm/likelihoods/blackbox.py)
   establishes signed `x=rt*response`, boundary separation `2*a` and zero
   `sv`, `sz`, `st` and `p_outlier` for this flat DDM. Call the reference directly;
   the HSSM wrapper introduces its own numerical floor. This is a different
   execution path from HSSM's analytical PyTensor likelihood, but can share
   mathematical ancestry, so require the additional checks below. Record the
   reference source/version and conventions with the test evidence.
3. Cover both choices, zero/positive/negative drift, asymmetric starting points,
   several boundaries and decision times, and valid times approaching `t`.
   Check reference normalization across both choices, choice-specific masses
   against closed-form hitting probabilities, and reflection symmetry as
   safeguards against an incorrect reference implementation. For comparisons
   to HSSM's floored numerical density, use a finite RT window with an explicit
   independently bounded tail and quadrature allowance; do not integrate its
   positive numerical floor to infinity and interpret that as normalization.
4. Separately test the `rt <= t` numerical-floor behavior and require the teaching
   model's configured `t` support to exclude it for every retained observation.
   A finite floor is not evidence of physically valid data support.

Choose and record tolerances from dtype/reference accuracy before comparing
results; retain maximum discrepancies by case. Investigate failures rather than
loosening tolerances to pass. If a credible independent reference cannot be
established, keep this gate open and avoid a numerical-correctness claim.

**Gate:** native/model-level densities and an independent reference agree on the
declared valid domain; prior support and response conventions are demonstrated.

**Executed finding:** four strict expected failures preserve two release defects
across both `t`-support cases: HSSM likelihood recomputation treats nondefault
sample labels as positional indices, while PyMC prior recomputation resets those
labels. The notebook now refuses nondefault chain/draw labels before either
density call and verifies coordinates afterward; it never silently reindexes.
The scalar adapter retains its separate arbitrary-unique-label contract.

## 4. Execute one full notebook, then inspect its evidence

Use the existing documented budget: **300 synthetic trials, 200 prior draws,
two chains, 1,000 tuning and 1,000 retained draws per chain, one process**.
Preserve the descriptive seed and record that HSSM's public posterior-predictive
API does not provide a seed argument. Do not demand identical PPC files on rerun.

Prepare reusable saved-artifact assertions before execution so the canonical
HTML export can be checked without running a second fit. Reuse the substantive
checks in the existing opt-in integration test; retain that test's explicit
execution gate for later regression runs. Run the export from the repo root:

```bash
mkdir -p examples/hssm-workflow/results/round-1
BAYGENT_RUN_HSSM=1 \
  BAYGENT_OUTPUT_ROOT=examples/hssm-workflow/results/round-1 \
  MPLBACKEND=Agg \
  marimo export html examples/hssm-workflow/analytical_ddm.py \
  -o examples/hssm-workflow/results/round-1/analytical_ddm.html
```

The command intentionally bypasses the interactive buttons. Review the saved
prior assessment as part of validation. Keep all artifacts after a failure;
the immediate posterior checkpoint should survive downstream processing errors.

Validate the saved posterior, complete prior densities, selected joint
likelihood values, original observations, PPC sample/trial alignment and scalar
views. Recompute choice/RT summaries independently from the saved predictions.
Check that every report number and figure link comes from this run. Inspect
the rendered notebook and native predictive, quantile-probability, trace, forest,
sensitivity and calibration figures for interpretable axes and correct claims.

**Software gate:** APIs, numerical assertions, persistence and reporting work
end to end. A poor diagnostic rating must not make a software test fail merely
because it is poor; silently hiding it must fail.

**Scientific review:** report convergence, influential trials, prior sensitivity,
choice proportions and conditional RT tails using the shared assessments. Explain
any unresolved flag before describing the example as suitable for interpretation.
Record estimates relative to generating values descriptively; do not require all
94% intervals to contain truth or call one dataset a recovery experiment.
Do not change seeds until diagnostics look good. Extra draws or model/prior
changes require a documented diagnostic reason and a new evidence record.

Publish compact evidence under `evals/hssm-workflow/iteration-1/runtime/`:
commands/settings, source hashes, resolved requirements, reports, HTML, figures,
synthetic CSVs, numerical summaries and raw diagnostic JSON. Keep large NetCDF
files in ignored results with hashes and regeneration instructions. Update the
validation status only after reviewing the resulting evidence.

**Executed finding:** the first completed fit preserved its checkpoint when
native QP plotting failed. `quantile_by="response"` duplicated a mandatory
column; explicit `ax` was then forwarded twice to seaborn. The notebook omits
both arguments because HSSM already separates response quantiles and can use
the current axes. A native literal-draw regression verifies the actual saved
figure's points. A documented retry with the same model, seed and sampler budget
then exported successfully; failed attempts remain in the evidence record.

**Executed finding:** figure review exposed shared PIT assessment/plot method
disagreement, duplicate coverage transformation and native ECDF coordinate
rescaling. The shared helper now prepares one PIT array for assessment and
plots, uses matching tests, and retains the raw PIT coordinate scale. Independent
literal-array checks, both upstream reporting harnesses, and a strict common-method
PyMC 5/6 comparison passed. The final HSSM export retains **fair RT calibration**
because the coverage-transformed PIT test rejects uniformity. Bambi's saved
full reports were reassessed using the corrected shared pipeline. See the
[calibration evidence](../evals/calibration-method-consistency/README.md).

## 5. Evaluate skill behavior in isolated agent runs

Proceed after local runtime gates pass. This phase makes fresh model-service
requests using skill instructions, synthetic prompts and, for artifact-review
cases, the minimal synthetic artifacts. The earlier automatic approval review
requires separate authorization for that transmission. Planning this phase does
not authorize or initiate it; local tests can proceed independently when requested.

Prepare the exact payload and runner configuration before requesting that final
authorization. Keep private Spine context, unrelated repositories and previous
reference solutions out of the evaluation workspaces.

**Six initial scenarios, one paired run each (12 agent runs):**

| Scenario | Principal check |
|---|---|
| Existing Bambi Gaussian regression | Correct expected-outcome contrast, native prediction and verified report. |
| Existing Bambi hierarchical Bernoulli survey | Event/reference coding, conditional probability target and complete prior densities. |
| Existing HSSM deferred-execution prompt | Useful preparation with no forbidden computation or invented evidence. |
| New HSSM authorized analytical DDM | Actual native workflow, paired/scalar handoff and evidence-based report. |
| New HSSM misaligned predictive artifact | Detect swapped/missing trial labels; never silently align or flatten the pair. |
| New HSSM diagnostically problematic fit | Preserve poor/unavailable shared checks and domain discrepancies without claiming recovery or joint calibration. |

Give each pair the same scenario inputs, package stack, tools, model settings
and resource limits. The baseline receives `bayesian-workflow`; the treatment
adds only the target package skill. Neither sees repository notebooks, previous
outputs, grading rubrics or the other arm's solution. HSSM treatment must work
without installing the Bambi skill. Use fresh sessions/directories, label-blind
grading and per-assertion evidence. Predeclare execution permissions and numerical
budgets per scenario; full runtime cases must not silently become planning tasks.
Freeze the synthetic datasets and generating-process metadata before evaluation
so paired fits use identical observations. Preserve the original metadata and
version any prompt revisions needed to supply those fixed inputs.

If the pilot exposes no critical failure, add two more independent paired runs
for each of the three full modeling scenarios: **24 agent runs total**. This
gives three paired replicates per modeling scenario without spending repeated
fits on every narrow boundary case. Record timeout/tool failures separately from
statistical findings and skill errors; retain unsuccessful runs.

For selection behavior, use the existing **16 Bambi + 8 HSSM trigger queries**,
with three fresh trials per query (**72 selection trials**). Present the normal
available skill catalog and record actual activation, not a prompted self-report
of which skill sounds appropriate. Report false positives, false negatives and
run-to-run consistency per skill; distinguish triggering an HSSM extension from
claiming that the initial flat-DDM workflow already validates it.

**Gate:** no critical treatment failure (wrong quantities/coding, corrupted
densities, ignored execution boundaries or fabricated results). Preserve every
assertion result and paired difference; do not infer broad superiority or a
population success rate from this small evaluation set. If a prompt or skill
changes after a failure, retain the old evidence and evaluate the revision under
a new iteration/source hash rather than overwriting it.

## Completion and deliberate deferrals

The round is complete when the HSSM runtime evidence is reviewed, identified
correctness defects are fixed with regression checks, and the authorized agent
evaluation record is complete. If external evaluations remain deferred, report
the narrower milestone precisely: **HSSM runtime validated; skill behavior pending**.

Formal simulation-based calibration, multi-dataset parameter recovery,
hierarchical HSSM, LAN/RLSSM models, a multi-platform dependency matrix and the
Bambi survey's alternative region-scale prior analysis are separate follow-ups.
Re-run Bambi's existing numerical suite only if affected code or dependencies
change; run the upstream dual-environment harness if shared Bayesian scripts
change. Keep commits focused on contracts, numerical verification, run evidence,
and agent evaluation results; update the fork branch without opening an upstream
PR as part of this test plan.


## Reduced pilot execution — 2026-09-14

After requesting economical evaluation, the user authorized two HSSM artifact-review
sessions with medium reasoning, 25k-token targets and 10-minute timeouts. The
[baseline attempt](../evals/hssm-workflow/iteration-3-handoff-pair/README.md)
stopped after 46.83 seconds when Codex's native rollout-budget guard reported
exhaustion. No usage event or finished review was returned; the second session
was not launched. The original 12-session/ultra proposal stays on hold.
Runtime validation remains complete, while paired behavior validation is incomplete.
No automatic retry, budget increase or replacement pilot is authorized by this result.
