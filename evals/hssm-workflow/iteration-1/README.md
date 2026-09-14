# HSSM runtime validation and pending behavior evaluations

**Status (2026-09-14): HSSM runtime validated; skill behavior pending.**
The flat analytical-DDM workflow passed deterministic contracts, independent
numerical density checks and a full synthetic notebook/report run. See the
[runtime evidence](runtime/README.md) for exact commands, source hashes,
resolved requirements, test results, figures and raw assessments. Paired agent
and trigger evaluations have not executed; their benchmark fields remain null.
The final combined suite recorded **139 passed, one intentionally skipped and
four strict expected failures**. The corrected canonical HTML export succeeded
with the original model, seed and sampling budget.

The [skill](../../../hssm-workflow/SKILL.md), installed RT/choice adapter and
[marimo notebook](../../../examples/hssm-workflow/analytical_ddm.py) target HSSM
0.5.0, using release source `fefed57d2142637af503b0e92cefe799715c0f46`.
The tested environment used Python 3.12.13/macOS arm64 with HSSM 0.5.0,
Bambi 0.19.0, PyMC 6.1.0, ArviZ 1.2.0 and NumPy 2.4.6. Its pip requirements
were installed with uv and system Graphviz. The alternative
[conda recipe](../../../environment-hssm.yml) itself was not tested.

## Completed runtime gates

- ~~Resolve a clean environment and verify imports, dependency consistency,
  labeled DataTree serialization and graph rendering.~~
- ~~Test the paired-response adapter and actual installed CLI with literal data,
  including exact values/labels, malformed input and unchanged source artifacts.~~
- ~~Test hand-calculated choice proportions, replicate conditional RT quantiles,
  absent-choice counts, execution buttons and failed/stale report handling.~~
- ~~Verify effective prior support/densities and native joint DDM likelihood
  against the separately compiled `hddm_wfpt` reference, including response
  conventions, reflection symmetry, normalization and physical support.~~
- ~~Execute the documented full notebook budget and inspect the saved joint and
  scalar artifacts, domain summaries, shared assessments and rendered figures.~~

The ordinary adapter suite's separate opt-in full-fit integration was skipped
intentionally: the full HTML export and saved-artifact checks provide the real
execution evidence without another duplicate fit. Strict expected failures
retain two native sample-coordinate bugs across two `t`-support cases. HSSM's
likelihood method uses chain/draw labels as positional indices, and PyMC's prior
method can reset those labels. The notebook rejects nondefault sample labels
before density recomputation and compares output coordinates afterward. The
scalar adapter independently preserves arbitrary unique sample/trial labels.

Native QP plotting needed two scoped call corrections: HSSM already separates
response quantiles, so `quantile_by="response"` unnecessarily duplicates a
column; explicit `ax` is also forwarded twice in that release. A native
literal-array regression verifies the plotted points. Failed export attempts
and their checkpoints are retained with the execution record.

## What the run establishes

The example generated 300 teaching trials and used 200 prior draws, two chains,
1,000 tuning and 1,000 retained draws per chain. It saved the fit before
post-processing and completed native predictions, independent density guards,
shared convergence/joint-trial LOO/sensitivity checks, and a canonical report.
The shared checks rated convergence and LOO **excellent**, prior sensitivity
**low**, and each fitted-data RT/choice marginal PPC-PIT assessment **excellent**.
However, review found that the final RT coverage plot reports a discrepancy
while the shared JSON marks its coverage as inside the bands. That disagreement
is under investigation: the JSON's RT rating cannot currently support a claim
of adequate RT marginal calibration. The linked report retains the actual
parameter and domain summaries, and the runtime record preserves both outputs.

These ratings apply to this dataset and specification. They do not establish
multi-dataset parameter recovery, joint/conditional/held-out calibration,
scientific validity for another task, or broad HSSM/platform compatibility.
HSSM 0.5's posterior-predictive API has no public seed argument; the saved draws
are evidence, not a claim of deterministic PPC reproduction. Power sensitivity
does not test changes to the data-informed non-decision-time support policy.

## Original implementation baseline

At the original `54bf1f77e801a2dcb1a539f4cbb4a37b8ab0c6c2` plan baseline,
validation consisted of release-source review, syntax/frontmatter/link checks,
Ruff, strict marimo checks and collection of 44 test cases without execution.
That earlier state remains in Git history. The runtime evidence above supersedes
its deferred numerical status; it does not convert authored agent metadata into
executed behavior evidence.

The installed skill remains self-contained with its direct Bayesian dependency.
Its scalar RT and binary `response == +1` views exclude joint log likelihood;
LOO and sensitivity retain the original paired artifact. No thresholds or
statistical ratings are reimplemented in the HSSM adapter.

## Remaining acceptance gates

1. Execute the prepared independent paired with-skill/without-skill scenarios
   and trigger cases after the model-service payloads are authorized. Preserve
   actual outputs, per-assertion grading and timing; authored fixtures have no
   benchmark score. See the [validation plan](../../../plans/validation-round-1.md).
2. Treat a hierarchical condition contrast as a separate model increment with
   HSSM's own safe-prior/link policy. LAN artifact-domain checks, RLSSMs, missing
   RTs, deadlines and lapse regressions need their own scoped verification.

Formal simulation-based calibration, multi-dataset recovery and a multi-platform
support matrix remain separate follow-ups.
