# Budgeted HSSM artifact-handoff pair

The user authorized this reduced pilot after requesting economical runs: two
sessions, gpt-6-astra with medium reasoning, a target of 25,000 input/output tokens
per session and a 10-minute timeout each. No retries, fitting, simulation,
predictive generation, trigger trials or automatic expansion are authorized.
The original 12-session pilot remains on hold; only the first baseline session
of this selected pair ran.

The [manifest](manifest.json) freezes the same literal misalignment scenario and
skill-source hashes as the earlier plan. Baseline receives bayesian-workflow;
treatment adds hssm-workflow. Only the resource wording changes. Each workspace
contains its own copied payload, environment versions, prompt and empty results.
The eight assertions remain those in
[the authored scenario](../iteration-2/predictive-misalignment/eval_metadata.json).

Native rollout-budget configuration was accepted locally but is under development;
this is a token target, not a verified hard spending limit. The controller enforces
wall time, refuses an accidental second launch, captures partial evidence and
stops after a failed first session, missing usage or a first-session overshoot.
Input tokens include cached input; cached tokens must not be counted twice.

Both sessions ignore user config, preserve execution-policy rules, disable the
optional tools from the original manifest and enable skip_host_skill_discovery.
These choices do not establish hard host-read isolation or prove the full model
catalog. The recorded tool actions will be inspected for contamination. This
small pilot receives controller review against predeclared assertions, without
an extra model-based grader; that review is not blind to arm identity.

## Result — stopped at the token budget

The baseline session ran for **46.83 seconds** and exited with
`shared rollout token budget exhausted`. The configured target was 25,000 tokens.
The guard stopped further model work, but the failed turn emitted **no token-usage
record**. Actual input/output/cached/reasoning totals are therefore unknown; this
is not evidence of exactly 25,000 tokens consumed or zero consumption.

It completed six local command groups, recognized the literal synthetic fixture
and mismatched observation IDs, reconstructed the paired arrays, and saved an
expected `xr.align(..., join="exact")` failure. It did not finish
`results/review.md` or a final response. The treatment session **did not start**
because the controller stopped after the first failure and missing usage.

This is an incomplete budget-limited attempt, with **no paired comparison or
skill-performance score**. No fitting, simulation, predictive generation,
extra agents, model-service calls from the tested agent, or silent repairs appear
in the recorded actions. Both workspaces' original payload hashes remain intact.
The observed source lookup stayed within the installed HSSM package.

- [Results and provenance](results.json), including the original/published hashes.
- [Raw event stream](run-01/events.jsonl) and [partial validation](run-01/validation.txt).
- [Assertion observations](assertion-observations.json): incomplete evidence,
  not substituted final grades.
- [Frozen prompt](prompt.txt), [literal input](predictive_fixture.json) and
  [environment](environment.json).
- [Original execution record](run-01/original-execution.json) and
  [original controller summary](original-controller-results.json).
- [Controller regression evidence](controller-tests.xml): five local tests,
  using dummy processes only.

The original controller's zero usage subtotal meant no complete usage events
were available; it must not be read as zero consumption. That reporting defect
was corrected afterward in commit 0c0d4a9, with regressions for native budget
errors and incomplete totals. The executed runner's original hash and commit
remain frozen; no evaluation was rerun.

The next economical design to consider is a separately scoped single-response
review with a short, fixed context. The six broad reading/tool rounds in this
attempt make repeated context a plausible cost driver, but no per-request usage
was returned to quantify it. Changing that design requires a new manifest;
no automatic follow-up is scheduled or authorized by this record.
