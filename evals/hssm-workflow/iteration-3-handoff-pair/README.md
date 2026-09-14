# Budgeted HSSM artifact-handoff pair

The user authorized this reduced pilot after requesting economical runs: two
sessions, gpt-6-astra with medium reasoning, a target of25,000 input/output tokens
per session and a10-minute timeout each. No retries, fitting, simulation,
predictive generation, trigger trials or automatic expansion are authorized.
The original12-session pilot remains unexecuted except for this selected pair.

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

Status: prepared; execution results will be recorded from actual sessions.
