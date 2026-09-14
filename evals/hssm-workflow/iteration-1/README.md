# HSSM implementation and deferred validation

**Status:** implemented for source baseline HSSM 0.5.0 (`fefed57d`); runtime
validation is deferred at the user's request. No simulated data, posterior
samples, rendered notebook, diagnostic scores, recovery results, or agent
benchmark grades were produced in this increment.

The [skill](../../../hssm-workflow/SKILL.md), installed RT/choice adapter, and
[marimo notebook](../../../examples/hssm-workflow/analytical_ddm.py) form the flat
analytical-DDM implementation. The separate [candidate environment](../../../environment-hssm.yml)
uses Bambi 0.19/PyMC 6.1/ArviZ 1.2/NumPy 2.4.6. Pins are source/metadata choices,
not a resolved or tested environment guarantee.

## Completed source and static checks

- HSSM tag `fefed57d2142637af503b0e92cefe799715c0f46` supplies the native sampling,
  predictive, graph and joint log-likelihood contracts. The flat prior bridge
  uses PyMC's public `compute_log_prior` with `model.pymc_model`.
- Bambi 0.19 constructs observed `rt,response_extra_dim_0` and predicted
  `rt,response_dim` axes, both labeled `[0,1]` for `[RT,response]`. The adapter
  accepts the differing names while enforcing sample/trial dimensions, exact
  matching observation coordinates, positive finite RTs and choices −1/+1.
- Scalar RT and binary +1-choice views contain only observed/predictive groups.
  The original joint log-likelihood stays with the joint artifact. No joint
  calibration guarantee follows from the two marginal PPC-PIT checks.
- Skill frontmatter, relative links and embedded example syntax are checked.
  Python/Ruff and strict marimo checks assess the implementation's structure;
  they do not execute statistical code or establish package compatibility.
- Pytest collection finds 44 cases: 43 deterministic adapter cases and one
  opt-in full notebook integration. No test bodies were executed.
- Independent static review checked the marginal helper CLI arguments, the
  scalar analytical prior guards and HSSM-specific next steps derived from
  shared ratings. Numerical behavior remains to be verified.

Static commands ran with the existing `baygent-bambi-m1` tooling environment,
without installing HSSM there or importing/executing the notebook:

```bash
ruff check hssm-workflow/scripts examples/hssm-workflow evals/smoke/test_hssm_workflow.py
ruff format --check hssm-workflow/scripts examples/hssm-workflow evals/smoke/test_hssm_workflow.py
marimo check --strict examples/hssm-workflow/analytical_ddm.py
python -m pytest --collect-only -q evals/smoke/test_hssm_workflow.py
git diff --cached --check
```

Additional parsing checked nine embedded Python snippets, five shell snippets,
37 repository Markdown links, evaluation JSON and candidate YAML syntax.
The skill frontmatter validator and all 11 installed-skill local links passed
in an isolated copy containing only `hssm-workflow` and `bayesian-workflow`.
That copy check validates installation structure, not HSSM package execution.

## Deferred execution gates

1. Resolve the candidate environment on the intended Python/platform and record
   every runtime dependency. Confirm simulator, plotting and netCDF dependencies.
2. Run `python -m pytest evals/smoke/test_hssm_workflow.py` for the authored
   deterministic adapter checks. They use literal array fixtures and are not
   evidence of a fitted HSSM model.
3. Run `BAYGENT_TEST_HSSM=1 python -m pytest evals/smoke/test_hssm_workflow.py`
   to include the explicitly gated real notebook integration path. It will
   generate synthetic data and sample; its run budget is deliberately visible.
4. Independently verify effective prior densities and selected joint analytical
   log-likelihood values, positive RT/choice semantics, and the scalar views.
   Finite log likelihood is insufficient: HSSM floors impossible `rt <= t`
   observations, and the example explicitly bounds `t` below the observed RTs.
5. Execute and inspect the full notebook/report, including the RT/choice domain
   checks, actual prior assessment, all figure links and sensitivity limitations.
   Missing or failed checks must remain visible; do not equate a completed fit
   with adequacy or a teaching example with a parameter-recovery experiment.
6. Run independent paired with-skill/without-skill scenarios and trigger cases,
   retaining actual outputs, grading evidence and timing. Do not infer scores
   from the authored metadata or from source inspection.

The next broader model milestone is a hierarchical condition contrast with
HSSM's own safe-prior/link policy. LAN artifact-domain checks, RLSSMs, missing
RTs, deadlines and lapse regressions remain separate increments.
