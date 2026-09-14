# HSSM analytical DDM example

[analytical_ddm.py](analytical_ddm.py) is a narrated **marimo notebook** for a
flat analytical DDM. It covers explicit priors, positive reaction times in
seconds, declared choices −1/+1, native HSSM predictions, choice proportions and
conditional RT quantiles, and baygent's shared diagnostic report.

**Runtime validated; skill behavior pending.** Deterministic contracts,
independent density checks, a full notebook export and saved-artifact checks
passed on the recorded HSSM 0.5 stack. See the [runtime evidence](../../evals/hssm-workflow/iteration-1/runtime/README.md)
for commands, resolved dependencies, figures and actual assessments. This is
one synthetic teaching dataset, not a parameter-recovery experiment or a
joint-calibration guarantee. Paired agent and trigger evaluations have not run.

## Open without running

Use the separate [environment-hssm.yml](../../environment-hssm.yml) package set.
HSSM 0.5 requires NumPy below 2.5, so do not reuse the Bambi example environment
unchanged. The tested installation used uv with Python 3.12.13 on macOS arm64,
the recipe's pip requirements and system Graphviz. The conda/mamba recipe below
is an alternative setup; it was not itself executed during validation.

```bash
uv venv --python 3.12 .venv-hssm
uv pip install --python .venv-hssm/bin/python \
  -r evals/hssm-workflow/iteration-1/runtime/requirements-resolved.txt
source .venv-hssm/bin/activate
dot -V  # Requires a system Graphviz installation for this uv setup.
marimo edit examples/hssm-workflow/analytical_ddm.py
```

Or create the untested conda alternative:

```bash
mamba env create -f environment-hssm.yml
conda activate baygent-hssm
marimo edit examples/hssm-workflow/analytical_ddm.py
```

Graphviz's `dot` executable is included in that conda recipe. The
[runtime record](../../evals/hssm-workflow/iteration-1/runtime/README.md) records
the tested uv setup and exact resolved versions. Keep personal environments in
Spine's `_local/venvs/` when working from HSSMSpine.

Opening the notebook leaves it in preview mode. Its run button explicitly opts
into generating the teaching data and prior predictions. A second button
continues to fitting after the reader reviews the prior checks. The final choice
selector only filters an already computed diagnostic table; it never refits.

## Run or export

For an explicitly requested command-line run/export, set `BAYGENT_RUN_HSSM=1`:

```bash
mkdir -p examples/hssm-workflow/results
BAYGENT_RUN_HSSM=1 marimo export html examples/hssm-workflow/analytical_ddm.py \
  -o examples/hssm-workflow/results/analytical_ddm.html
```

This explicit command-line opt-in runs the whole workflow, bypassing both
interactive review buttons; inspect its saved prior checks when assessing the run.
It is a real computation: 300 generated trials, two chains, 1,000 tuning and
1,000 retained draws per chain, using HSSM's `pymc` sampler and one process.
The environment variable is off by default. Use `--force` to replace an export.
`BAYGENT_OUTPUT_ROOT=/another/folder` changes the results root; rerunning writes
to the same analysis folder. Generated results are gitignored.

The notebook checkpoints immediately after sampling. The completed artifact
retains one joint RT/choice response and one joint log-likelihood per trial.
HSSM 0.5's posterior predictive method has no public random-seed argument; the
notebook records this limitation instead of claiming seeded repeatability of
those draws. The non-decision-time prior has an explicitly data-informed upper
bound below the minimum RT; power-sensitivity checks do not assess changes to
that support policy.

## Report and artifact contract

`hssm_example_support.py` only assembles example plots and the canonical report.
The standalone installed skill does not depend on this notebook or helper.
The installed [RT/choice adapter](../../hssm-workflow/scripts/prepare_rt_choice.py)
validates the paired artifact and writes two scalar PPC views:

```text
results/analytical-ddm/
  posterior.nc                 # immediate fit checkpoint
  inference_data.nc            # original joint response, PPC and joint densities
  prior_data.nc
  data.csv
  domain_checks.csv            # choice rates and conditional RT quantiles
  summary.csv
  diagnostics.json             # shared convergence + joint-trial LOO
  psense.json
  check_report.json            # shared convergence/LOO/sensitivity assessment
  report.md                    # canonical sections, actual results when run
  model_graph.png
  prior_predictive.png
  posterior_predictive.png
  quantile_probability.png
  trace.png
  forest.png
  psense.png
  calibration/
    manifest.json
    rt/                        # scalar marginal RT in seconds
      inference_data.nc
      calibration.json
      check_report.json
      pit_ecdf.png
      pit_coverage.png
    choice/                    # scalar event response == +1; same filenames
```

Shared helpers assess each scalar margin independently. The scalar artifacts
omit the joint log-likelihood, so they must not be used for marginal LOO-PIT.
Passing both fitted-data PPC-PIT checks does not establish joint, conditional,
or held-out calibration; the choice/conditional-RT checks supply separate domain
evidence. No diagnostic rating is reimplemented. Two legacy template
explanations are adjusted for modern draw-order trace plots and the distinction
between PIT location patterns and coverage direction. The report and notebook
translate shared ratings into HSSM-specific next steps; the raw shared output
is preserved in JSON.

## Checks and remaining evaluations

Static checks do not execute the notebook:

```bash
marimo check --strict examples/hssm-workflow/analytical_ddm.py
ruff check hssm-workflow/scripts examples/hssm-workflow evals/smoke/test_hssm*.py
ruff format --check hssm-workflow/scripts examples/hssm-workflow evals/smoke/test_hssm*.py
```

[Validation status](../../evals/hssm-workflow/iteration-1/README.md) distinguishes
the completed adapter/CLI, domain-summary, execution-gate, report-failure,
numerical-density and full-run checks from pending paired agent and trigger
evaluations. The numerical suite uses the separate compiled `hddm_wfpt` reference
recorded in the runtime requirements. The ordinary adapter suite retains an
opt-in full-fit integration test; validation instead ran the full HTML export
once successfully and checked its saved artifacts without a duplicate fit.

Four strict expected failures document two native density-coordinate defects
across two `t`-support cases. The notebook rejects nondefault chain/draw labels
before density recomputation; the scalar adapter still preserves arbitrary
unique labels. HSSM's QP call also needs the documented release workaround for
`quantile_by="response"` and explicit `ax`; a native literal-data regression
checks the actual plotted points. See the runtime record for exact commands,
test totals and the preserved failed export attempts.
