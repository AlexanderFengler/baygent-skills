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
joint-calibration guarantee. No paired agent or trigger evaluation has completed.
The corrected report retains an RT marginal calibration flag; passing the
software checks does not make every scientific diagnostic pass.

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
the tested uv setup and exact resolved versions. Keep the virtual environment
outside tracked source directories.

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

Run the maintained software checks in the HSSM environment:

```bash
python -m pytest -q evals/smoke/test_hssm*.py
```

These cover our adapter/CLI, domain summaries, execution gates, report failures,
and the example's declared priors/support and joint-likelihood handoff. The
numerical handoff uses the separate compiled `hddm_wfpt` reference recorded in
the runtime requirements. Ordinary checks do not fit; `BAYGENT_TEST_HSSM=1`
explicitly opts into a full notebook integration run.

[Validation status](../../evals/hssm-workflow/iteration-1/README.md) separates
these maintained checks from historical full-export validation and the
behavioral evaluation planned for a follow-up PR. Broader HSSM/PyMC numerical
certification and the saved-export audit are
[preserved on the development fork](https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/hssm-workflow/iteration-1/runtime/README.md).
Their historical totals are not the current suite's test count.

The notebook still rejects nondefault chain/draw labels before native density
recomputation, and the scalar adapter preserves arbitrary unique labels. The
notebook guard and native QP plotting workaround have local regression checks;
we do not duplicate the dependencies' coordinate-defect tests here.
