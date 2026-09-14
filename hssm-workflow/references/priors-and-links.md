# Priors, Support and Links

## Three different objects

The prior expresses uncertainty, bounds restrict the parameter domain, and a
regression link maps a linear predictor to a likelihood parameter. This flat
DDM has no covariate or group formulas: specify the physical parameters
directly. Do not add `link=` to a simple non-regression `hssm.Param`; that
combination is rejected. The `link_settings="log_logit"` preset affects
regression parameters, not these flat parameter priors.

A LogNormal prior on positive `a` uses log-scale distribution arguments while
the resulting `a` remains a likelihood-scale value. That is distinct from a
log link on a regression, or the sampler's internal unconstraining transform.
Inspect the final HSSM parameter specifications instead of substituting a
Bambi regression prior recipe.

## Configure non-decision-time support

For complete positive RTs without a lapse mixture, decision time is `rt - t`.
Choose a scientifically justified prior whose upper support is **strictly below
the minimum retained RT**. The lower/upper limits below are illustrative
configuration values in seconds, not defaults that fit every experiment:

```python
t_lower = 0.0
t_upper = 0.20
if not 0 <= t_lower < t_upper < float(data["rt"].min()):
    raise ValueError("Review the configured t prior: its upper bound must be below min(rt).")
```

Review units, measurement artifacts and the intended data regime if this fails.
Do not discard the fastest observations or silently squeeze a prior into a
data-dependent range. If using an observed-data constraint to define support,
document that choice and its implications for prior interpretation/sensitivity.

HSSM 0.5.0's analytical DDM returns a finite numerical floor for sufficiently
small or invalid decision times. A finite initial log probability therefore
does not prove that `t` has physically valid support.

## Construct the flat model

This continues from the validated real `data` and configured `t` limits. These
priors are examples to elicit and check, not universal psychological defaults.
All four parameters are shared across trials; no hierarchy is introduced.

```python
import hssm
import numpy as np

model = hssm.HSSM(
    data=data,
    model="ddm",
    loglik_kind="analytical",
    choices=[-1, 1],
    p_outlier=None,
    include=[
        hssm.Param(name="v", prior={"name": "Normal", "mu": 0, "sigma": 1.5}),
        hssm.Param(
            name="a", prior={"name": "LogNormal", "mu": 0, "sigma": 0.4},
            bounds=(0.0, np.inf),
        ),
        hssm.Param(
            name="z", prior={"name": "Beta", "alpha": 2, "beta": 2},
            bounds=(0.0, 1.0),
        ),
        hssm.Param(
            name="t", prior={"name": "Uniform", "lower": t_lower, "upper": t_upper},
            bounds=(t_lower, t_upper),
        ),
    ],
)
print(model)
for parameter_name, parameter in model.params.items():
    print(parameter_name, parameter)
```

The drift prior permits either direction; the boundary prior is positive;
the starting-position prior favors the interior without fixing `z=0.5`; the
non-decision-time prior encodes its declared seconds-scale limits. Explain
these choices in relation to the task and inspect what HSSM actually built.
For an intentionally fixed parameter, use a numeric prior and report the
restriction; fixing it changes the inference problem.

HSSM accepts dictionary specs and `hssm.Param` objects. Its simple-parameter
processing can turn a dictionary prior plus bounds into a truncated prior.
Verify the effective support and distribution; recording only a bounds tuple
does not replace inspecting the constructed prior.

## Prior predictions and model revisions

When generation is in scope, use `model.sample_prior_predictive(...)` and the
HSSM plots described in [prediction-and-reporting.md](prediction-and-reporting.md).
Inspect RT tails, very fast responses, and the range of choice proportions.
State which patterns are plausible and which require revision. Finite draws
or a saved figure alone are not a scientific prior check.

Changing priors after fitting requires a clearly identified new model/run;
do not silently attach old posterior draws to a revised specification.

## Extensions and sensitivity

`prior_settings="safe"` fills eligible missing regression-term priors. It does
not make every explicit simple prior safe. Hierarchical HSSM has package-specific
rules about centered versus non-centered group terms, bounds and links; honor
those rules when extending the scope instead of enforcing universal
non-centering.

HSSM has no public log-prior method in this release. For this **flat model
only**, the source-checked bridge is PyMC's public `compute_log_prior` with
`model.pymc_model`. HSSM preserves the free parameter names `v`, `a`, `z`, `t`
in the posterior; there are no covariate-centering adjustments or group offsets
in this specification. Do not copy the Bambi example's centering workaround
or assume this bridge applies unchanged to later hierarchical models.

After fitting, before sensitivity or saving the final combined artifact:

```python
import pymc as pm

free_names = {rv.name for rv in model.pymc_model.free_RVs}
if not free_names.issubset(set(idata["posterior"].data_vars)):
    raise ValueError("Saved posterior is missing free variables needed for prior density.")
pm.compute_log_prior(idata, model=model.pymc_model, progressbar=False)
if set(idata["log_prior"].data_vars) != free_names:
    raise ValueError("Prior density must cover exactly the model's free random variables.")
```

Completeness is necessary but does not prove correct density values. The
deferred execution check must also compare these densities with independent
calculations of the **effective** Normal, bounded LogNormal, bounded Beta and
Uniform priors, including any truncation normalization. Verify support and
chain/draw alignment. Record the actual HSSM/PyMC/Bambi versions. Until these
checks have run, leave sensitivity explicitly unassessed; the source review
alone does not establish numerical correctness or robustness. Continue with
the shared sensitivity and reporting handoff in
[prediction-and-reporting.md](prediction-and-reporting.md) after verification.

Sources: HSSM 0.5.0
[parameter interface and presets](https://github.com/lnccbrown/HSSM/blob/fefed57d2142637af503b0e92cefe799715c0f46/src/hssm/hssm.py),
[simple-parameter handling](https://github.com/lnccbrown/HSSM/blob/fefed57d2142637af503b0e92cefe799715c0f46/src/hssm/param/simple_param.py),
[bounded priors](https://github.com/lnccbrown/HSSM/blob/fefed57d2142637af503b0e92cefe799715c0f46/src/hssm/prior.py),
and [hierarchical safe-prior policy](https://github.com/lnccbrown/HSSM/blob/fefed57d2142637af503b0e92cefe799715c0f46/src/hssm/param/regression_param.py).
