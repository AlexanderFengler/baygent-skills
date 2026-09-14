# Data and the Analytical DDM

## Define the observational unit

This first workflow models each completed trial as a pair: a positive reaction
time and a choice between two boundaries. A flat model estimates one parameter
set for those trials. It does not account for participant heterogeneity,
learning, condition effects, trial dependence, censoring, or a lapse process.
Pooling incompatible participants or conditions changes the scientific model;
it is not a harmless way to satisfy the flat-model interface.

Confirm the response mapping and which events the RT clock measures. DDM
parameters are interpretable only under the task and measurement assumptions.
An RT regression and a sequential-sampling model answer different questions;
honor the requested model class.

## Validate a real dataset

The snippet assumes `trials.csv` is the user-selected input and RTs have already
been documented as seconds. It reads and validates data; it generates none.

```python
import numpy as np
import pandas as pd

data = pd.read_csv("trials.csv")
required = ["rt", "response"]
if not set(required).issubset(data.columns):
    raise ValueError("The data require rt and response columns.")
if data.empty or data[required].isna().any().any():
    raise ValueError("This workflow requires nonempty, complete RT/choice rows.")
if not np.isfinite(data[required].to_numpy(dtype=float)).all():
    raise ValueError("RT and response must be finite numeric values.")
if not (data["rt"] > 0).all():
    raise ValueError("This workflow requires strictly positive RTs in seconds.")
if not data["response"].isin([-1, 1]).all():
    raise ValueError("Document and apply the intended -1/+1 choice mapping first.")
choice_counts = data["response"].value_counts().reindex([-1, 1], fill_value=0)
print(choice_counts)
print(data.groupby("response")["rt"].describe())
```

Before modeling, also establish that these are complete responses rather than
timeouts or deadline-censored observations. Missing trials need not appear as
NaNs in the retained table. Record exclusions and the raw/retained counts;
do not let the model's missing-data behavior make that decision implicitly.

When source RTs are milliseconds, convert explicitly and record the factor.
When source choices are `0/1`, map their documented meanings to `-1/+1` and
retain the mapping. Do not infer “correct” from the larger numerical code:
stimulus, response boundary, and accuracy are different variables. Signed RTs
also require deliberate decoding; this initial input convention keeps RT and
choice in separate columns.

If one choice is absent or very rare, flag the resulting weak information
about its RT distribution and about model components. A valid label check
does not establish that both conditional distributions can be assessed.

## Select the DDM likelihood

```python
import hssm

print(hssm.show_defaults("ddm", "analytical"))
```

Use `model="ddm", loglik_kind="analytical", choices=[-1, 1]` when constructing
this analysis. The release's DDM configuration supports multiple likelihood
implementations; choosing the analytical one avoids a LAN training-domain
claim. It does not remove numerical approximations inside its density.

Specify `p_outlier=None` for this clean, no-lapse model. HSSM otherwise defaults
to a fixed lapse probability of 0.05. Removing a lapse mixture is a substantive
assumption: if contaminants or nonresponses are part of the problem, extend
the model rather than silently excluding them or treating `None` as a universal
best practice.

## Parameter meanings and conventions

| Parameter | Meaning in this DDM | Relevant support |
|---|---|---|
| `v` | Drift direction and magnitude | Real-valued |
| `a` | Positive boundary magnitude | Positive |
| `z` | Starting position as a fraction between boundaries | Between zero and one |
| `t` | Non-decision time added to decision time | Nonnegative and below every retained RT for this no-lapse model |

The analytical implementation converts `a` to a full separation `2*a` internally.
Check conventions before comparing a boundary parameter to another package or
paper. Likewise, interpret the sign of drift and starting bias relative to the
declared response boundaries, not an assumed accuracy label.

Read [priors-and-links.md](priors-and-links.md) before construction, particularly
the non-decision-time support constraint.

Sources: HSSM 0.5.0
[DDM configuration](https://github.com/lnccbrown/HSSM/blob/fefed57d2142637af503b0e92cefe799715c0f46/src/hssm/modelconfig/ddm_config.py),
[analytical density](https://github.com/lnccbrown/HSSM/blob/fefed57d2142637af503b0e92cefe799715c0f46/src/hssm/likelihoods/analytical.py),
and [data validation](https://github.com/lnccbrown/HSSM/blob/fefed57d2142637af503b0e92cefe799715c0f46/src/hssm/data_validator.py).
