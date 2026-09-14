# Formula Syntax and Data Encoding

## Formula notation

Bambi uses Formulae to parse Wilkinson-style formulas. These are the core
forms needed by the initial Gaussian and Bernoulli workflows:

| Form | Meaning |
|---|---|
| `y ~ x` | Intercept and common predictor effect. |
| `y ~ x + z` | Additive common effects. |
| `y ~ x * z` | Main effects for `x` and `z`, plus their interaction. |
| `y ~ x:z` | Interaction term; include main effects when the model requires them. |
| `y ~ 0 + x` | Remove the common intercept. |
| `y ~ x + (1|group)` | Common effect of `x` and a varying intercept across groups. |

A formula expresses structure. The likelihood, link, priors and prediction
semantics are chosen by the modeling package. A downstream skill can reuse
this notation without inheriting Bambi constructor or sampling instructions.

## Check the analysis rows

Verify that formula columns exist, continuous predictors vary, response values
are valid, and the grouping variable has the intended units and levels. Decide
how to handle missing data explicitly; document retained rows and exclusions.
Sparse groups can benefit from partial pooling, but their individual estimates
may remain uncertain. Inspect the counts rather than treating every group as
equally informative.

Center/scale predictors when it makes priors and interpretation clearer. Record
the transformation and apply the same transformation to prediction data.
Bambi's internal predictor centering is not a substitute for documenting the
units in which an effect or contrast is reported.

## Categorical predictors and reference levels

Bambi converts object columns to categorical during construction. Numerical
category codes need explicit treatment, such as `categorical=["region"]`.
Neither that argument nor `C(x)` alone chooses a domain-meaningful reference.

For the survey example, preserve a known level order and inspect the resulting
encoding. Here `df` is the validated analysis DataFrame:

```python
import bambi as bmb
import pandas as pd

income_levels = ["low", "medium", "high"]
assert df["income_bracket"].isin(income_levels).all()
df = df.copy()
df["income_bracket"] = pd.Categorical(
    df["income_bracket"], categories=income_levels, ordered=True
)
df["region"] = df["region"].astype("category")
df["age_centered"] = df["age"] - 40

model = bmb.Model(
    "support ~ age_centered + income_bracket + urban + (1|region)",
    df,
    family="bernoulli",
    link="logit",
    center_predictors=False,
)
income = model.parameters["p"].common_terms["income_bracket"]
print(income.levels)
print(income.coords)
print(income.data[:5])
```

With the intercept and ordinary treatment coding, check that `low` is the
omitted baseline, that the reported columns correspond to `medium` and `high`,
and that a few encoded rows match their labels. Verify this after changing
contrasts or intercept handling. With Formulae 0.6.2, an **unordered** pandas
Categorical can be encoded in alphabetical order despite its `categories`
list; the ordered dtype above preserves the intended low-income baseline.
Ordering here controls level order, not an ordinal likelihood or a numerical
dose effect. Unknown categories must not turn into missing values silently
when constructing new prediction rows.

## Binary outcome coding

Use numeric `0=no, 1=yes` for the supported Bernoulli example and record the
mapping. Confirm it against the model's encoded response:

```python
assert df["support"].isin([0, 1]).all()
assert model.response_term.success == 1
print(model.response_term.data)
```

For a categorical/string response, inspect the selected success event before
interpreting `p`; category order can change which event is modeled. Recoding
must preserve the scientific question and be documented.

## Grouped predictions

For the first workflow, predict for an explicitly named **observed** region.
Keep the training categorical levels in new data and pass the centered age
used by the formula. Setting group effects to zero, averaging existing groups,
and predicting a previously unseen group are different targets; none is an
automatic substitute for that request. See
[interpretation-and-reporting.md](interpretation-and-reporting.md).

Source: Bambi 0.21.0
[model construction and category conversion](https://github.com/bambinos/bambi/blob/3f795e07c8ec87d26f48e476b98e5579e0eb1a42/bambi/models.py)
and [term encoding](https://github.com/bambinos/bambi/blob/3f795e07c8ec87d26f48e476b98e5579e0eb1a42/bambi/terms/common.py).
