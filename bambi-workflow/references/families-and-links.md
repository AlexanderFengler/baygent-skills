# Families and Links

## Supported likelihoods

Choose the likelihood from the outcome's meaning and support. Bambi's
constructor defaults to `family="gaussian"`; it does not select the intended
likelihood by examining the outcome's dtype.

| Outcome and question | Family and link | Parent parameter | Other parameter |
|---|---|---|---|
| Continuous response with approximately Gaussian residual variation | `family="gaussian", link="identity"` | `mu`, on the outcome scale | Positive residual scale `sigma` |
| A single binary trial, coded `0/1` | `family="bernoulli", link="logit"` | `p = Pr(Y=1)`, between zero and one | None |

The formula models the linked parent parameter. Gaussian regression models
`mu` directly; logistic regression models `logit(p)`. The Bernoulli intercept
and slope priors therefore live on the log-odds scale. Their effect on
probability depends on the other predictors and, in a hierarchical model,
the selected group.

```python
gaussian = bmb.Model(
    "y ~ x", df, family="gaussian", link="identity", center_predictors=False
)
survey = bmb.Model(
    "support ~ age_centered + income_bracket + urban + (1|region)",
    survey_df,
    family="bernoulli",
    link="logit",
    center_predictors=False,
)
print(gaussian.family)
print(survey.family)
```

Here `bmb` is `import bambi as bmb`; both DataFrames must contain the named
columns with validated encoding. The formula reference supplies the survey
encoding and event checks. For these examples with explicit predictor scales,
`center_predictors=False` keeps saved intercept draws in the prior coordinates;
see the [Bambi 0.21 sensitivity limitation](priors-in-bambi.md#prior-sensitivity-after-fitting).

## Constant residual scale is legitimate

For `y ~ x` with a Gaussian family, `sigma` can be a single uncertain parameter
shared across observations. This is a valid homoscedastic model; it does not
need its own regression formula merely because it is an auxiliary parameter.
Inspect its prior using `model.marginal_parameters["sigma"].prior` and judge
whether shared residual variation is plausible.

If predictive residuals vary systematically by a covariate, a distributional
model may be appropriate. That is an extension beyond the initial workflow:
consult the release's `bmb.Formula` and parameter-specific prior/link guidance
instead of adding formulas for every auxiliary parameter automatically.

## What the prediction means

For either family, `model.predict(..., kind="response_params")` returns
posterior draws of the likelihood parameters at the chosen predictor values.
`kind="response"` additionally simulates outcomes. A Bernoulli draw is 0 or 1;
a draw of `p` is a probability. Use uncertainty in `p` when answering a question
about probability and predictive outcomes when checking replicated data.

For Gaussian models, distinguish an interval for `mu` from a prediction interval
for a future `y`, which also includes residual variation. State the quantity
and interval type in tables and prose.

## Extensions

Counts, proportions with denominators, ordinal outcomes, heavy tails, and
zero-inflated data may require different families. Recognize these as valid
Bambi questions, state the initial skill's narrower support, and inspect
current release APIs before extending the implementation. Do not force them
into Gaussian or single-trial Bernoulli models to fit these examples.

Source: Bambi 0.21.0
[model API](https://github.com/bambinos/bambi/blob/3f795e07c8ec87d26f48e476b98e5579e0eb1a42/bambi/models.py)
and [univariate families](https://github.com/bambinos/bambi/blob/3f795e07c8ec87d26f48e476b98e5579e0eb1a42/bambi/families/univariate.py).
