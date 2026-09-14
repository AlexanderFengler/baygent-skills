# HSSM workflow validation

The flat analytical-DDM example passed local runtime and numerical validation on
HSSM 0.5.0 / Bambi 0.19 / PyMC 6.1 / ArviZ 1.2. See the
[runtime record](runtime/README.md) for the tested environment, commands,
source hashes and limitations. These measurements describe the recorded
historical development commits; they are not agent-performance benchmarks.

The recorded full export and saved-artifact suite passed 171 tests (139 HSSM
and 32 shared calibration checks), with one intentional duplicate-fit skip and
four strict expected failures for native sample-coordinate defects.
That broader suite is [archived on the development fork](https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/hssm-workflow/iteration-1/runtime/README.md).
The maintained suite now focuses on our adapter, notebook, reporting and compact
prior/likelihood integration contracts; it excludes saved-export auditing and
broad dependency certification. Historical totals are not current suite counts.

The [skill](../../../hssm-workflow/SKILL.md) depends directly on
`bayesian-workflow`. Its adapter preserves paired RT/choice labels and creates
separate scalar PPC views without duplicating the joint likelihood. Joint-trial
LOO and prior sensitivity retain the original paired artifact.

## Scientific scope

Convergence and joint-trial LOO were excellent, and prior sensitivity was low.
The RT marginal assessment was **fair**: coverage-transformed PIT rejected
uniformity (`p=3.1641e-15`); ordinary RT PIT and both choice tests passed. The
[archived report](https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/hssm-workflow/iteration-1/runtime/analytical-ddm/report.md) retains the discrepancy
and recommends examining conditional RT tails and model assumptions. One
teaching dataset does not establish recovery or joint/held-out calibration.

## Agent evaluation status

[Scenario metadata](ddm-flat/eval_metadata.json), additional
[frozen scenarios](../iteration-2/README.md), and the
[trigger set](../trigger_eval_set.json) are prepared. No paired comparison or
trigger evaluation has completed; scores in [benchmark.json](benchmark.json)
remain null. One [archived baseline attempt](https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/hssm-workflow/iteration-3-handoff-pair/README.md)
exhausted its token budget before producing a final review, and its treatment
was never launched. It provides no evidence of skill improvement or failure.

Hierarchical models, LAN likelihoods, RLSSMs, missing RTs/deadlines, formal SBC,
and a platform compatibility matrix remain outside this contribution.
