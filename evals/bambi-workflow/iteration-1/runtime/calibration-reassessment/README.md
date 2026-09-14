# Bambi saved-draw calibration reassessment

The reports linked below were regenerated from the original four-chain Bambi
teaching fits, with `Model.fit`, `Model.prior_predictive` and `Model.predict`
replaced by sentinels that fail if called. No new fitted or predictive draws were
generated. [Original execution.json](../execution.json) records the full runs.
All measurements and hashes describe development sources, not a fresh PR run.

| Archived report | Raw PIT p-value | Coverage p-value | Retained assessment |
|---|---:|---:|---|
| [Gaussian regression](https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/bambi-workflow/iteration-1/runtime/calibration-reassessment/gaussian-regression/report.md) | 0.481679 | 0.456364 | Both calibration checks pass; low prior sensitivity. |
| [Hierarchical survey](https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/bambi-workflow/iteration-1/runtime/calibration-reassessment/hierarchical-bernoulli/report.md) | 0.231972 | 0.350566 | Both calibration checks pass; strong sensitivity for `1\|region_sigma`. |

The native `pot_c` tests use α=0.01. Assessment and plots share the same PIT
values; coverage uses `2 * abs(PIT - 0.5)` once. The corrected plots show ΔECDF
on the original probability axis, with p-values matching `calibration.json`.
See the [shared method and compatibility evidence](../../../../calibration-method-consistency/README.md).
Passing fitted-data marginal PPC-PIT checks does not establish conditional,
joint or held-out calibration. The survey still requires a justified regional
scale prior and sensitivity comparison for its requested probability.

[Reassessment execution.json](execution.json) preserves source hashes, original
NetCDF/input hashes, exact numerical assessments and archived output locations.
The [complete report bundles](https://github.com/AlexanderFengler/baygent-skills/tree/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/bambi-workflow/iteration-1/runtime/calibration-reassessment/)
contain figures, CSVs and JSON at the preserved fork commit. Large NetCDF files
were never committed and are not available from that archive. Maintained tests
create temporary outputs; they do not depend on these report bundles.

At the original reassessment, all four shared/helper source hashes, both notebook
hashes and ten original input hashes were checked before copying. The original
and reassessed posterior/prior DataTrees retained identical groups, arrays and
coordinates in all four files. Report image links and calibration plot paths
resolved within the full bundles. The four corrected PIT/coverage plots and the
survey sensitivity plot were visually inspected.

The convergence tables fill missing shared extrema from the unrounded
selected-parameter `summary.csv`, while retaining every shared pass/flag status.
For Gaussian regression the displayed maxima/minima are R-hat 1.003, bulk ESS
5684 and tail ESS 2888; the survey shows 1.002, 951.4 and 969.2. Raw values remain
in the archived CSVs; these selected summaries do not replace the full assessment.

Two archived validation records accompany this reassessment:

- [Bambi integration regression](https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/bambi-workflow/iteration-1/runtime/calibration-reassessment/bambi-calibration-regression.xml):
  **13 passed, 27.420 s** after the calibration fix. This separate check ran both
  notebooks at two chains with 150 tuning and 150 retained draws.
- [Report display regression](https://github.com/AlexanderFengler/baygent-skills/blob/b5d6d0702f3f0d8dc73d5a6247b733dc79c60131/evals/bambi-workflow/iteration-1/runtime/calibration-reassessment/bambi-reporting-regression.xml):
  **5 passed** using literal summary/diagnostic inputs, without model fitting.

The original notebook HTML files remain historical views from before the
calibration fix. The archived reassessed reports and figures supersede their
calibration output; no replacement executed HTML is claimed.
