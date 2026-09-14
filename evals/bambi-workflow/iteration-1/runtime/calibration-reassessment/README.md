# Bambi saved-draw calibration reassessment

These are the current reports for the original four-chain Bambi teaching fits.
They were regenerated from the existing posterior and predictive draws, with
`Model.fit`, `Model.prior_predictive` and `Model.predict` replaced by sentinels
that fail if called. This reassessment generated no new fitted or predictive
draws. Original full-run provenance is in [execution.json](../execution.json).

| Current report | Raw PIT p-value | Coverage p-value | Retained assessment |
|---|---:|---:|---|
| [Gaussian regression](gaussian-regression/report.md) | 0.481679 | 0.456364 | Both calibration checks pass; low prior sensitivity. |
| [Hierarchical survey](hierarchical-bernoulli/report.md) | 0.231972 | 0.350566 | Both calibration checks pass; strong sensitivity for `1\|region_sigma`. |

The native `pot_c` tests use α=0.01. Assessment and plots share the same PIT
values; coverage uses `2 * abs(PIT - 0.5)` once. The corrected plots show ΔECDF
on the original probability axis, with p-values matching `calibration.json`.
See the [shared method and compatibility evidence](../../../../calibration-method-consistency/README.md).
Passing fitted-data marginal PPC-PIT checks does not establish conditional,
joint or held-out calibration. The survey still requires a justified regional
scale prior and sensitivity comparison for its requested probability.

[Reassessment execution.json](execution.json) records the source hashes, original
NetCDF/input hashes and exact numerical assessments. All four shared/helper
source hashes, both notebook hashes and ten original input hashes were checked
before copying. Large `.nc` files remain in the gitignored example results;
reports, figures, CSVs and JSON are copied here without changing their contents.
The original and reassessed posterior/prior DataTrees were also compared: all
four files retain identical groups, arrays and coordinates.
All report image links and calibration JSON plot paths resolve locally. The four
corrected PIT/coverage plots and survey sensitivity plot were visually inspected.

The convergence tables now fill missing shared extrema from the unrounded
selected-parameter `summary.csv`, while retaining every shared pass/flag status.
For Gaussian regression the displayed maxima/minima are R-hat 1.003, bulk ESS
5684 and tail ESS 2888; the survey shows 1.002, 951.4 and 969.2. Raw values remain
in each CSV; these selected summaries do not replace the full shared assessment.

Two distinct validation records accompany this reassessment:

- [Bambi integration regression](bambi-calibration-regression.xml): **13 passed,
  27.420 s** after the calibration fix. This separately authorized check ran both
  notebooks at two chains with 150 tuning and 150 retained draws; it is not the
  no-fit reassessment described above.
- [Report display regression](bambi-reporting-regression.xml): **5 passed** using
  literal summary/diagnostic inputs, with no model construction or fitting.

The original notebook HTML files remain historical views from before the
calibration fix; this directory's Markdown reports and figures supersede their
calibration output. No replacement executed HTML is claimed.
