# Detailed results analysis (source material for the full paper)

This document works through every result in `results/metrics/` in the
depth needed to write the paper's Results and Discussion sections. Numbers
are pulled directly from the committed JSON/CSV files, not re-derived.

## 1. Expected-output regression benchmark

Chronological-split test RMSE clusters tightly across six of eight models:
MLP 63.73, XGBoost 65.18, LightGBM 65.37, Linear 66.34, Ridge 66.38, Random
Forest 66.71 kW -- all within ~5% of each other. The naive baseline sits at
364.67 kW (R2 = -0.015), so every real model captures the vast majority of
learnable structure.

**What the narrow spread means.** Once irradiation, ambient/module
temperature and time-of-day are the inputs, the weather-to-power mapping is
close enough to a smooth, mostly-monotonic function of irradiation that
model capacity stops mattering much -- linear regression is barely worse
than the ensembles. The paper's claim here should be about *feature
framing dominating model choice* for this problem, not "our model beats
theirs."

**The CNN underperformed (69.16 kW), and that is worth reporting as-is.**
The 1D-CNN was built on the hypothesis that a short lag-window would
capture thermal lag (module heating/cooling affecting conversion
efficiency a few readings after an irradiance change). It did not help --
it did slightly worse than the pointwise tabular models, most likely
because (a) at 15-minute resolution there is little exploitable thermal
lag left to capture, and (b) building lag windows drops the first
`window-1` readings of every inverter-day, shrinking the effective
training set. This is a legitimate negative result, not a modeling
failure: it supports a specific physical claim (AC power response to
weather is close to instantaneous at this sampling rate) that a reviewer
can independently sanity-check.

**Vertex AI AutoML Tables (63.80 kW, R2 = 0.969) is statistically tied with
the best local model (MLP, 63.73 kW).** This is the most quotable applied
finding for a Digital Intelligence track audience: over an hour of managed
training and real cloud spend produced a model indistinguishable from a
five-layer MLP trained in seconds on a laptop CPU. Frame this as evidence
that for a well-posed, feature-engineered tabular problem, commercial
AutoML buys convenience, not accuracy -- a genuinely useful practitioner
takeaway, not just a "we compared against AutoML" checkbox.

**Chronological vs. random split is itself a finding.** An earlier
random-split experiment on the same data (not committed as code, but
referenced for contrast) produced RMSE around 46 kW -- roughly 30% lower
than the chronological-split numbers above. The gap is not modeling
improvement; it is a random split letting near-duplicate daylight readings
from the same or adjacent days leak across the train/test boundary,
overstating accuracy. This directly supports the paper's methodological
argument that *split choice materially changes the reported number* for
15-minute PV telemetry, and should be called out explicitly rather than
only reporting the better number.

## 2. Underperformance detection

Two inverters are flagged by the control-chart rule (daily performance
ratio below fleet median - 3xMAD on >=50% of the 34-day window):

- `bvBOhCH3iADSZry`: mean ratio 93.2%, flagged on 32/34 days
- `1BY6WEcLGh8j5v7`: mean ratio 94.7%, flagged on 30/34 days

The remaining 20 inverters range 100-111%, and none besides the two
flagged units exceeds 4/34 flagged days. **Persistence across the entire
34-day window, not a handful of days, is the strongest piece of evidence
that this is a structural fault** (soiling, partial/permanent shading
specific to that inverter's string layout, or a degraded electrical
connection) rather than measurement noise or a transient weather artifact
-- a one-off cloud shadow or bird dropping would show up as a few bad
days, not 88-94% of the entire study period.

**A nuance worth stating explicitly in the paper (it strengthens rather
than weakens the finding):** the expected-output model was trained on
pooled data from all 22 inverters, including the two underperforming
ones. That means the fleet's "expected" curve is pulled down very slightly
by the two bad actors, which if anything makes the reported ~93-95%
figures a *conservative* (understated) estimate of how underperforming
those two inverters really are relative to a fault-free reference. This
is the kind of self-aware methodological detail that reads as rigor to a
reviewer.

## 3. Cross-plant generalization

Training on all of Plant 1 and testing zero-shot on Plant 2 collapses R2
from ~0.966 (in-plant) to 0.255, with RMSE rising from ~66 kW to 337.6 kW
(best model: random forest). This is a genuine, unflattering domain-shift
result and should be reported as such, not minimized.

**Why it happens, and why that matters for the paper's contribution:**
Plant 1 and Plant 2 use different inverter/installation hardware in this
public dataset, so raw `AC_POWER` is not a directly comparable quantity
across sites -- a given irradiance level does not map to the same kW
figure at both plants. The practical implication, which belongs in
Limitations/Future Work, is that this method needs either (a) per-site
retraining, or (b) reformulating the target as a capacity-normalized
performance ratio *before* the regression step, rather than modeling raw
kW and only normalizing afterward for anomaly detection. That reframing
is a concrete, well-motivated future-work item -- exactly what a
"limitations that read as scoping, not hedging" section should contain.

## 4. Robustness to sensor degradation

Gaussian noise on the three sensor features degrades gracefully: RMSE
66.71 -> 68.84 (5% noise) -> 76.10 (10%) -> 98.68 (20%) -> 127.27 kW (30%).
Mean-imputed missing data degrades much faster at comparable severity:
66.71 -> 103.98 (5% missing) -> 130.98 (10%) -> 170.45 kW (20%).

**The asymmetry is the actual finding, and it has an operational
implication.** Irradiation carries ~94-99% of the model's predictive
weight (see interpretability below), and it swings from 0 to its
near-solar-noon maximum within a single day -- so replacing a missing
irradiance reading with its unconditional mean discards almost all
information in that reading, whereas additive Gaussian noise only
perturbs a genuine one. The paper should state the direct consequence for
deployment: a fielded version of this system should treat a missing
irradiance reading as "skip/flag this timestep," never silently
mean-impute it, because imputation can either mask a real fault (falsely
low expected value) or manufacture a false alarm (falsely high expected
value).

## 5. Uncertainty calibration

A nominal 90% interval built from the 5th/95th percentile spread of
individual random-forest tree predictions achieves only 55.7% empirical
coverage on the chronological test set -- substantially overconfident.

**Report this as an honest limitation, not a smoothed-over footnote.**
Tree-to-tree disagreement within one forest mostly reflects
bootstrap/estimation variance, not the full predictive uncertainty a
chronological (out-of-time) test period actually contains. The
correct fix, and the paper's specific, citable recommendation, is
conformal prediction (Angelopoulos & Bates, reference list below): a
distribution-free wrapper that would guarantee the stated coverage
regardless of this underlying miscalibration, without assuming the
forest's internal variance is a good uncertainty proxy. This turns a
negative result into a concrete, well-grounded future-work item instead
of an unresolved weakness.

It is also worth explicitly noting, in the same discussion, *why* the
underperformance-detection conclusion still holds despite this
miscalibration: the two flagged inverters' deficits (5-7% of expected
output, persistent over 32-34 days) are far larger and far more durable
than a single day's ~65 kW point-estimate RMSE against daylight AC power
readings that regularly exceed several hundred kW -- the anomaly signal
sits well outside the model's noise floor even if the interval width
itself is not trustworthy at face value.

## 6. Interpretability

SHAP mean |value| and the random forest's built-in impurity-based
importance agree on the ranking and are close in magnitude: irradiation
94.1% (SHAP) / 99.2% (impurity), with ambient temperature, module
temperature, and the two time-of-day encodings each contributing a few
percent or less by either measure. This cross-check is the paper's
evidence that the model has learned the correct physical driver of solar
output rather than a spurious correlate (e.g., time-of-day acting as a
proxy for irradiation) -- which is the basis for trusting it as an
"expected output" reference for anomaly detection in the first place.

The modest gap between the two importance measures (94% vs. 99%) is a
known property of impurity-based importance, which tends to overstate a
single dominant continuous feature relative to Shapley-value-based
measures that account for feature interactions more carefully; worth one
sentence in the paper rather than treating it as a discrepancy to
resolve.

## 7. Deployability (inference latency and model size)

Single-row inference: naive 0.001 ms, linear/ridge ~0.06 ms, MLP 0.26 ms
(13 KB), LightGBM 0.78 ms (944 KB), XGBoost 0.85 ms (967 KB), random
forest 63.9 ms median / 71.7 ms p95 (49 MB).

**This is a first-class result, not a footnote, and directly supports
PESA's Digital Intelligence / Thermal Management & System Integration
tracks' interest in deployability.** Random forest is statistically
indistinguishable in accuracy from XGBoost and LightGBM (66.71 vs. 65.18 /
65.37 kW RMSE -- well inside the seed-to-seed variation already reported)
while being roughly 75-100x slower per prediction and 50x larger on disk.
For a system meant to score every inverter reading in near-real-time on
edge or gateway hardware, that trade-off is decisive: the paper's explicit
recommendation should be XGBoost, LightGBM, or the small MLP over random
forest, despite random forest's marginally lower point-estimate RMSE.

## Suggested paper structure

1. **Title.** Something in the register of: "Weather-Conditioned
   Expected-Output Modeling for Inverter-Level Underperformance Detection
   in a Grid-Connected Solar PV Fleet."
2. **Abstract** (~180 words): problem, dataset, method, headline numbers
   (best RMSE, the two flagged inverters and their ratios, the AutoML tie,
   the cross-plant collapse), one-line contribution statement.
3. **Introduction.** Motivate why raw inverter output is not diagnostic on
   its own (cites IEC 61724 and the performance-ratio framing it
   standardizes); state contributions as a bullet list: (i) an explicit
   chronological-vs-random split comparison exposing a real accuracy gap
   most PV-ML papers on this dataset do not report; (ii) a
   statistically-grounded control-chart rule for flagging underperforming
   inverters instead of an arbitrary threshold; (iii) five rigor studies
   (cross-plant transfer, sensor robustness, uncertainty calibration,
   interpretability, deployability) evaluated together, which applied
   PV-ML papers rarely report as a set; (iv) a head-to-head comparison
   against a commercial AutoML product.
4. **Related work.** Antonanzas et al. (PV power forecasting review),
   Francisti et al. and Khandeparkar et al. (PV/inverter ML fault-detection
   papers), IEC 61724 (industry monitoring standard). Explicitly
   distinguish this work from the author's own open-set thermal-image PV
   fault classifier (different data modality and task -- see the README's
   "Relationship to other work" section) to preempt any overlap concern.
5. **Dataset.** Kaggle two-plant description, the DC/AC scale artifact,
   per-plant date-format and weather-sensor quirks.
6. **Methodology.** Chronological split rationale; feature set; all 8
   models; 3-seed repetition rationale; the control-chart underperformance
   rule; each rigor study's method (cross-plant, noise/missing injection,
   tree-quantile interval, SHAP, latency benchmark, AutoML setup).
7. **Results.** One subsection per section above (1-7), each anchored to
   its figure/table.
8. **Discussion.** Synthesize: model complexity buys little here once
   features are right; the two flagged inverters are actionable
   maintenance targets; generalization and calibration, not accuracy, are
   the open problems; a concrete deployment recommendation (favor
   XGBoost/LightGBM/MLP over random forest; a hand-built model matches
   AutoML here).
9. **Limitations.** Single-plant, 34-day window (no cross-season
   validation); cross-plant transfer needs capacity normalization before
   it could work; uncertainty intervals need conformal calibration before
   being deployment-trustworthy; the two flagged inverters are a
   statistical inference, not a confirmed maintenance-log fault --
   physical inspection would be needed to close the loop.
10. **Conclusion.**
11. **References** (see `references.md`).
