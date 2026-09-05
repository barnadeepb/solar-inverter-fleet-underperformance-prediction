# Solar Inverter Underperformance Detection

Detecting individual underperforming inverters in a grid-connected solar PV
plant by comparing each inverter's real output against a weather-conditioned
expected-output model, using real 15-minute telemetry from two Indian solar
plants.

## Problem

A solar inverter's raw AC power output tells you very little on its own --
low output on a cloudy day is normal, and high output on a sunny day is
normal. What actually indicates a fault (soiling, partial shading, a
degraded string, a wiring or connection problem) is output that is
persistently low *relative to what the weather at that moment predicts it
should be*. This project builds that reference model and uses it to flag
which inverters in a 22-inverter fleet are consistently underperforming.

## Dataset

Kaggle's `anikannal/solar-power-generation-data`: 15-minute generation logs
(DC/AC power, daily/total yield) and co-located weather sensor readings
(ambient temperature, module temperature, irradiation) for two real solar
plants in India, May 15 - June 17, 2020 (34 days). Plant 1 has 22 inverters;
Plant 2 has 22 as well but different hardware/site conditions, used here
only for the cross-plant generalization check. Raw CSVs are committed under
`data/raw/`; `src/fetch_data.py` re-downloads them from a GitHub mirror.

Known dataset quirks handled in `src/data_prep.py`:
- Plant 1 and Plant 2's generation files use different date formats
  (DD-MM-YYYY vs. YYYY-MM-DD), detected automatically rather than assumed.
- Weather is recorded once per plant (one sensor), not per inverter, so it
  is joined on `(DATE_TIME, PLANT_ID)`.
- Plant 1's `DC_POWER` is on a documented ~10x different scale than
  `AC_POWER`; this project models `AC_POWER`, the physically delivered
  quantity, and never mixes `DC_POWER` across plants.

## Method

1. **Expected-output model.** Train regressors to predict `AC_POWER` from
   weather (irradiation, ambient/module temperature, time-of-day) alone --
   deliberately excluding any inverter's own power history, so the
   "expected" value for a faulty inverter isn't contaminated by its own
   fault.
2. **Underperformance detection.** Use the trained model as a per-timestep
   expected value for every real inverter reading. Compute each inverter's
   performance ratio (actual / expected), aggregate to a daily value, and
   flag inverters whose daily ratio falls below a fleet-wide control-chart
   band (median - 3 x MAD) on at least half the study period -- a
   statistical rule rather than an arbitrary fixed threshold.

**Evaluation split.** The primary split is *chronological* (first 24 days
train, remaining ~10 days test), not a random shuffle -- this tests genuine
forecasting into an unseen future period rather than interpolation between
neighboring readings of the same day, which a random split would silently
allow. Night-time readings (`IRRADIATION == 0`) are excluded from all
accuracy metrics, since a model that always predicts ~0 at night would
otherwise dominate the error statistics with a trivially solved case.

**Repetition.** Every stochastic model (random forest, XGBoost, LightGBM,
MLP, CNN) is refit under 3 random seeds; reported metrics are mean +/- std.
Deterministic models (naive baseline, linear regression, ridge) have zero
variance by construction.

## Results

### AC power prediction (Plant 1, chronological split, daylight only)

| Model | Test RMSE (kW) | Test MAE (kW) | Test R2 |
|---|---|---|---|
| Naive mean baseline | 364.67 | 313.69 | -0.015 |
| CNN (1D, lag-window) | 69.16 +/- 1.01 | 36.20 | 0.961 |
| Random Forest | 66.71 +/- 0.02 | 32.41 | 0.966 |
| Ridge | 66.38 | 34.89 | 0.966 |
| Linear Regression | 66.34 | 34.80 | 0.966 |
| LightGBM | 65.37 +/- 0.00 | 30.71 | 0.967 |
| XGBoost | 65.18 +/- 0.00 | 30.84 | 0.968 |
| MLP | 63.73 +/- 0.19 | 30.08 | 0.969 |
| Vertex AI AutoML Tables | 63.80 | 29.62 | 0.969 |

All models clear the naive baseline by a wide margin; differences between
linear, tree-ensemble, and neural approaches are modest once the split is
chronological rather than random shuffle (compare against random-split
numbers, which run closer to RMSE ~46 kW -- the chronological split is the
harder, more honest test, since a random split lets nearby-in-time daylight
readings leak information across the train/test boundary). Notably, a
managed AutoML product (Vertex AI AutoML Tables) does not beat a
lightweight, hand-built MLP on this problem -- it matches it almost
exactly, while costing over an hour of training time and real compute
spend versus seconds on a laptop CPU.

See `results/figures/model_comparison.png`.

### Underperformance detection (Plant 1, all 34 days, 22 inverters)

Two inverters are flagged as consistently underperforming (daily
performance ratio below the fleet control band on >=50% of days):

| Inverter | Mean performance ratio | Days below band |
|---|---|---|
| `bvBOhCH3iADSZry` | 93.2% | 32 / 34 |
| `1BY6WEcLGh8j5v7` | 94.7% | 30 / 34 |

The remaining 20 inverters cluster between 100-111% with at most 4/34 days
below the band. This is a persistent pattern across the entire study
period, not a one-off dip -- consistent with a physical fault (soiling,
partial shading, or a wiring/connection issue) rather than measurement
noise. See `results/figures/inverter_performance_ranking.png` and
`results/metrics/inverter_performance_ratios.csv`.

## Rigor checks beyond the accuracy benchmark

- **Cross-plant generalization** (`src/cross_plant_eval.py`): training on
  Plant 1 and testing zero-shot on Plant 2 drops R2 from ~0.97 to ~0.25 --
  a real domain-shift finding, showing the model has learned a
  site-specific mapping rather than a universally transferable one.
  (`results/metrics/cross_plant_generalization.json`)
- **Robustness to sensor noise / dropout** (`src/robustness_eval.py`): test
  RMSE degrades gracefully under injected Gaussian sensor noise (66.7 -> 
  127.3 kW at 30% noise) and more sharply under mean-imputed missing
  readings (66.7 -> 170.5 kW at 20% missing).
  (`results/metrics/robustness.json`)
- **Uncertainty calibration** (`src/uncertainty_eval.py`): a naive 90%
  prediction interval built from random-forest tree-prediction spread
  achieves only ~56% empirical coverage under the chronological test split
  -- i.e. it is meaningfully overconfident, and should not be trusted at
  face value for an alarm threshold without proper conformal calibration.
  Reported as an honest limitation, not smoothed over.
  (`results/metrics/uncertainty_calibration.json`)
- **Interpretability** (`src/interpretability_eval.py`): SHAP values and
  built-in feature importance both attribute ~94-99% of the model's
  predictive weight to irradiation, matching known solar-generation
  physics rather than a spurious correlation (e.g. time-of-day).
  (`results/metrics/interpretability.json`,
  `results/figures/shap_feature_importance.png`)
- **Deployability** (`src/latency_benchmark.py`): random forest is barely
  more accurate than XGBoost/LightGBM but ~100x slower per single-row
  prediction (64ms vs <1ms) and ~50x larger on disk (49 MB vs <1 MB) --
  a real trade-off for anyone deploying this on edge/gateway hardware.
  (`results/metrics/latency_benchmark.json`)

## Vertex AI AutoML benchmark

As a "what does a managed AutoML product get you on the same split"
comparison point, `src/vertex_automl_launch.py` trains a Vertex AI AutoML
Tables regressor on the identical train/test split, and
`src/vertex_automl_evaluate.py` scores it via batch prediction on the
held-out test set using the same RMSE/MAE/R2 metrics as every other model.

**Status:** complete. Training took roughly an hour; results are in
`results/metrics/regression_benchmark.json` under `vertex_automl_tables`
and in the table above.

## Repository layout

```
data/raw/                 raw plant generation and weather CSVs
src/
  fetch_data.py           re-download raw data
  data_prep.py            loading, merging, feature engineering, splits
  models_classical.py     naive/linear/random forest/XGBoost/LightGBM
  models_neural.py        MLP and 1D-CNN (PyTorch)
  train.py                main regression benchmark
  anomaly_detection.py    expected-output model -> inverter performance ratios
  cross_plant_eval.py     Plant 1 -> Plant 2 generalization check
  robustness_eval.py      sensor noise / missing-data robustness
  uncertainty_eval.py     prediction-interval calibration
  interpretability_eval.py SHAP feature-importance check
  latency_benchmark.py    single-sample inference latency and model size
  vertex_automl_launch.py launches the Vertex AI AutoML Tables job
  vertex_automl_evaluate.py scores the completed AutoML job
  make_figures.py         regenerates the two headline figures
results/
  metrics/                all metric JSON/CSV outputs
  figures/                all generated figures
paper/                    full-paper draft (in progress)
```

## Reproducing

```
pip install -r requirements.txt
cd src
python fetch_data.py            # optional -- data/raw/ is already committed
python train.py                 # regression benchmark
python anomaly_detection.py     # underperformance detection
python cross_plant_eval.py
python robustness_eval.py
python uncertainty_eval.py
python interpretability_eval.py
python latency_benchmark.py
python make_figures.py
```

The Vertex AI scripts additionally require a GCP project with the Vertex AI
API enabled and `gcloud auth login` completed; they are billable and not
required to reproduce the core (non-AutoML) results above.

## Relationship to other work

A related but distinct project,
[`open-set-solar-fault-detection`](https://github.com/barnadeepb/open-set-solar-fault-detection),
addresses a different problem in the same domain: open-set recognition on
*thermal infrared images* of PV modules, classifying known fault types and
testing whether the classifier honestly flags a fault type it has never
seen. This project instead works from *numerical production telemetry*
(irradiation, temperature, power) and frames the problem as regression-based
operational monitoring -- detecting *which inverters* in a live fleet are
underperforming relative to weather-conditioned expectation, not
classifying *what visual fault type* a module has. Different data
modality, different task, complementary rather than overlapping angles on
solar PV predictive maintenance.

## License

MIT -- see `LICENSE`.
