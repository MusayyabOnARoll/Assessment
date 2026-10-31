# Freight Rate Prediction Challenge

Predicts `posted_rate` for freight loads. `data/train_test.csv` (48,000 loads, Jan-Oct 2025) is used to build and
validate the model. The final model predicts the 12,000 loads in `data/validation.csv` (Nov-Dec 2025) and the 31 fixed
December 2025 chart loads.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Put the four assessment files in `data/` (they are not committed to this repo):

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

## Run

Run everything from the repository root. The code lives in `src/`, so set `PYTHONPATH` once per terminal session.

| Step | macOS / Linux | Windows PowerShell |
|---|---|---|
| Point Python at `src/` | `export PYTHONPATH=src` | `$env:PYTHONPATH="src"` |

```bash
# 1. Train the final model and write both prediction files (about 1-2 minutes)
python -m freight.train_predict

# 2. Validate the files and draw the December chart with the provided scorer
python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_chart_predictions.csv

# 3. Optional: reproduce the validation results (about 10 minutes)
python -m freight.validate
```

| Output | Created by | What it is |
|---|---|---|
| `outputs/validation_predictions.csv` | step 1 | **Submission file**: `load_id,predicted_rate`, 12,000 rows |
| `outputs/december_chart_predictions.csv` | step 1 | The 31 December rows with `predicted_rate` filled in |
| `scorer_results/candidate_december.png` | step 2 | December chart produced by `score.py` |
| `reports/*.csv`, `reports/*.json` | step 3 | Validation tables (per-fold results, unseen-city and random-split checks) |

## Approach

**Validation.** `validation.csv` is entirely in the future of the development data (Nov-Dec vs Jan-Oct) and contains
8 cities never seen in training, so a random split would overstate accuracy. The model is validated with
expanding-window, out-of-time folds (train on all earlier months, test on the next two: May-Jun, Jul-Aug, Sep-Oct),
plus an unseen-city check. A random 80/20 split is computed only as a reference. The final model is refit on all clean
development data.

**Cleaning.** 677 training rates (1.4%) are about 3.4x or 0.28x the normal level and cannot be predicted from any
feature (classifier AUC 0.51), so they are removed from training only (robust baseline by equipment and distance,
6 x MAD). Negative weights are sign errors (absolute value used), missing weights are kept with a flag, and the
47,500 lb cap is flagged. Missing `market_index` is replaced by that day's average across loads. City names are not
used as inputs, so cities not seen in training are handled through coordinates and distance.

**Model** (`src/freight/model.py`), predicting `log(posted_rate)`:
1. *Stage 1, load level:* a smooth linear baseline in log-distance plus a histogram gradient-boosting model on its
   residual. It sees no date or market inputs.
2. *Stage 2, market level:* a small ridge regression on the daily mean residual (computed out-of-fold) using the daily
   market index, daily mean `quote_signal`, day of week and a linear trend.

**December chart.** `december_chart_inputs.csv` has no market index or quote signal, so each December date uses that
day's average market index and average quote signal from `validation.csv` (features only, never labels).

## Results (MAPE %, clean rows, out-of-time folds)

| Model | May-Jun | Jul-Aug | Sep-Oct | Mean |
|---|---|---|---|---|
| Gradient boosting, load features + `quote_signal` | 2.89 | 3.39 | 1.50 | 2.59 |
| + raw `market_index` | 2.00 | 2.88 | 3.57 | 2.82 |
| **Final two-stage model** | **1.96** | **1.85** | **2.03** | **1.95** |

Unseen-city check: 1.64-1.79% (mean 1.70%) versus 1.56% for a random split. See the full table in the report.

## Layout

```
data/                          input CSVs (not committed)
src/freight/features.py        cleaning, outlier flagging, feature engineering, daily market table
src/freight/model.py           two-stage model
src/freight/data.py            data loading
src/freight/metrics.py         MAE / RMSE / MAPE / MedAPE
src/freight/validate.py        out-of-time validation, ablations, unseen-city and random-split checks
src/freight/train_predict.py   final fit, validation and December predictions
score.py                       provided scorer (unchanged)
```

## Limitations

- Model variants were compared on the same three folds that are reported, so scores are slightly optimistic.
- Predictions target the typical (median) rate. About 1.4% of training labels are unpredictable spikes or drops; if the
  hidden validation labels contain them, they will dominate squared-error metrics for any model.
- No holiday information is modelled and the development data stops in October.
- Predictions are reproducible in a fixed environment; across library versions they differ by about 0.3% on average.