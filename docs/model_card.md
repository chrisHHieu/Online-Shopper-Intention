# Model card: online shopper intention

## Overview

| Item | Value |
|---|---|
| Model name | `shopper_intent` LightGBM, version `20261031T144050Z` |
| Model type | Gradient boosted trees (`LGBMClassifier`) with isotonic calibration |
| Baseline | Logistic regression with standardised features and balanced class weights |
| Task | Binary classification: does this session end in a purchase |
| Output | Calibrated probability of conversion, contact or skip decision, expected net value, top-3 additive contributions |
| Owner | Hoang Hieu Nguyen |
| Licence | MIT for the code, CC BY 4.0 for the dataset |

## Intended use

Prioritising which live shopping sessions are worth an outreach action, and understanding which observed session features move the model's estimate. The output is a probability from a model trained on 2016 session logs from one retailer, not a forecast of an individual customer's behaviour.

Out of scope: credit, pricing, eligibility or any decision with legal or safety consequences; any use where a false negative is materially worse than a false positive without re-tuning the value and cost inputs; individual-level profiling across sites.

## Data

Source: UCI Machine Learning Repository, dataset 468, Online Shoppers Purchasing Intention. 12,330 sessions, 17 features and the `Revenue` label, positive rate 0.1547, SHA256 `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`. The dataset is downloaded at runtime; no copy is stored in the repository.

Features: counts and durations of administrative, informational and product-related pages, bounce and exit rates, page values, special-day proximity, month, operating system, browser, region, traffic type, visitor type and weekend flag. Derived features are page value per product page, duration per page viewed, and the bounce-exit spread. The one-hot vocabulary for month, visitor type and weekend is fixed from the configuration, and nothing is computed from the label.

The dataset has no customer identifiers and no timestamps, so a random stratified split is used. Duplicate rows (125) are kept as published.

## Training protocol

1. Stratified 80/20 split with seed 42: 9,864 training sessions, 2,466 held-out sessions.
2. Stratified 5-fold cross-validation inside the training split. Out-of-fold probabilities produce the headline metrics and the calibration fit.
3. Isotonic regression fitted on out-of-fold probabilities. Logistic regression and LightGBM are both calibrated so the Brier comparison is like for like.
4. The decision threshold maximises expected net value on the same out-of-fold probabilities. It never sees the held-out split.
5. Both models are refitted on the whole training split and registered as `models/<type>/<version>/`.
6. The held-out split is scored once by `evaluate`; its numbers are in the report and in this card.

LightGBM settings after the Optuna search (`configs/tuned.yaml`): 250 trees, learning rate 0.018, 10 leaves, minimum 46 samples per leaf, row subsampling 0.755, column subsampling 0.636, L2 penalty 5.13, `is_unbalance` for class imbalance, one thread with `deterministic` enabled. The search ran on the training split only, TPE sampler with a MedianPruner, and every trial was scored by out-of-fold net value, not accuracy. Logistic regression: L2 with C=0.315, `class_weight=balanced`, standardised inputs.

## Metrics

Cross-validated, out-of-fold, 9,864 sessions:

| Model | PR-AUC | ROC-AUC | Brier | Precision | Recall | F1 | Net value per 1000 |
|---|---|---|---|---|---|---|---|
| Logistic regression | 0.6614 | 0.9060 | 0.0780 | 0.2061 | 0.9948 | 0.3415 | 14,642.64 |
| LightGBM (champion) | 0.7332 | 0.9359 | 0.0676 | 0.2592 | 0.9980 | 0.4115 | 14,844.28 |
| XGBoost | 0.7325 | 0.9290 | 0.0694 | 0.2472 | 0.9980 | 0.3962 | 14,815.29 |
| CatBoost | 0.7351 | 0.9350 | 0.0671 | 0.2425 | 0.9987 | 0.3903 | 14,813.06 |

Held-out, 2,466 sessions, LightGBM at threshold 0.0192:

| Metric | Value |
|---|---|
| Precision | 0.2513 |
| Recall | 1.0000 |
| F1 | 0.4017 |
| PR-AUC | 0.7196 |
| ROC-AUC | 0.9303 |
| Brier | 0.0710 |
| ECE | 0.0084 |
| Confusion matrix | TN 946, FP 1,138, FN 0, TP 382 |
| Sessions contacted | 1,520 of 2,466 (61.6 percent) |
| Net value | 36,680.00, or 14,874.29 per 1000 sessions |
| Conformal band, nominal 90 percent | coverage 0.8986, mean set size 0.997 labels |

Policy comparison at 100 value per conversion and 1 cost per outreach: always-contact gives 14,490.67 per 1000, the 0.5 default gives 9,871.45, the chosen threshold gives 14,874.29. The gain over always-contact is 383.62 per 1000, or 2.6 percent.

## Threshold and calibration

The break-even probability at these economics is cost divided by value, 0.01. Because the base rate is 0.155, contacting every session already has positive expected value, which is why the value-optimal threshold sits just above the break-even point and why the margin over blanket outreach is small. Raising `business.cost_per_outreach` moves the threshold up and makes the ranking more valuable; the trade-off is visible in `reports/figures/threshold_value.png`.

The cost sweep in `reports/cost_sensitivity.csv` measures this on the held-out split, with each threshold chosen on the training out-of-fold probabilities. The gain over always-contact is 383.62 per 1000 sessions at cost 1, 2,293.19 at cost 5, 5,304.14 at cost 10, 16,311.84 at cost 25 and 37,246.55 at cost 50. Always-contact breaks even at 15.49 per outreach and loses money above it, while the tuned policy stays positive across the sweep.

Isotonic calibration is chosen per candidate by out-of-fold Brier score and produced the held-out figures above: 0.0710 for LightGBM, 0.0707 for CatBoost, 0.0739 for XGBoost and 0.0823 for the logistic baseline. The reliability curve in `reports/figures/calibration.png` shows raw tree scores are under-confident at low probabilities and over-confident above roughly 0.2; calibrated scores track the diagonal.

## Explainability

Global: permutation importance on the held-out split, measured as the drop in ROC-AUC when one column is shuffled, averaged over five repeats. `PageValues` dominates (0.1063), followed by `PageValuePerProductPage` (0.0442) and month indicators (0.0222 for May, 0.0201 for November).

Local: the API returns the top three contributions per prediction. For LightGBM these are exact tree SHAP values from `pred_contrib=True`, which sum to the model output plus the base value; for logistic regression they are coefficient times scaled feature value, which sum to the logit. Both are additive attributions of the model, not causal effects.

## Ethical and operational notes

The dataset contains no personal data, but the model processes session behaviour. It should not be used to infer sensitive attributes. Browser, operating system, region and traffic type are coarse codes; the model may pick up artefacts of the logging setup rather than user intent.

Monitoring: run `python -m shopper_intent drift` against a new batch. PSI below 0.10 is stable, 0.10 to 0.25 moderate, 0.25 and above significant. Input drift alone does not prove performance decay; conversion outcomes should be joined back to scored sessions on a delay.

## Limitations

- One retailer, one collection period (2016), session level, no customer history and no timestamps.
- The held-out split is a single random split of 2,466 sessions; differences of a few points between runs are within noise.
- The value and cost inputs are assumptions. Net value is not measured incremental revenue, and no experiment was run.
- Isotonic calibration is fitted on out-of-fold predictions from 4/5 models and applied to the full-data refit.
- The model is not tuned for accuracy. At the business threshold accuracy is 0.51, which reflects the cost ratio, not a defect in the ranking.
- Exact duplicate rows may land in both splits (about 1 percent of rows), slightly favouring test scores.
- The original notebook's 0.926 accuracy is not a reference point: it used SMOTE before the split and preprocessing fitted on all rows.

## Maintenance

Retrain with `python -m shopper_intent train` after a config or data change; every run writes a new version and moves the `latest` pointer, and `metadata.json` records config hash, data SHA256, git commit, seed and metrics. Roll back by writing an older version string into `models/<type>/latest`. Re-run `evaluate` after retraining so the report and figures match the served artifact.
