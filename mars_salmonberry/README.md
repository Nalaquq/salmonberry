# MARS modeling pipeline: salmonberry occurrence, LiDAR slope vs. moundness

A configuration-driven R pipeline that compares MARS (Multivariate Adaptive
Regression Splines) classifiers of salmonberry (*Rubus spectabilis*) presence
and absence using spatial cross-validation, then evaluates the selected model
once on an untouched test set.

> **The data do not exist yet.** This folder contains only code,
> configuration, and documentation. The pipeline never creates, simulates, or
> downloads data. It stops with an explanatory message until you point
> `config.yaml` at real CSV files exported from ArcGIS Pro.

| File | Purpose |
|---|---|
| `mars_salmonberry.R` | The pipeline. Modular functions, with detailed comments on the statistics. |
| `config.yaml` | Template configuration: paths, column names, models, and settings. |
| `DATA_SCHEMA.md` | Required structure of the future training and test CSV files. |
| `README.md` | This document. |

---

## 1. Purpose

The pipeline quantifies how well LiDAR-derived terrain metrics, with and
without PlanetScope spectral information, predict salmonberry occurrence at
locations the model has not seen. It is built to produce defensible,
reproducible out-of-sample estimates for a scientific publication.

## 2. Scientific hypothesis

> A MARS model using LiDAR-derived **slope** has greater out-of-sample
> predictive performance for salmonberry occurrence than a MARS model using
> LiDAR-derived **moundness**.

| Model | Formula |
|---|---|
| M1 | `SALMONBERRY ~ MOUNDNESS` |
| M2 | `SALMONBERRY ~ SLOPE` |
| M3 | `SALMONBERRY ~ MOUNDNESS + BLUE` |
| M4 | `SALMONBERRY ~ SLOPE + BLUE` |
| M5 | `SALMONBERRY ~ MOUNDNESS + SLOPE + BLUE` |

- **Primary contrast:** ΔAUC = AUC(M2) − AUC(M1), per spatial fold.
- **Secondary contrast:** ΔAUC = AUC(M4) − AUC(M3), per spatial fold.
- **Spectral contribution:** compare M3 with M1, M4 with M2, and M5 with the
  others (add further `comparisons` in `config.yaml` if you want these as
  formal paired contrasts).

## 3. Expected future input data

There are two CSV files, one for training and one for testing. Each row is one
labeled point. See **`DATA_SCHEMA.md`** for the full specification. In
summary:

```text
ID, X, Y, SALMONBERRY (1/0), SLOPE, MOUNDNESS, BLUE, BLOCK_ID, FOLD_ID
```

The test file needs the same response and predictor columns. It does not need
`FOLD_ID`.

The intended upstream workflow in ArcGIS Pro:

```text
LiDAR DTM ──► Horn's slope ─────────► Slope raster     ┐
LiDAR DTM ──► Focal statistics ─────► Moundness raster ├─► Extract values at
PlanetScope ─► Blue band ───────────► Blue raster      ┘   labeled points
                                                             │
Salmonberry field / labeled observations ────────────────────┘
      ▼
Attribute table ─► Spatial blocks ─► Spatial folds ─► Train/test split ─► CSV
      ▼
THIS R PIPELINE
```

The R script does not compute slope or moundness. It works only on the
exported table.

## 4. How the YAML configuration works

`mars_salmonberry.R` contains **no file paths and no column names**.
Everything comes from `config.yaml`:

| Section | What it controls |
|---|---|
| `data` | `train_path`, `test_path`, and text values treated as missing. |
| `variables` | Column names for ID, response, coordinates, block, fold, and a **named map of predictors** (`slope: "SLOPE"`, …). |
| `response_coding` | Codes for presence (1) and absence (0). |
| `models` | The competing models, each a list of predictor keys. |
| `comparisons` | Paired ΔAUC contrasts (`model_a − model_b`). |
| `model` | Seed, classification threshold, number of folds, MARS settings (`degree`, `nprune`, `penalty`, …), and tuning method. |
| `final_model` | Which model goes to the test set: pre-registered, or the one with the highest mean CV AUC. |
| `evaluation` | Whether to read and score the test set. |
| `validation` | Thresholds for data checks (minimum fold size, and more). |
| `output`, `plots`, `variable_importance` | Output location and figure settings. |

**Relative paths are resolved against the folder containing the YAML
file**, not the working directory. The template uses `path/to/...`
placeholders. The script detects these and refuses to run until you replace
them.

You can keep several configurations, for example `config_degree2.yaml`, and
choose one with `--config`.

## 5. Installing packages

R ≥ 4.1 is required. Install the packages once:

```r
install.packages(c("earth", "caret", "yaml", "dplyr", "readr",
                   "pROC", "ggplot2", "tidyr"))
```

| Package | Role in the pipeline |
|---|---|
| `earth` | MARS implementation: forward pass, GCV pruning, binomial GLM step, and variable importance (`evimp`). |
| `caret` | Optional nested **spatial** tuning of `degree` and `nprune` (`tuning$method: "spatial_cv"`), using fold indices built from `FOLD_ID`. |
| `yaml` | Reads the configuration and records the configuration used. |
| `dplyr` | Grouping, summarising, and joining results. |
| `readr` | Reads the input CSVs and writes the output CSVs. |
| `pROC` | ROC-AUC from predicted probabilities, and the DeLong CI for the test AUC. |
| `ggplot2` | Figures. |
| `tidyr` | Reshaping, e.g. wide AUC tables for paired ΔAUC. |

The script checks for these packages at start-up and prints the install
command if any are missing. It never installs packages itself.

## 6. Preparing the future training data

1. Label each observation point as `1` (present) or `0` (absent).
2. Use *Extract Multi Values to Points* (or an equivalent tool) to attach
   `SLOPE`, `MOUNDNESS`, and `BLUE` to each point.
3. Remove or fix points that fall on NoData cells.
4. Create spatial blocks and assign each block to a fold (Section 7).
5. Split the points into training and test sets, preferably **by block** so
   that test points are spatially separate from training points.
6. Export both attribute tables to CSV (see the checklist in
   `DATA_SCHEMA.md`).
7. Edit `config.yaml` so that `data$train_path` and `data$test_path` point to
   the files.
8. Run `Rscript mars_salmonberry.R --config config.yaml --validate-only` and
   fix anything it reports **before** fitting models.

## 7. Generating `BLOCK_ID` and `FOLD_ID`

- **Blocks.** Cover the study area with contiguous blocks, for example a
  square grid made with *Generate Tessellation* or *Create Fishnet*. Give each
  point the ID of the block that contains it (*Spatial Join*). The block size
  should ideally be at least the range of spatial autocorrelation in the
  predictors or the model residuals. A variogram of slope, moundness, or blue
  is a practical way to estimate that range. Larger blocks give more honest
  but more variable estimates.
- **Folds.** Assign **whole blocks** to K folds (default 5). Every point in a
  block receives that block's `FOLD_ID`. Balance the folds roughly for number
  of points and for presences and absences: every fold must contain both
  classes, or AUC is undefined. Record how the assignment was made (random,
  stratified, or checkerboard) and the random seed used in ArcGIS, for the
  Methods section.
- **The pipeline checks, but never changes, this structure.** If a block
  appears in two folds, the run stops:

```text
ERROR:
Spatial-fold integrity error.
  BLOCK_ID 17 occurs in multiple FOLD_ID values (1, 4).
This violates the spatial cross-validation requirement: ...
```

## 8. How spatial cross-validation works, and why it is needed

Observations are points in a landscape. Nearby points tend to share elevation,
slope, moundness, vegetation, soils, moisture, and spectral reflectance, and
salmonberry presence is itself spatially clustered. If individual points were
randomly split into training and validation sets, most validation points
would have a near-identical neighbour in the training set. The model would be
rewarded for recognising local conditions it has effectively already seen.
This is **spatial leakage**. It inflates accuracy and can bias comparisons
between predictors (Roberts et al. 2017; Valavi et al. 2019).

```text
Study area
     ↓
Spatial blocks
     ↓
Assign blocks to folds
     ↓
Each block belongs to exactly one fold
     ↓
Spatial cross-validation
```

For fold *k* = 1…K, each model is trained on all folds except *k* and
predicts fold *k*:

```text
Fold 1: train = folds 2–5, validate = fold 1
Fold 2: train = folds 1,3,4,5, validate = fold 2
...
Fold 5: train = folds 1–4, validate = fold 5
```

**All five models use exactly the same folds.** Each ΔAUC is therefore a
paired comparison on the same held-out areas. Nothing is re-randomised
between models.

With `tuning$method: "spatial_cv"`, MARS complexity is also tuned by
**nested** spatial CV. Within each outer training set, caret holds out each
remaining fold in turn. The outer validation fold is never used for tuning.

## 9. How the MARS models work

MARS builds a flexible function from **hinge functions** at knots *t*:

```text
h(x − t) = max(0, x − t)          h(t − x) = max(0, t − x)
```

A pair of hinges at a knot lets the slope of the relationship change at *t*.
Several knots approximate nonlinear, threshold, or plateau responses without
fixing their shape in advance. The model is

```text
f(X) = β0 + Σ βm Bm(X)
```

where β0 is the intercept, βm are coefficients, and each Bm is a hinge or a
product of hinges on different predictors. A product represents an
**interaction**; the maximum number of hinges in a product is `degree`.

- **Forward pass:** greedily adds the pair of hinges (predictor, knot, and
  parent term) that most reduces the residual sum of squares, until `nk`
  terms are reached or the improvement falls below `thresh`.
- **Backward pass:** removes terms one at a time and keeps the subset that
  minimises generalized cross-validation,
  GCV = (RSS/N) / (1 − C(M)/N)², with C(M) = M + d·(M − 1)/2. The penalty
  *d* (default 2, or 3 with interactions) charges each knot for having been
  estimated from the data.
- **Binary response:** the response is 0/1, so this is **classification**.
  With `glm = list(family = binomial)`, earth selects the basis functions as
  above and then fits a **logistic GLM** on them:
  logit P(present) = β0 + Σ βm Bm(X). Predictions are probabilities of
  occurrence. Hinges are piecewise linear on the logit scale, so the
  probability curves are smooth S-shapes that bend near the knots.

The comments in Part B of `mars_salmonberry.R` give the full derivation. They
are written to support the statistical Methods section.

**Suggested Methods wording (adapt as needed):**
*"Salmonberry presence/absence was modelled with multivariate adaptive
regression splines (MARS; Friedman 1991) using the R package earth. Basis
functions and knots were selected with the MARS forward-selection and
backward-pruning algorithm, using generalized cross-validation (penalty = 2
per knot, additive models [degree = 1]). The coefficients of the retained
basis functions were then estimated by maximum likelihood in a binomial GLM
with a logit link. Predictive performance was estimated by K = 5-fold blocked
spatial cross-validation, with folds defined by spatial blocks so that all
observations in a block were held out together, and identical folds for all
candidate models."*

## 10. How the five models are compared

For each model and fold the pipeline computes ROC-AUC, sensitivity,
specificity, precision, F1, balanced accuracy, and accuracy
(`cv_metrics.csv`). It then summarises them across folds
(`model_comparison.csv`: mean, SD, min, and max AUC, and so on, plus the
pooled out-of-fold AUC).

For each configured contrast it computes the **per-fold paired ΔAUC**
(`delta_auc.csv`) and a summary (`delta_auc_summary.csv`): mean, SD, minimum,
maximum, and the number of folds in which each model was higher.

**No model is declared "better" because its mean AUC is numerically
larger.** Differences are reported objectively. Limitations of inference from
CV folds:

- Fold AUCs are **not independent**. With K = 5, any two training sets share
  75% of their data, so the SD across folds understates the true uncertainty
  (Bengio & Grandvalet 2004).
- K is small (5), so a t-test on five paired differences has very little
  power, and its assumptions are violated for the reason above. The pipeline
  therefore does **not** report p-values.
- Spatial folds differ in environment and prevalence. Some folds may be
  intrinsically harder, which is why the comparison is paired.
- Evidence for the hypothesis is stronger if ΔAUC is consistently positive
  across folds, the mean difference is large relative to fold-to-fold
  variation, and the result holds on the test set and under alternative block
  sizes or fold assignments. You can repeat the ArcGIS blocking with a
  different seed or block size and re-run the pipeline with a second config.
  If a formal test is needed, consider the corrected resampled t-test
  (Nadeau & Bengio 2003) and state its assumptions, or a DeLong test on the
  single test set.

**Model selection for the test set** uses only training-data CV. Either
pre-register a model in `final_model$selected_model` (recommended for
hypothesis testing), or let the pipeline choose the highest mean CV AUC, with
ties broken by fewer predictors.

## 11. How the test set is protected

```text
Training data → Spatial CV → Model comparison → Model selection
      → Retrain selected model on ALL training data
      → Untouched test data → Final evaluation
```

- The test CSV is read in **one place only**, `evaluate_test_data()`. This
  runs after every modeling decision has been made and saved.
- The test data are never used to select predictors, tune MARS, choose knots
  or complexity, set the threshold, or choose between models.
- The script stops if any `ID` occurs in both files, and warns if a
  `BLOCK_ID` is shared.
- Cross-validation outputs are written to disk **before** the test file is
  read.
- **Two-stage option:** set `evaluation$run_test_evaluation: false` to run
  CV without the test file even existing. Fix `selected_model` and the
  threshold, then set it to `true` and run once.
- By default only the selected model is scored on the test set. Setting
  `evaluate_all_models_on_test: true` reports all of them, labeled
  `secondary_descriptive_only`. These results must not be used to change the
  model choice.

## 12. Interpreting the metrics

With presence as the positive class:

| Metric | Formula | Meaning |
|---|---|---|
| **ROC-AUC** | area under the ROC curve | Probability that a randomly chosen presence gets a higher predicted probability than a randomly chosen absence. 0.5 means no discrimination and 1.0 means perfect. It is computed from **probabilities** and needs no threshold. It measures ranking, not calibration. |
| **Sensitivity** (recall) | TP / (TP + FN) | Share of real presences predicted as present. |
| **Specificity** | TN / (TN + FP) | Share of real absences predicted as absent. |
| **Precision** | TP / (TP + FP) | Share of predicted presences that are real. It is undefined (NA) if the model predicts no presences in a fold, and it depends strongly on prevalence. |
| **F1** | 2TP / (2TP + FP + FN) | Harmonic mean of precision and sensitivity. It ignores true absences. |
| **Balanced accuracy** | (sensitivity + specificity) / 2 | Accuracy that is not inflated by class imbalance. |
| **Accuracy** | (TP + TN) / N | Can look high when one class dominates, so interpret it together with balanced accuracy. |

All metrics except AUC depend on the **classification threshold**
(`model$classification_threshold`, default 0.5; predicted present if
probability ≥ threshold). With imbalanced data, 0.5 may give low sensitivity
even when AUC is good. If you change the threshold, justify the choice
**before** looking at the test set, for example from prevalence or from
training out-of-fold predictions. Never tune it on the test set.

**Variable importance** (`variable_importance.csv`; `earth::evimp`) measures
how much each predictor contributes to the GCV / RSS improvement across the
pruning sequence, scaled to 0–100. It reflects **predictive association in
this fitted model, not causation**. Slope and moundness are both derived from
the same DTM and may be correlated, so they can share or swap importance. In
one-predictor models the importance is trivially 100.

**Response-curve figures** show fitted probability against slope or
moundness, with other predictors held at their training medians, and mark the
MARS knots. They describe the model's fitted association and must not be read
as causal effects.

## 13. Where outputs are saved

The output folder is `output/` next to `config.yaml` by default
(`output$directory`):

```text
output/
├── models/               final_<model>.rds, cv_fold_models.rds,
│                         final_model_terms.csv, final_model_summaries.txt
├── predictions/          fold_predictions.csv, test_predictions.csv
├── metrics/              cv_metrics.csv, model_comparison.csv,
│                         delta_auc.csv, delta_auc_summary.csv, test_metrics.csv
├── variable_importance/  variable_importance.csv
├── plots/                cv_auc_by_model, delta_auc_by_fold,
│                         response_curve_slope, response_curve_moundness (.png/.pdf)
├── diagnostics/          data_validation_report.txt, fold_summary.csv,
│                         cv_fit_diagnostics.csv (terms, GCV, GLM convergence,
│                         warnings), model_selection.txt, tuning tables
└── session_info/         config_original.yaml, config_used.yaml,
                          run_manifest.yaml (input MD5s, versions), sessionInfo.txt
```

Prediction files contain `predicted_probability` and `predicted_class`, the
observed class, the ID, and the coordinates and block when available.

Reload a fitted model with `fit <- readRDS("output/models/final_M2_slope.rds")`
and predict with `predict(fit, newdata, type = "response")`.

## 14. Reproducing a run

```bash
# 1. Check data and folds only (no fitting)
Rscript mars_salmonberry.R --config config.yaml --validate-only

# 2. Full run
Rscript mars_salmonberry.R --config config.yaml
```

From R:

```r
source("mars_salmonberry.R")        # defines functions; does not run
results <- run_pipeline("config.yaml")
```

- MARS fitting in earth is deterministic. The seed (`model$seed`) is set
  before every fit so that any stochastic step (caret) is reproducible.
- `session_info/` records the exact configuration, MD5 checksums of the input
  CSVs, package versions, and `sessionInfo()`. To reproduce a run, use the
  same CSVs (the checksums should match), the saved `config_used.yaml`, and
  the same package versions.
- Writing to a non-empty output folder overwrites same-named files, and the
  script warns first. Use a different `output$directory` per run to keep
  results separate.

## References

- Friedman, J. H. (1991). Multivariate adaptive regression splines. *Annals of Statistics* 19:1–67.
- Milborrow, S. earth: Multivariate Adaptive Regression Splines. R package. https://CRAN.R-project.org/package=earth
- Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40:913–929.
- Valavi, R. et al. (2019). blockCV: an R package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models. *Methods in Ecology and Evolution* 10:225–232.
- Bengio, Y. & Grandvalet, Y. (2004). No unbiased estimator of the variance of K-fold cross-validation. *JMLR* 5:1089–1105.
- Nadeau, C. & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning* 52:239–281.
- Kuhn, M. (2008). Building predictive models in R using the caret package. *Journal of Statistical Software* 28(5).
- Robin, X. et al. (2011). pROC: an open-source package for R and S+ to analyze and compare ROC curves. *BMC Bioinformatics* 12:77.
