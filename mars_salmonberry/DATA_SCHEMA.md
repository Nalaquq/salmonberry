# Input data schema (future CSV files)

The pipeline reads two CSV files that **do not exist yet**. You will create them
in ArcGIS Pro after labeling salmonberry observations. This document describes
what they must contain. No example data files are provided, and the pipeline
never generates any.

Column names below are the defaults. Every name can be changed in the
`variables` section of `config.yaml`. Names are case-sensitive.

## Training file (`data$train_path`)

One row per labeled observation (point) used for model development.

| Column | Required | Type | Allowed values / meaning |
|---|---|---|---|
| `ID` | yes | text or integer | Unique observation identifier. No duplicates. Must not appear in the test file. |
| `X` | recommended | numeric | Easting / longitude in the project CRS. Copied into the prediction files. **Not used as a predictor.** |
| `Y` | recommended | numeric | Northing / latitude. As for `X`. |
| `SALMONBERRY` | yes | integer | `1` = present, `0` = absent. No other values, no blanks. Labels are never imputed. |
| `SLOPE` | yes | numeric | LiDAR DTM slope (Horn's method), extracted at the point. Record the units (degrees or percent) in your Methods. |
| `MOUNDNESS` | yes | numeric | LiDAR moundness from focal statistics on the DTM, extracted at the point. |
| `BLUE` | yes | numeric | PlanetScope blue-band value at the point. Use one consistent product (e.g. surface reflectance) and one scale. |
| `BLOCK_ID` | yes | integer | Spatial block containing the point. |
| `FOLD_ID` | yes | integer | Cross-validation fold assigned to the point's block. There must be exactly `model$number_of_folds` distinct values (default 5, e.g. 1–5). |

Rules the pipeline enforces (it stops with an error if any is broken):

- Every `BLOCK_ID` maps to exactly **one** `FOLD_ID`.
- Every fold has at least `validation$min_observations_per_fold` rows and at
  least `validation$min_per_class_per_fold` presences **and** absences.
- Predictors are numeric and finite. Missing values stop the run by default,
  or those rows are dropped and reported if `validation$missing_values: "drop"`.
- `ID` values are unique.

Conceptual example (**illustration of layout only, not data**):

```text
ID,X,Y,SALMONBERRY,SLOPE,MOUNDNESS,BLUE,BLOCK_ID,FOLD_ID
<id>,<easting>,<northing>,<0 or 1>,<slope>,<moundness>,<blue>,<block>,<fold>
```

## Test file (`data$test_path`)

Held-out observations, never used for fitting, tuning, or model choice.

| Column | Required | Notes |
|---|---|---|
| `ID` | yes | Unique; must not appear in the training file. |
| `X`, `Y` | recommended | Copied into `test_predictions.csv`. |
| `SALMONBERRY` | yes | `1` / `0`. |
| `SLOPE`, `MOUNDNESS`, `BLUE` | yes | Same definitions, units, and processing as in training. |
| `BLOCK_ID` | optional | If present, the pipeline warns when a test block also occurs in training (spatial leakage between train and test). |
| `FOLD_ID` | not used | May be present; it is ignored. |

## Adding predictors later

1. Add the column to both CSV files.
2. Add a line under `variables: predictors:` in `config.yaml`, e.g.
   `green: "GREEN"`.
3. Reference the key in one or more models, e.g. `predictors: [slope, green]`.

Predictor column names must be valid R names: letters, digits, `_` and `.`,
not starting with a digit or `.`. ArcGIS field names normally satisfy this.

## Export checklist (ArcGIS Pro)

- Export the attribute table with **Table To Table / Export Table** to `.csv`.
- Check that NoData cells did not become `-9999` or a similar sentinel. If they
  did, fix them in ArcGIS or add the sentinel to `data$na_values`.
- Use `.` as the decimal separator.
- Do not include thousands separators.
- Keep one header row with the field names.
