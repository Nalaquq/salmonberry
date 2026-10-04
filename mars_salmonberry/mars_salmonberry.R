#!/usr/bin/env Rscript
# =============================================================================
# mars_salmonberry.R
#
# Configuration-driven MARS (Multivariate Adaptive Regression Splines)
# classification pipeline comparing LiDAR-derived SLOPE and MOUNDNESS (with and
# without PlanetScope BLUE reflectance) as predictors of salmonberry
# (Rubus spectabilis) occurrence, evaluated with user-supplied SPATIAL
# cross-validation folds and a separate, untouched test set.
#
# -----------------------------------------------------------------------------
# STATUS OF THE DATA
# -----------------------------------------------------------------------------
# This script was written BEFORE any labeled observations exist. It contains no
# data, generates no data, downloads no data, and never fabricates labels,
# predictor values, spatial blocks, or folds. Every number it produces comes
# from the CSV files named in the YAML configuration. If those files do not
# exist, the script stops with an explanation of what must be supplied.
#
# -----------------------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------------------
#   # Check configuration and data only (no model fitting):
#   Rscript mars_salmonberry.R --config config.yaml --validate-only
#
#   # Full run (spatial CV, comparison, final fit, test evaluation):
#   Rscript mars_salmonberry.R --config config.yaml
#
#   # Interactively, from an R session (sourcing does NOT start a run):
#   source("mars_salmonberry.R")
#   results <- run_pipeline("config.yaml")
#
# Relative paths in the YAML file are resolved relative to the directory that
# contains the YAML file, not the current working directory.
#
# Requires R >= 4.1 (native pipe |>).
#
# =============================================================================
# PART A. SCIENTIFIC BACKGROUND
# =============================================================================
#
# A.1 Hypothesis
# --------------
# Preliminary hypothesis: a MARS model using LiDAR-derived slope has greater
# out-of-sample predictive performance (spatially cross-validated ROC-AUC) for
# salmonberry occurrence than a MARS model using LiDAR-derived moundness.
#
# Five competing models are fit with identical settings and identical folds:
#
#   M1  SALMONBERRY ~ MOUNDNESS
#   M2  SALMONBERRY ~ SLOPE
#   M3  SALMONBERRY ~ MOUNDNESS + BLUE
#   M4  SALMONBERRY ~ SLOPE + BLUE
#   M5  SALMONBERRY ~ MOUNDNESS + SLOPE + BLUE
#
# Primary contrast:    Delta AUC = AUC(M2) - AUC(M1)      (per spatial fold)
# Secondary contrast:  Delta AUC = AUC(M4) - AUC(M3)      (per spatial fold)
#
# The model list and contrasts are defined in config.yaml, so predictors can be
# added later (e.g. other PlanetScope bands) without editing this script.
#
# A.2 Why this is a CLASSIFICATION problem, not ordinary regression
# -----------------------------------------------------------------
# The response Y is binary: 1 = salmonberry present, 0 = absent. It is a
# Bernoulli random variable, Y | X ~ Bernoulli(p(X)), where p(X) = P(Y = 1 | X)
# is the probability of occurrence given the predictors X. Treating Y as a
# continuous Gaussian response (ordinary least-squares regression) would be
# inappropriate because:
#   (1) fitted values would not be constrained to [0, 1] and so cannot be read
#       as probabilities;
#   (2) the variance of a Bernoulli variable, p(1 - p), depends on the mean,
#       violating the constant-variance assumption of least squares;
#   (3) the errors are not normally distributed (Y takes only two values).
# The appropriate model is a generalized linear model with a binomial error
# distribution and a logit link. Here the linear predictor is replaced by a
# MARS basis expansion (see Part B), giving a "MARS-logistic" model:
#
#   logit(p(X)) = log( p(X) / (1 - p(X)) ) = eta(X) = b0 + sum_m bm * Bm(X)
#   p(X)        = 1 / (1 + exp(-eta(X)))
#
# The model therefore outputs a PROBABILITY of salmonberry occurrence for each
# location. A hard presence/absence class is obtained only afterwards by
# comparing p(X) with a classification threshold (config:
# model$classification_threshold, default 0.5): predicted_class = 1 if
# p(X) >= threshold, else 0. ROC-AUC is computed from the probabilities and
# does not depend on that threshold.
#
# In the earth package this is requested with glm = list(family = binomial).
#
# =============================================================================
# PART B. THE MATHEMATICS OF MARS (for the statistical Methods section)
# =============================================================================
#
# B.1 Hinge (truncated linear) basis functions
# --------------------------------------------
# MARS (Friedman 1991, Annals of Statistics 19:1-67) builds a flexible function
# out of "hinge" functions. For a predictor x and a knot t:
#
#   h(x - t) = max(0, x - t)     zero left of t, rises with slope 1 right of t
#   h(t - x) = max(0, t - x)     falls with slope -1 left of t, zero right of t
#
# The two functions form a "reflected pair" at knot t. A weighted sum such as
#
#   eta(x) = b0 + b1 * max(0, x - t) + b2 * max(0, t - x)
#
# is continuous and piecewise linear: its slope is -b2 for x < t and +b1 for
# x > t. The knot t is where the slope changes. With several knots on the same
# predictor, MARS approximates smooth nonlinear, threshold-like, or unimodal
# responses (for example, occurrence rising with slope up to a point and then
# levelling off) without the analyst choosing the shape in advance. Knot
# locations are chosen from the observed values of the predictor.
#
# If the best "knot" for a predictor lies at its minimum, the hinge is simply a
# linear term; earth then enters the predictor linearly (Auto.linpreds = TRUE).
#
# NOTE: in this binary model, the piecewise-linear shape applies on the LOGIT
# scale. After the inverse-logit transform, the fitted probability curve is
# smooth and sigmoid-like between knots, with bends near the knots.
#
# B.2 General model form
# ----------------------
#   f(X) = b0 + sum_{m = 1}^{M} bm * Bm(X)
#
#   b0     intercept
#   bm     coefficient for basis function m
#   Bm(X)  a MARS basis function: either a single hinge, h(xj - t) or
#          h(t - xj), or a product of hinges on DIFFERENT predictors
#   M      number of basis functions retained after pruning
#
# For this binary response, f(X) is the linear predictor eta(X) on the logit
# scale (Part A.2).
#
# B.3 Interactions
# ----------------
# An interaction is represented by a product of hinges on different predictors,
# for example
#
#   B(X) = max(0, SLOPE - 4.2) * max(0, 0.15 - BLUE)
#
# which is non-zero only where SLOPE > 4.2 AND BLUE < 0.15, so the effect of
# slope can differ depending on blue reflectance. The maximum number of hinges
# in one product is the "degree" (config: model$mars$degree). degree = 1 gives
# an additive model (no interactions; each predictor has its own piecewise-
# linear effect); degree = 2 permits pairwise interactions. A predictor may
# appear at most once within a product. For one-predictor models (M1, M2) the
# degree setting has no effect.
#
# B.4 Forward pass (basis-function construction)
# ----------------------------------------------
# Starting from the intercept-only model (B0(X) = 1), MARS repeatedly searches
# over
#   - every existing basis function Bl (the "parent", whose degree is below the
#     maximum),
#   - every predictor xj not already in Bl,
#   - every candidate knot t among the observed values of xj (subject to the
#     minspan / endspan rules that keep knots away from the data extremes and
#     from each other),
# and adds the reflected pair  Bl(X) * h(xj - t)  and  Bl(X) * h(t - xj)
# that most reduces the residual sum of squares. The forward pass stops when
# the maximum number of terms (nk) is reached, when adding terms improves R^2 by
# less than `thresh`, or when R^2 is close to 1. It deliberately OVERFITS: it
# produces a large model to be simplified in the backward pass.
#
# B.5 Backward pass (pruning) and Generalized Cross-Validation
# ------------------------------------------------------------
# The backward pass removes, one at a time, the term whose removal increases
# the residual sum of squares the least, producing a nested sequence of
# candidate models of every size from M = 1 (intercept only) upward. The size
# chosen is the one that minimizes the Generalized Cross-Validation criterion
#
#   GCV(M) = [ RSS(M) / N ] / [ 1 - C(M) / N ]^2
#
#   C(M) = M + d * (M - 1) / 2
#
# where N is the number of observations, M the number of terms, and d the
# "penalty" per knot (earth default d = 2 for additive models, d = 3 when
# interactions are allowed). C(M) is the effective number of parameters: each
# knot costs more than one ordinary coefficient because its location was also
# estimated from the data. GCV approximates leave-one-out prediction error and
# penalizes complexity, so it protects against overfitting.
#
# B.6 Model complexity and how earth controls it
# ----------------------------------------------
# Complexity is the number of retained basis functions (and hence knots) and
# their maximum interaction degree. In earth it is controlled by:
#   degree   maximum interaction order (config: model$mars$degree)
#   nk       maximum number of terms in the forward pass
#   thresh   minimum R^2 improvement required to continue the forward pass
#   minspan, endspan  minimum spacing of knots / distance from data ends
#            (0 = earth's automatic rules based on Friedman 1991)
#   penalty  GCV cost per knot (d above)
#   pmethod  pruning method ("backward" = standard MARS pruning by GCV)
#   nprune   maximum number of terms kept after pruning (NULL = let GCV decide)
#
# B.7 How earth fits the binary model
# -----------------------------------
# With glm = list(family = binomial), earth works in two stages:
#   (1) the forward and backward passes above are run on the 0/1 response with
#       least squares, and GCV selects the basis functions and knots;
#   (2) a logistic GLM (binomial error, logit link) is then fit by maximum
#       likelihood, using the selected basis functions as predictors.
# Predictions with type = "response" are therefore probabilities from the
# logistic GLM. A Methods sentence could read: "Basis functions and knots were
# selected with the MARS forward-selection / backward-pruning algorithm using
# generalized cross-validation, and the coefficients of the retained basis
# functions were then estimated by maximum likelihood in a binomial GLM with a
# logit link (R package earth; Milborrow 2024)."
#
# B.8 Optional tuning with caret (config: model$tuning$method = "spatial_cv")
# ---------------------------------------------------------------------------
# By default ("gcv") complexity is set by earth's internal GCV pruning, as in
# standard MARS. Alternatively, degree and nprune can be chosen by NESTED
# spatial cross-validation using caret: inside each outer training set, caret
# holds out each of the remaining folds in turn (so the inner resamples also
# respect the spatial blocks), chooses the (degree, nprune) with the highest
# inner ROC-AUC, and refits on the full outer training set. The outer
# validation fold and the test set are never seen during tuning.
#
# =============================================================================
# PART C. WHY SPATIAL CROSS-VALIDATION IS NECESSARY
# =============================================================================
#
# The observations are points in a landscape. Nearby points tend to share
# elevation, slope, microtopography (moundness), vegetation, soils, moisture,
# and spectral reflectance, and whether salmonberry is present is also
# spatially clustered (spatial autocorrelation; Tobler's "first law of
# geography"). If single points were randomly assigned to training and
# validation sets, most validation points would have near-identical neighbours
# in the training set. The model could then score well by recognising local
# conditions it has, in effect, already seen, not by learning a relationship
# that carries over to new places. This SPATIAL LEAKAGE makes performance
# estimates optimistic and can also favour the more flexible or more locally
# specific predictor, which would bias the slope-vs-moundness comparison
# (Roberts et al. 2017, Ecography 40:913-929; Valavi et al. 2019, Methods in
# Ecology and Evolution 10:225-232).
#
# Blocked spatial cross-validation reduces this:
#
#   Study area
#        |
#   Spatial blocks (contiguous areas, ideally larger than the range of spatial
#        |          autocorrelation in the predictors / residuals)
#   Assign whole blocks to K folds
#        |
#   Each block belongs to exactly ONE fold
#        |
#   Spatial CV: for k = 1..K, train on all folds except k, validate on fold k
#
# Because an entire block is held out together, validation points are
# separated from training points by (at least part of) a block, which makes the
# estimate closer to predicting salmonberry in an unsurveyed area.
#
# Blocks and folds are created OUTSIDE R (ArcGIS Pro) and supplied as the
# BLOCK_ID and FOLD_ID columns. This script NEVER creates, reshuffles, or
# repairs folds; it only checks them. validate_spatial_folds() stops the run if
# any BLOCK_ID appears in more than one FOLD_ID, because that would put
# neighbouring observations on both sides of the train/validation split.
#
# All five models use exactly the same folds, so fold-to-fold differences in
# difficulty affect every model equally and the per-fold Delta AUC is a paired
# comparison.
#
# =============================================================================
# PART D. PROTECTING THE TEST SET
# =============================================================================
#
#   Training data -> Spatial CV -> Model comparison -> Model selection
#     -> Refit selected model on ALL training data -> Untouched test data
#     -> Final evaluation
#
# The test CSV is read only inside evaluate_test_data(), which runs after every
# modeling decision (predictors, knots, complexity, tuning, threshold, model
# choice) has been fixed from the training data. The test data are never used
# to select predictors, tune MARS, select knots or complexity, choose a
# threshold, or choose between models. The script also checks that no ID
# appears in both files and warns if any spatial block is shared.
#
# =============================================================================
# PART E. PACKAGES AND THEIR ROLES
# =============================================================================
#   earth    MARS implementation (forward pass, GCV pruning, binomial GLM step,
#            variable importance via evimp()).
#   caret    optional nested spatial tuning of degree / nprune
#            (caret::train with user-supplied fold indices).
#   yaml     reads config.yaml; writes the configuration actually used.
#   dplyr    data manipulation (grouping, summarising, joining).
#   readr    reads the CSV inputs and writes the CSV outputs.
#   pROC     ROC curves and ROC-AUC from predicted probabilities; DeLong
#            confidence interval for the single test-set AUC.
#   ggplot2  publication figures.
#   tidyr    reshaping (wide AUC tables for paired Delta AUC; long tables for
#            plotting).
# No other packages are required. Packages are never installed automatically.
# =============================================================================


# =============================================================================
# 1. Constants and small utilities
# =============================================================================

REQUIRED_PACKAGES <- c(
  "earth", "caret", "yaml", "dplyr", "readr", "pROC", "ggplot2", "tidyr"
)

# Internal (standardized) column names. User column names come from the YAML
# file; data are renamed to these internally so the code is independent of how
# the fields were named in ArcGIS Pro. Outputs use the user's names again.
INTERNAL_ID <- ".id"
INTERNAL_X <- ".x"
INTERNAL_Y <- ".y"
INTERNAL_BLOCK <- ".block"
INTERNAL_FOLD <- ".fold"
INTERNAL_RESPONSE <- ".response01"      # numeric 0 / 1
INTERNAL_CLASS <- ".response_class"     # factor: absent / present
CLASS_LEVELS <- c("absent", "present")  # earth and caret model the 2nd level

PLACEHOLDER_PATTERN <- "path/to"

# Colour-blind-safe categorical palette, applied in fixed order.
SERIES_COLOURS <- c(
  "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
  "#e87ba4", "#008300", "#4a3aa7", "#e34948"
)


#' Stop with a formatted, user-facing error message.
#'
#' @param ... Character pieces; each becomes one line of the message.
pipeline_stop <- function(...) {
  message_text <- paste(c(...), collapse = "\n")
  stop(paste0("\n\nERROR:\n", message_text, "\n"), call. = FALSE)
}


#' Emit a formatted warning that is also easy to spot in logs.
pipeline_warning <- function(...) {
  warning(paste0("WARNING: ", paste(c(...), collapse = "\n")),
          call. = FALSE, immediate. = TRUE)
}


#' Print a time-stamped progress message.
log_message <- function(...) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"),
              paste0(...)))
}


#' Evaluate an expression, returning its value plus any warnings it raised.
#'
#' Warnings from glm() (e.g. "fitted probabilities numerically 0 or 1") are
#' recorded in the diagnostics instead of being lost or printed repeatedly.
with_captured_warnings <- function(expr) {
  captured <- character(0)
  value <- withCallingHandlers(
    expr,
    warning = function(w) {
      captured <<- c(captured, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(value = value, warnings = unique(captured))
}


#' Format numbers for console summaries and interpretation notes.
format_number <- function(x, digits = 3, signed = FALSE) {
  if (length(x) == 0 || is.na(x)) {
    return("NA")
  }
  formatC(x, format = "f", digits = digits, flag = if (signed) "+" else "")
}


#' Is the path absolute (Windows drive, UNC, or POSIX root)?
is_absolute_path <- function(path) {
  grepl("^([A-Za-z]:)?[/\\\\]", path) || startsWith(path, "~")
}


#' Resolve a (possibly relative) path against a base directory.
resolve_path <- function(path, base_dir) {
  if (is.null(path) || !nzchar(path)) {
    return(NULL)
  }
  full_path <- if (is_absolute_path(path)) path else file.path(base_dir, path)
  normalizePath(path.expand(full_path), winslash = "/", mustWork = FALSE)
}


# =============================================================================
# 2. Command line and package checks
# =============================================================================

#' Parse command-line arguments.
#'
#' Supported: --config <path> (default "config.yaml"), --validate-only.
parse_command_line <- function(args = commandArgs(trailingOnly = TRUE)) {
  parsed <- list(config_path = "config.yaml", validate_only = FALSE)
  i <- 1
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg == "--config") {
      if (i == length(args)) {
        pipeline_stop("--config must be followed by the path to a YAML file.")
      }
      parsed$config_path <- args[[i + 1]]
      i <- i + 2
    } else if (startsWith(arg, "--config=")) {
      parsed$config_path <- sub("^--config=", "", arg)
      i <- i + 1
    } else if (arg == "--validate-only") {
      parsed$validate_only <- TRUE
      i <- i + 1
    } else if (arg %in% c("-h", "--help")) {
      cat(
        "Usage: Rscript mars_salmonberry.R [--config config.yaml]",
        "[--validate-only]\n"
      )
      quit(save = "no", status = 0)
    } else {
      pipeline_stop(sprintf("Unrecognised argument: '%s'.", arg),
                    "Use --config <path> and optionally --validate-only.")
    }
  }
  parsed
}


#' Stop if any required package is missing. Never installs anything.
check_required_packages <- function(packages = REQUIRED_PACKAGES) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1),
                              quietly = TRUE)]
  if (length(missing) > 0) {
    pipeline_stop(
      sprintf("Required R package(s) not installed: %s.",
              paste(missing, collapse = ", ")),
      "Install them once with:",
      sprintf("  install.packages(c(%s))",
              paste(sprintf('"%s"', missing), collapse = ", "))
    )
  }
  invisible(TRUE)
}


# =============================================================================
# 3. Configuration
# =============================================================================

#' Default settings used when a key is absent from config.yaml.
#'
#' Data paths, variable names, models, and comparisons have NO defaults: they
#' must be stated explicitly in the configuration.
default_settings <- function() {
  list(
    data = list(
      na_values = c("", "NA", "NaN", "NULL", "<Null>")
    ),
    response_coding = list(presence = 1, absence = 0),
    model = list(
      seed = 12345,
      classification_threshold = 0.5,
      number_of_folds = 5,
      mars = list(
        degree = 1,
        nprune = NULL,
        penalty = NULL,
        nk = NULL,
        thresh = 0.001,
        minspan = 0,
        endspan = 0,
        pmethod = "backward",
        glm_maxit = 100
      ),
      tuning = list(
        method = "gcv",
        degree_grid = c(1, 2),
        nprune_grid = c(2, 3, 4, 5, 6, 8, 10)
      )
    ),
    final_model = list(
      selected_model = NULL,
      selection_rule = "highest_mean_cv_auc"
    ),
    evaluation = list(
      run_test_evaluation = TRUE,
      evaluate_all_models_on_test = FALSE
    ),
    validation = list(
      missing_values = "error",
      min_observations_per_fold = 10,
      min_per_class_per_fold = 2,
      minority_class_warning_proportion = 0.2,
      check_duplicate_coordinates = TRUE
    ),
    output = list(
      directory = "output",
      save_fold_models = TRUE
    ),
    variable_importance = list(metric = "gcv"),
    plots = list(
      width_in = 7,
      height_in = 5,
      dpi = 300,
      formats = c("png", "pdf"),
      response_curve_predictors = c("slope", "moundness"),
      response_curve_points = 200
    )
  )
}


#' Recursively fill missing configuration entries from defaults.
#'
#' Only named lists are merged; unnamed lists (models, comparisons) and vectors
#' supplied by the user replace the default entirely.
merge_with_defaults <- function(defaults, supplied) {
  if (is.null(supplied)) {
    return(defaults)
  }
  for (key in names(defaults)) {
    default_value <- defaults[[key]]
    if (!(key %in% names(supplied)) || is.null(supplied[[key]])) {
      supplied[key] <- list(default_value)
    } else if (is.list(default_value) && !is.null(names(default_value)) &&
               is.list(supplied[[key]])) {
      supplied[[key]] <- merge_with_defaults(default_value, supplied[[key]])
    }
  }
  supplied
}


#' Read config.yaml, fill defaults, resolve paths, and validate its structure.
#'
#' Does not touch any data file.
load_config <- function(config_path) {
  if (!file.exists(config_path)) {
    pipeline_stop(
      sprintf("Configuration file not found: %s", config_path),
      "Pass the path with: Rscript mars_salmonberry.R --config <path>"
    )
  }
  config_path <- normalizePath(config_path, winslash = "/")
  raw_config <- tryCatch(
    yaml::read_yaml(config_path),
    error = function(e) {
      pipeline_stop(sprintf("Could not parse YAML file %s:", config_path),
                    conditionMessage(e))
    }
  )
  config <- merge_with_defaults(default_settings(), raw_config)
  config$meta <- list(
    config_path = config_path,
    config_dir = dirname(config_path),
    raw_config = raw_config
  )
  validate_config(config)

  base_dir <- config$meta$config_dir
  config$data$train_path_resolved <- resolve_path(config$data$train_path,
                                                  base_dir)
  config$data$test_path_resolved <- resolve_path(config$data$test_path,
                                                 base_dir)
  config$output$directory_resolved <- resolve_path(config$output$directory,
                                                   base_dir)
  config
}


#' Check that the configuration is complete and internally consistent.
validate_config <- function(config) {
  problems <- character(0)
  add_problem <- function(...) problems <<- c(problems, paste0(...))

  is_single_string <- function(x) {
    is.character(x) && length(x) == 1 && !is.na(x) && nzchar(x)
  }

  # --- data ---
  if (!is_single_string(config$data$train_path)) {
    add_problem("data$train_path must be a file path.")
  }
  if (isTRUE(config$evaluation$run_test_evaluation) &&
      !is_single_string(config$data$test_path)) {
    add_problem("data$test_path must be a file path when ",
                "evaluation$run_test_evaluation is true.")
  }

  # --- variables ---
  variables <- config$variables
  for (key in c("id", "response", "block_id", "fold_id")) {
    if (!is_single_string(variables[[key]])) {
      add_problem(sprintf("variables$%s must name a column.", key))
    }
  }
  for (key in c("x", "y")) {
    if (!is.null(variables[[key]]) && !is_single_string(variables[[key]])) {
      add_problem(sprintf("variables$%s must name a column or be null.", key))
    }
  }
  predictor_map <- variables$predictors
  if (!is.list(predictor_map) || length(predictor_map) == 0 ||
      is.null(names(predictor_map))) {
    add_problem("variables$predictors must be a named mapping, e.g. ",
                "slope: \"SLOPE\".")
  } else {
    predictor_columns <- unlist(predictor_map)
    bad_names <- predictor_columns[make.names(predictor_columns) !=
                                     predictor_columns |
                                     startsWith(predictor_columns, ".")]
    if (length(bad_names) > 0) {
      add_problem("Predictor column names must be syntactic R names ",
                  "(letters, digits, '_' and '.', not starting with a digit ",
                  "or '.'). Rename in ArcGIS Pro: ",
                  paste(bad_names, collapse = ", "))
    }
    if (anyDuplicated(predictor_columns)) {
      add_problem("Two predictor keys map to the same column name.")
    }
  }

  # --- response coding ---
  presence <- config$response_coding$presence
  absence <- config$response_coding$absence
  if (length(presence) != 1 || length(absence) != 1 ||
      identical(as.character(presence), as.character(absence))) {
    add_problem("response_coding$presence and $absence must be two ",
                "different single values (default 1 and 0).")
  }

  # --- models ---
  models <- config$models
  if (!is.list(models) || length(models) == 0) {
    add_problem("The 'models' section must list at least one model.")
  } else {
    model_names <- vapply(models, function(m) {
      if (is_single_string(m$name)) m$name else NA_character_
    }, character(1))
    if (anyNA(model_names)) {
      add_problem("Every model needs a 'name'.")
    }
    if (anyDuplicated(stats::na.omit(model_names))) {
      add_problem("Model names must be unique.")
    }
    for (m in models) {
      unknown <- setdiff(unlist(m$predictors), names(predictor_map))
      if (length(unlist(m$predictors)) == 0) {
        add_problem(sprintf("Model '%s' has no predictors.", m$name))
      }
      if (length(unknown) > 0) {
        add_problem(sprintf(
          "Model '%s' uses predictor key(s) not in variables$predictors: %s",
          m$name, paste(unknown, collapse = ", ")))
      }
    }

    # --- comparisons ---
    for (comparison in config$comparisons) {
      for (side in c("model_a", "model_b")) {
        if (!(comparison[[side]] %in% model_names)) {
          add_problem(sprintf("Comparison '%s': %s '%s' is not a model name.",
                              comparison$name, side, comparison[[side]]))
        }
      }
    }

    selected <- config$final_model$selected_model
    if (!is.null(selected) && !(selected %in% model_names)) {
      add_problem(sprintf("final_model$selected_model '%s' is not a model name.",
                          selected))
    }
  }

  # --- model settings ---
  threshold <- config$model$classification_threshold
  if (!is.numeric(threshold) || length(threshold) != 1 ||
      threshold <= 0 || threshold >= 1) {
    add_problem("model$classification_threshold must be a number in (0, 1).")
  }
  n_folds <- config$model$number_of_folds
  if (!is.numeric(n_folds) || length(n_folds) != 1 || n_folds < 2 ||
      n_folds != round(n_folds)) {
    add_problem("model$number_of_folds must be an integer >= 2.")
  }
  if (!is.numeric(config$model$seed) || length(config$model$seed) != 1) {
    add_problem("model$seed must be a single number.")
  }
  degree <- config$model$mars$degree
  if (!is.numeric(degree) || degree < 1 || degree != round(degree)) {
    add_problem("model$mars$degree must be a positive integer.")
  }
  tuning_method <- config$model$tuning$method
  if (!(tuning_method %in% c("gcv", "spatial_cv"))) {
    add_problem("model$tuning$method must be \"gcv\" or \"spatial_cv\".")
  }
  if (identical(tuning_method, "spatial_cv") && is.numeric(n_folds) &&
      n_folds < 3) {
    add_problem("spatial_cv tuning needs number_of_folds >= 3 (nested ",
                "inner folds need at least two folds).")
  }
  if (identical(tuning_method, "spatial_cv") &&
      any(config$model$tuning$nprune_grid < 2)) {
    add_problem("model$tuning$nprune_grid values must be >= 2 ",
                "(nprune counts the intercept).")
  }
  if (!(config$validation$missing_values %in% c("error", "drop"))) {
    add_problem("validation$missing_values must be \"error\" or \"drop\".")
  }
  if (!(config$variable_importance$metric %in% c("gcv", "nsubsets", "rss"))) {
    add_problem("variable_importance$metric must be gcv, nsubsets, or rss.")
  }
  if (!(config$final_model$selection_rule %in% "highest_mean_cv_auc")) {
    add_problem("final_model$selection_rule must be \"highest_mean_cv_auc\" ",
                "(or set final_model$selected_model explicitly).")
  }

  if (length(problems) > 0) {
    pipeline_stop(
      sprintf("The configuration file %s is invalid:",
              config$meta$config_path),
      paste0("  - ", problems)
    )
  }
  invisible(TRUE)
}


# =============================================================================
# 4. Input files
# =============================================================================

#' Confirm that configured input files exist before any work is done.
#'
#' @param include_test Whether the test file must exist for this run.
validate_file_paths <- function(config, include_test) {
  checks <- list(list(
    label = "Training data (data$train_path)",
    configured = config$data$train_path,
    resolved = config$data$train_path_resolved
  ))
  if (include_test) {
    checks[[2]] <- list(
      label = "Test data (data$test_path)",
      configured = config$data$test_path,
      resolved = config$data$test_path_resolved
    )
  }

  problems <- character(0)
  for (check in checks) {
    if (grepl(PLACEHOLDER_PATTERN, check$configured, fixed = TRUE)) {
      problems <- c(problems, sprintf(
        "  - %s still contains the template placeholder '%s'.",
        check$label, check$configured))
    } else if (!file.exists(check$resolved)) {
      problems <- c(problems, sprintf("  - %s not found: %s",
                                      check$label, check$resolved))
    }
  }

  if (length(problems) > 0) {
    pipeline_stop(
      "The labeled salmonberry data required by this pipeline are not",
      "available yet.",
      "",
      problems,
      "",
      "This pipeline does not create, simulate, or download data. Before",
      "running it you need to:",
      "  1. label salmonberry presence (1) / absence (0) observations;",
      "  2. extract SLOPE, MOUNDNESS, and BLUE raster values at each point;",
      "  3. assign every point a BLOCK_ID (spatial block) and FOLD_ID",
      "     (each block in exactly one fold);",
      "  4. split the points into training and test sets and export both",
      "     as CSV (see DATA_SCHEMA.md);",
      sprintf("  5. put the file paths in %s.", config$meta$config_path),
      "Relative paths are resolved against the folder containing the YAML file."
    )
  }
  invisible(TRUE)
}


#' Read one observation CSV with every column as text.
#'
#' Reading as text lets validate_data() report exactly which values are not
#' numeric, instead of readr silently turning them into NA.
load_data <- function(path, config) {
  data <- tryCatch(
    readr::read_csv(
      path,
      col_types = readr::cols(.default = readr::col_character()),
      na = character(0),
      trim_ws = TRUE,
      show_col_types = FALSE,
      progress = FALSE
    ),
    error = function(e) {
      pipeline_stop(sprintf("Could not read CSV file %s:", path),
                    conditionMessage(e))
    }
  )
  parse_problems <- readr::problems(data)
  if (nrow(parse_problems) > 0) {
    pipeline_stop(
      sprintf("readr reported %d parsing problem(s) in %s, e.g.:",
              nrow(parse_problems), path),
      utils::capture.output(print(utils::head(parse_problems, 5)))
    )
  }
  if (nrow(data) == 0) {
    pipeline_stop(sprintf("The file %s contains no rows.", path))
  }
  data <- as.data.frame(data, stringsAsFactors = FALSE)

  # Convert configured NoData strings to NA (ArcGIS exports often use "" or
  # "<Null>"). Numeric sentinels such as -9999 should be removed in ArcGIS or
  # added to data$na_values explicitly.
  na_values <- as.character(config$data$na_values)
  data[] <- lapply(data, function(column) {
    column[column %in% na_values] <- NA_character_
    column
  })
  data
}


# =============================================================================
# 5. Data validation
# =============================================================================

#' Collect all predictor column names used by any configured model.
all_model_predictor_columns <- function(config) {
  keys <- unique(unlist(lapply(config$models, function(m) m$predictors)))
  unname(unlist(config$variables$predictors[keys]))
}


#' Convert text to numbers, reporting any value that is not numeric.
parse_numeric_column <- function(values, column_name, role) {
  numbers <- suppressWarnings(as.numeric(values))
  not_numeric <- !is.na(values) & is.na(numbers)
  if (any(not_numeric)) {
    examples <- utils::head(unique(values[not_numeric]), 5)
    pipeline_stop(
      sprintf("%s data: column '%s' contains %d non-numeric value(s),",
              role, column_name, sum(not_numeric)),
      sprintf("e.g. %s.", paste(sprintf("'%s'", examples), collapse = ", ")),
      "Predictors, coordinates, BLOCK_ID and FOLD_ID must be numeric.",
      "Check the export from ArcGIS Pro (decimal separator, text fields)."
    )
  }
  numbers
}


#' Validate one data set and convert it to the internal standard format.
#'
#' Checks: missing columns, response coding, non-numeric predictors, missing
#' and infinite values, duplicate IDs, class imbalance, duplicate coordinates,
#' and (training data only) valid BLOCK_ID / FOLD_ID values.
#'
#' @param raw_data Data frame from load_data().
#' @param role "Training" or "Test".
#' @return list(data = standardized data frame, report = character lines)
validate_data <- function(raw_data, config, role = c("Training", "Test")) {
  role <- match.arg(role)
  variables <- config$variables
  predictor_columns <- all_model_predictor_columns(config)
  is_training <- role == "Training"
  report <- sprintf("%s data: %d rows read.", role, nrow(raw_data))

  # --- required columns ---
  required <- c(variables$id, variables$response, predictor_columns)
  if (is_training) {
    required <- c(required, variables$block_id, variables$fold_id)
  }
  missing_columns <- setdiff(required, names(raw_data))
  if (length(missing_columns) > 0) {
    pipeline_stop(
      sprintf("%s data: required column(s) missing: %s",
              role, paste(missing_columns, collapse = ", ")),
      sprintf("Columns found: %s", paste(names(raw_data), collapse = ", ")),
      "Column names are case-sensitive; change them in config.yaml",
      "(variables section) if your export uses different names."
    )
  }
  coordinate_columns <- c(variables$x, variables$y)
  absent_coordinates <- setdiff(coordinate_columns, names(raw_data))
  if (length(absent_coordinates) > 0) {
    pipeline_warning(sprintf(
      "%s data: coordinate column(s) %s not found; coordinates will not be ",
      role, paste(absent_coordinates, collapse = ", ")),
      "carried into the prediction files.")
  }
  has_coordinates <- length(coordinate_columns) == 2 &&
    length(absent_coordinates) == 0

  # --- response: never imputed, never recoded beyond the configured codes ---
  response_raw <- raw_data[[variables$response]]
  presence_code <- config$response_coding$presence
  absence_code <- config$response_coding$absence
  response_numeric <- suppressWarnings(as.numeric(response_raw))
  matches_code <- function(code) {
    if (is.numeric(code)) {
      !is.na(response_numeric) & response_numeric == code
    } else {
      !is.na(response_raw) & response_raw == as.character(code)
    }
  }
  is_presence <- matches_code(presence_code)
  is_absence <- matches_code(absence_code)
  if (anyNA(response_raw)) {
    pipeline_stop(
      sprintf("%s data: %d observation(s) have a missing %s value.",
              role, sum(is.na(response_raw)), variables$response),
      "Labels are never imputed. Label or remove these points in ArcGIS Pro."
    )
  }
  invalid_response <- !(is_presence | is_absence)
  if (any(invalid_response)) {
    pipeline_stop(
      sprintf("%s data: column %s must contain only %s (presence) or %s",
              role, variables$response, presence_code, absence_code),
      sprintf("(absence). Found invalid value(s): %s (%d rows).",
              paste(utils::head(unique(response_raw[invalid_response]), 10),
                    collapse = ", "),
              sum(invalid_response))
    )
  }

  # --- identifiers ---
  ids <- raw_data[[variables$id]]
  if (anyNA(ids)) {
    pipeline_stop(sprintf("%s data: %d row(s) have a missing %s.",
                          role, sum(is.na(ids)), variables$id))
  }
  duplicated_ids <- unique(ids[duplicated(ids)])
  if (length(duplicated_ids) > 0) {
    pipeline_stop(
      sprintf("%s data: %d duplicated %s value(s), e.g. %s.",
              role, length(duplicated_ids), variables$id,
              paste(utils::head(duplicated_ids, 10), collapse = ", ")),
      "Each observation must have a unique identifier."
    )
  }

  # --- standardized data frame ---
  data <- data.frame(row.names = seq_len(nrow(raw_data)))
  data[[INTERNAL_ID]] <- ids
  if (has_coordinates) {
    data[[INTERNAL_X]] <- parse_numeric_column(raw_data[[variables$x]],
                                               variables$x, role)
    data[[INTERNAL_Y]] <- parse_numeric_column(raw_data[[variables$y]],
                                               variables$y, role)
  }
  data[[INTERNAL_RESPONSE]] <- as.integer(is_presence)
  data[[INTERNAL_CLASS]] <- factor(
    ifelse(is_presence, CLASS_LEVELS[2], CLASS_LEVELS[1]),
    levels = CLASS_LEVELS
  )
  for (column in predictor_columns) {
    data[[column]] <- parse_numeric_column(raw_data[[column]], column, role)
  }

  has_block <- variables$block_id %in% names(raw_data)
  if (has_block) {
    data[[INTERNAL_BLOCK]] <- parse_numeric_column(
      raw_data[[variables$block_id]], variables$block_id, role)
  }
  if (is_training) {
    data[[INTERNAL_FOLD]] <- parse_numeric_column(
      raw_data[[variables$fold_id]], variables$fold_id, role)
  }

  # --- infinite values: always an error ---
  numeric_columns <- intersect(
    c(predictor_columns, INTERNAL_X, INTERNAL_Y, INTERNAL_BLOCK,
      INTERNAL_FOLD),
    names(data))
  for (column in numeric_columns) {
    n_infinite <- sum(is.infinite(data[[column]]))
    if (n_infinite > 0) {
      pipeline_stop(sprintf("%s data: column '%s' contains %d infinite value(s).",
                            role, column, n_infinite))
    }
  }

  # --- missing values in predictors / fold structure ---
  structural_columns <- intersect(c(predictor_columns, INTERNAL_BLOCK,
                                    INTERNAL_FOLD), names(data))
  incomplete <- !stats::complete.cases(data[, structural_columns, drop = FALSE])
  if (any(incomplete)) {
    column_counts <- vapply(structural_columns,
                            function(col) sum(is.na(data[[col]])), integer(1))
    column_counts <- column_counts[column_counts > 0]
    count_text <- paste(sprintf("%s: %d", display_column_name(
      names(column_counts), config), column_counts), collapse = "; ")
    if (config$validation$missing_values == "error") {
      pipeline_stop(
        sprintf("%s data: %d row(s) have missing predictor/block/fold values",
                role, sum(incomplete)),
        sprintf("(%s).", count_text),
        "Fix these in ArcGIS Pro (e.g. points outside raster extent or on",
        "NoData cells), or set validation$missing_values: \"drop\" to exclude",
        "them. Missing values are never imputed."
      )
    }
    dropped_ids <- data[[INTERNAL_ID]][incomplete]
    data <- data[!incomplete, , drop = FALSE]
    rownames(data) <- NULL
    report <- c(report, sprintf(
      "%s data: dropped %d row(s) with missing values (%s). Dropped IDs: %s",
      role, length(dropped_ids), count_text,
      paste(dropped_ids, collapse = ", ")))
    pipeline_warning(sprintf(
      "%s data: dropped %d row(s) with missing values (%s).",
      role, length(dropped_ids), count_text))
  }

  # --- integer-valued block / fold identifiers ---
  for (column in intersect(c(INTERNAL_BLOCK, INTERNAL_FOLD), names(data))) {
    values <- data[[column]]
    if (any(values != round(values))) {
      pipeline_stop(sprintf("%s data: %s must contain whole numbers only.",
                            role, display_column_name(column, config)))
    }
  }

  # --- class balance ---
  n_presence <- sum(data[[INTERNAL_RESPONSE]] == 1)
  n_absence <- sum(data[[INTERNAL_RESPONSE]] == 0)
  if (n_presence == 0 || n_absence == 0) {
    pipeline_stop(
      sprintf("%s data contain only one response class (%d presences, %d ",
              role, n_presence, n_absence),
      "absences). Both presences and absences are required."
    )
  }
  prevalence <- n_presence / (n_presence + n_absence)
  report <- c(report, sprintf(
    "%s data: %d observations used; %d presences, %d absences (prevalence %.3f).",
    role, nrow(data), n_presence, n_absence, prevalence))
  minority_share <- min(prevalence, 1 - prevalence)
  if (minority_share < config$validation$minority_class_warning_proportion) {
    message_text <- sprintf(
      paste0("%s data are imbalanced: the minority class is %.1f%% of ",
             "observations. ROC-AUC is threshold-free, but threshold-based ",
             "metrics at %.2f may be poor; interpret sensitivity/specificity ",
             "with care."),
      role, 100 * minority_share, config$model$classification_threshold)
    pipeline_warning(message_text)
    report <- c(report, message_text)
  }

  # --- duplicate coordinates (possible pseudo-replication) ---
  if (has_coordinates && isTRUE(config$validation$check_duplicate_coordinates)) {
    coordinate_key <- paste(data[[INTERNAL_X]], data[[INTERNAL_Y]])
    n_duplicate <- sum(duplicated(coordinate_key))
    if (n_duplicate > 0) {
      message_text <- sprintf(
        "%s data: %d observation(s) share coordinates with another observation.",
        role, n_duplicate)
      pipeline_warning(message_text,
                       "Check for duplicated points (pseudo-replication).")
      report <- c(report, message_text)
    }
  }

  list(data = data, report = report)
}


#' Map internal column names back to the user's column names for messages.
display_column_name <- function(columns, config) {
  lookup <- c(
    stats::setNames(config$variables$block_id, INTERNAL_BLOCK),
    stats::setNames(config$variables$fold_id, INTERNAL_FOLD),
    stats::setNames(config$variables$id, INTERNAL_ID)
  )
  ifelse(columns %in% names(lookup), lookup[columns], columns)
}


#' Verify the spatial fold structure supplied from ArcGIS Pro.
#'
#' Stops if:
#'   - the number of distinct FOLD_IDs differs from model$number_of_folds;
#'   - any BLOCK_ID occurs in more than one FOLD_ID (spatial-fold integrity);
#'   - a fold has fewer than validation$min_observations_per_fold rows;
#'   - a fold has fewer than validation$min_per_class_per_fold presences or
#'     absences (a fold with only one class makes ROC-AUC undefined).
#' Never modifies, repairs, or regenerates folds.
#'
#' @return Data frame summarising each fold.
validate_spatial_folds <- function(train_data, config) {
  block_name <- config$variables$block_id
  fold_name <- config$variables$fold_id
  expected_folds <- config$model$number_of_folds
  folds <- sort(unique(train_data[[INTERNAL_FOLD]]))

  if (length(folds) != expected_folds) {
    pipeline_stop(
      sprintf("Training data contain %d distinct %s value(s) (%s), but",
              length(folds), fold_name, paste(folds, collapse = ", ")),
      sprintf("model$number_of_folds is %d.", expected_folds),
      "Correct the fold assignment in ArcGIS Pro or the configuration."
    )
  }
  if (!identical(as.numeric(folds), as.numeric(seq_len(expected_folds)))) {
    pipeline_warning(sprintf(
      "%s values are %s rather than 1..%d. They are used as supplied.",
      fold_name, paste(folds, collapse = ", "), expected_folds))
  }

  # Spatial-fold integrity: every block in exactly one fold.
  folds_per_block <- unique(train_data[, c(INTERNAL_BLOCK, INTERNAL_FOLD)]) |>
    dplyr::group_by(dplyr::across(dplyr::all_of(INTERNAL_BLOCK))) |>
    dplyr::summarise(
      n_folds = dplyr::n(),
      fold_list = paste(sort(.data[[INTERNAL_FOLD]]), collapse = ", "),
      .groups = "drop"
    ) |>
    dplyr::filter(.data$n_folds > 1)
  if (nrow(folds_per_block) > 0) {
    offending <- sprintf(
      "  %s %s occurs in multiple %s values (%s).",
      block_name, folds_per_block[[INTERNAL_BLOCK]], fold_name,
      folds_per_block$fold_list)
    pipeline_stop(
      "Spatial-fold integrity error.",
      offending,
      "This violates the spatial cross-validation requirement: all",
      "observations in a spatial block must belong to the same fold, otherwise",
      "neighbouring (spatially autocorrelated) points fall on both sides of",
      "the training/validation split and performance is overestimated.",
      "Reassign whole blocks to folds in ArcGIS Pro and re-export."
    )
  }

  fold_summary <- train_data |>
    dplyr::group_by(dplyr::across(dplyr::all_of(INTERNAL_FOLD))) |>
    dplyr::summarise(
      N_OBSERVATIONS = dplyr::n(),
      N_BLOCKS = dplyr::n_distinct(.data[[INTERNAL_BLOCK]]),
      N_PRESENCE = sum(.data[[INTERNAL_RESPONSE]] == 1),
      N_ABSENCE = sum(.data[[INTERNAL_RESPONSE]] == 0),
      .groups = "drop"
    ) |>
    dplyr::mutate(PREVALENCE = .data$N_PRESENCE / .data$N_OBSERVATIONS) |>
    dplyr::rename(FOLD = dplyr::all_of(INTERNAL_FOLD))

  min_obs <- config$validation$min_observations_per_fold
  min_class <- max(1, config$validation$min_per_class_per_fold)
  problems <- character(0)
  for (i in seq_len(nrow(fold_summary))) {
    row <- fold_summary[i, ]
    if (row$N_OBSERVATIONS < min_obs) {
      problems <- c(problems, sprintf(
        "  %s %s has %d observations (minimum %d).",
        fold_name, row$FOLD, row$N_OBSERVATIONS, min_obs))
    }
    if (row$N_PRESENCE == 0 || row$N_ABSENCE == 0) {
      problems <- c(problems, sprintf(
        "  %s %s contains only one response class (%d presences, %d absences); ROC-AUC is undefined for this fold.",
        fold_name, row$FOLD, row$N_PRESENCE, row$N_ABSENCE))
    } else if (row$N_PRESENCE < min_class || row$N_ABSENCE < min_class) {
      problems <- c(problems, sprintf(
        "  %s %s has %d presences and %d absences (minimum %d of each).",
        fold_name, row$FOLD, row$N_PRESENCE, row$N_ABSENCE, min_class))
    }
  }
  if (length(problems) > 0) {
    pipeline_stop(
      "Spatial folds are not usable for cross-validation:",
      problems,
      "Revise the block-to-fold assignment in ArcGIS Pro (e.g. balance",
      "presences and absences across folds) or adjust the thresholds in",
      "the validation section of config.yaml."
    )
  }

  size_ratio <- max(fold_summary$N_OBSERVATIONS) /
    min(fold_summary$N_OBSERVATIONS)
  if (size_ratio > 3) {
    pipeline_warning(sprintf(
      "The largest fold has %.1f times as many observations as the smallest.",
      size_ratio), "Fold-level metrics will differ in precision.")
  }
  fold_summary
}


#' Warn about identifier or block overlap between training and test data.
check_train_test_separation <- function(train_data, test_data, config) {
  shared_ids <- intersect(train_data[[INTERNAL_ID]], test_data[[INTERNAL_ID]])
  if (length(shared_ids) > 0) {
    pipeline_stop(
      sprintf("%d %s value(s) appear in both the training and test data, e.g. %s.",
              length(shared_ids), config$variables$id,
              paste(utils::head(shared_ids, 10), collapse = ", ")),
      "The test set must be independent of the training set."
    )
  }
  if (INTERNAL_BLOCK %in% names(test_data)) {
    shared_blocks <- intersect(train_data[[INTERNAL_BLOCK]],
                               test_data[[INTERNAL_BLOCK]])
    if (length(shared_blocks) > 0) {
      pipeline_warning(sprintf(
        "%d %s value(s) occur in both training and test data (e.g. %s).",
        length(shared_blocks), config$variables$block_id,
        paste(utils::head(shared_blocks, 10), collapse = ", ")),
        "Test points inside training blocks are spatially close to training",
        "points; the test score may be optimistic.")
    }
  }
  invisible(TRUE)
}


# =============================================================================
# 6. Model definitions
# =============================================================================

#' Build the list of model specifications from the configuration.
#'
#' Each specification holds the model name, a label, the predictor keys, the
#' predictor column names, and a human-readable formula. All models share the
#' same MARS settings, folds, and threshold.
define_models <- function(config) {
  predictor_map <- config$variables$predictors
  specs <- lapply(config$models, function(model) {
    keys <- unlist(model$predictors)
    columns <- unname(unlist(predictor_map[keys]))
    list(
      name = model$name,
      label = if (is.null(model$label)) model$name else model$label,
      predictor_keys = keys,
      predictors = columns,
      formula_text = paste(config$variables$response, "~",
                           paste(columns, collapse = " + "))
    )
  })
  stats::setNames(specs, vapply(specs, `[[`, character(1), "name"))
}


# =============================================================================
# 7. Model fitting and prediction
# =============================================================================

#' Fit one binary MARS model with earth, complexity chosen by GCV pruning.
#'
#' Forward pass + backward pruning by GCV on the 0/1 response, followed by a
#' binomial (logit) GLM on the retained basis functions (Part B.7).
fit_mars_gcv <- function(train_data, predictors, config) {
  mars <- config$model$mars
  degree <- as.integer(mars$degree)
  # earth's own defaults, stated explicitly so they are recorded in the output.
  penalty <- mars$penalty
  if (is.null(penalty)) {
    penalty <- if (degree > 1) 3 else 2
  }
  nk <- mars$nk
  if (is.null(nk)) {
    nk <- min(200, max(20, 2 * length(predictors))) + 1
  }

  model_formula <- stats::reformulate(predictors, response = INTERNAL_RESPONSE)
  model_data <- train_data[, c(INTERNAL_RESPONSE, predictors), drop = FALSE]

  earth::earth(
    formula = model_formula,
    data = model_data,
    glm = list(family = stats::binomial(link = "logit"),
               maxit = mars$glm_maxit),
    degree = degree,
    nprune = mars$nprune,
    penalty = penalty,
    nk = nk,
    thresh = mars$thresh,
    minspan = mars$minspan,
    endspan = mars$endspan,
    pmethod = mars$pmethod
  )
}


#' Tune degree and nprune by nested spatial CV with caret, then refit.
#'
#' Inner resamples are leave-one-fold-out over the folds present in
#' `train_data`, so they respect the ArcGIS Pro spatial blocks. The outer
#' validation fold and the test set are never passed to this function.
tune_mars_spatial_cv <- function(train_data, predictors, config) {
  mars <- config$model$mars
  tuning <- config$model$tuning
  inner_folds <- sort(unique(train_data[[INTERNAL_FOLD]]))
  if (length(inner_folds) < 2) {
    pipeline_stop("Nested spatial tuning needs at least two inner folds.")
  }
  fold_vector <- train_data[[INTERNAL_FOLD]]
  index_in <- lapply(inner_folds, function(f) which(fold_vector != f))
  index_out <- lapply(inner_folds, function(f) which(fold_vector == f))
  names(index_in) <- names(index_out) <- paste0("Fold", inner_folds)

  degree_grid <- if (length(predictors) == 1) 1 else tuning$degree_grid
  tune_grid <- expand.grid(nprune = sort(unique(tuning$nprune_grid)),
                           degree = sort(unique(degree_grid)))

  control <- caret::trainControl(
    method = "cv",
    index = index_in,
    indexOut = index_out,
    classProbs = TRUE,
    summaryFunction = caret::twoClassSummary,
    savePredictions = "none",
    returnData = FALSE,
    allowParallel = FALSE
  )

  set.seed(config$model$seed)
  tuned <- caret::train(
    x = train_data[, predictors, drop = FALSE],
    y = train_data[[INTERNAL_CLASS]],
    method = "earth",
    metric = "ROC",
    maximize = TRUE,
    tuneGrid = tune_grid,
    trControl = control,
    glm = list(family = stats::binomial(link = "logit"),
               maxit = mars$glm_maxit),
    thresh = mars$thresh,
    minspan = mars$minspan,
    endspan = mars$endspan,
    pmethod = mars$pmethod
  )

  list(
    fit = tuned$finalModel,
    best_tune = tuned$bestTune,
    tuning_results = tuned$results
  )
}


#' Fit one model specification on `train_data` using the configured method.
#'
#' @return list(fit, warnings, best_tune, tuning_results)
fit_mars_model <- function(train_data, spec, config) {
  set.seed(config$model$seed)
  if (config$model$tuning$method == "spatial_cv") {
    result <- with_captured_warnings(
      tune_mars_spatial_cv(train_data, spec$predictors, config))
    out <- result$value
  } else {
    result <- with_captured_warnings(
      fit_mars_gcv(train_data, spec$predictors, config))
    out <- list(fit = result$value, best_tune = NULL, tuning_results = NULL)
  }
  out$warnings <- result$warnings
  out
}


#' Predicted probability of salmonberry presence, P(Y = 1 | X).
#'
#' type = "response" returns probabilities from earth's binomial GLM. For a
#' factor response earth models the second level ("present"); for the 0/1
#' response it models 1. Both give P(present).
predict_probability <- function(fit, new_data, predictors) {
  probabilities <- stats::predict(
    fit, newdata = new_data[, predictors, drop = FALSE], type = "response")
  probabilities <- as.numeric(probabilities[, 1])
  if (any(!is.finite(probabilities)) ||
      any(probabilities < 0 | probabilities > 1)) {
    pipeline_stop("MARS returned probabilities outside [0, 1]; check the fit.")
  }
  probabilities
}


#' Hard classification: 1 if probability >= threshold, else 0.
classify_probability <- function(probabilities, threshold) {
  as.integer(probabilities >= threshold)
}


#' Summaries of a fitted earth model for the diagnostics files.
describe_fit <- function(fit) {
  glm_fit <- fit$glm.list[[1]]
  list(
    n_terms = length(fit$selected.terms),
    gcv = fit$gcv,
    rsq = fit$rsq,
    glm_converged = if (is.null(glm_fit)) NA else isTRUE(glm_fit$converged),
    glm_deviance = if (is.null(glm_fit)) NA_real_ else glm_fit$deviance
  )
}


#' Knot locations used by a fitted earth model for one predictor.
extract_knots <- function(fit, predictor) {
  if (!(predictor %in% colnames(fit$dirs))) {
    return(numeric(0))
  }
  selected <- fit$selected.terms
  directions <- fit$dirs[selected, predictor]
  cuts <- fit$cuts[selected, predictor]
  # dirs = +1 / -1 mark hinge terms; 2 marks a linear (no-knot) term.
  sort(unique(cuts[directions %in% c(-1, 1)]))
}


#' Retained basis functions and their logistic-GLM coefficients.
extract_model_terms <- function(fit, model_name) {
  coefficients <- fit$glm.coefficients
  if (is.null(coefficients)) {
    coefficients <- fit$coefficients
  }
  data.frame(
    MODEL = model_name,
    TERM = rownames(coefficients),
    COEFFICIENT_LOGIT_SCALE = as.numeric(coefficients[, 1]),
    stringsAsFactors = FALSE
  )
}


# =============================================================================
# 8. Performance metrics
# =============================================================================

#' ROC-AUC from predicted probabilities (not hard classes).
#'
#' direction = "<" fixes the orientation (higher probability = presence), so a
#' model worse than random is reported as AUC < 0.5 instead of being silently
#' flipped by pROC's automatic direction detection.
compute_auc <- function(observed, probabilities) {
  if (length(unique(observed)) < 2) {
    return(NA_real_)
  }
  roc_curve <- pROC::roc(response = observed, predictor = probabilities,
                         levels = c(0, 1), direction = "<", quiet = TRUE)
  as.numeric(pROC::auc(roc_curve))
}


#' Threshold-based and threshold-free classification metrics.
#'
#' Sensitivity (recall) = TP / (TP + FN)   share of presences detected
#' Specificity          = TN / (TN + FP)   share of absences correctly rejected
#' Precision            = TP / (TP + FP)   share of predicted presences that are
#'                                         real (NA if no presences predicted)
#' F1                   = 2TP / (2TP + FP + FN)  (harmonic mean of precision
#'                                         and sensitivity)
#' Balanced accuracy    = (sensitivity + specificity) / 2
#' Accuracy             = (TP + TN) / N
#' AUC                  from probabilities; threshold-free
compute_classification_metrics <- function(observed, probabilities, threshold) {
  predicted <- classify_probability(probabilities, threshold)
  tp <- sum(predicted == 1 & observed == 1)
  fp <- sum(predicted == 1 & observed == 0)
  tn <- sum(predicted == 0 & observed == 0)
  fn <- sum(predicted == 0 & observed == 1)
  safe_ratio <- function(numerator, denominator) {
    if (denominator == 0) NA_real_ else numerator / denominator
  }
  sensitivity <- safe_ratio(tp, tp + fn)
  specificity <- safe_ratio(tn, tn + fp)
  data.frame(
    N = length(observed),
    N_PRESENCE = sum(observed == 1),
    N_ABSENCE = sum(observed == 0),
    AUC = compute_auc(observed, probabilities),
    SENSITIVITY = sensitivity,
    SPECIFICITY = specificity,
    PRECISION = safe_ratio(tp, tp + fp),
    F1 = safe_ratio(2 * tp, 2 * tp + fp + fn),
    BALANCED_ACCURACY = (sensitivity + specificity) / 2,
    ACCURACY = safe_ratio(tp + tn, length(observed)),
    TP = tp, FP = fp, TN = tn, FN = fn,
    THRESHOLD = threshold
  )
}


# =============================================================================
# 9. Spatial cross-validation
# =============================================================================

#' Spatial K-fold cross-validation using the supplied FOLD_ID.
#'
#' For each model and each fold k: train on all folds except k, predict fold k.
#' The same folds are used for every model, and the validation fold is never
#' seen by model fitting (including caret tuning when enabled).
#'
#' @return list(metrics, predictions, fit_diagnostics, tuning, fold_models)
run_spatial_cv <- function(train_data, model_specs, config) {
  threshold <- config$model$classification_threshold
  folds <- sort(unique(train_data[[INTERNAL_FOLD]]))
  metrics <- list()
  predictions <- list()
  diagnostics <- list()
  tuning <- list()
  fold_models <- list()

  for (spec in model_specs) {
    for (fold in folds) {
      log_message(sprintf("Spatial CV: %s, fold %s", spec$name, fold))
      in_validation <- train_data[[INTERNAL_FOLD]] == fold
      fold_train <- train_data[!in_validation, , drop = FALSE]
      fold_valid <- train_data[in_validation, , drop = FALSE]

      fitted <- fit_mars_model(fold_train, spec, config)
      probabilities <- predict_probability(fitted$fit, fold_valid,
                                           spec$predictors)
      observed <- fold_valid[[INTERNAL_RESPONSE]]
      key <- paste(spec$name, fold, sep = "__")

      fit_summary <- describe_fit(fitted$fit)
      metrics[[key]] <- cbind(
        data.frame(MODEL = spec$name, MODEL_LABEL = spec$label,
                   FORMULA = spec$formula_text, FOLD = fold),
        compute_classification_metrics(observed, probabilities, threshold),
        data.frame(N_TERMS = fit_summary$n_terms)
      )
      predictions[[key]] <- build_prediction_table(
        fold_valid, spec, probabilities, threshold, fold = fold)
      diagnostics[[key]] <- data.frame(
        MODEL = spec$name, FOLD = fold,
        N_TRAIN = nrow(fold_train), N_VALIDATION = nrow(fold_valid),
        N_TERMS = fit_summary$n_terms, GCV = fit_summary$gcv,
        RSQ = fit_summary$rsq, GLM_CONVERGED = fit_summary$glm_converged,
        WARNINGS = paste(fitted$warnings, collapse = " | ")
      )
      if (!is.null(fitted$best_tune)) {
        tuning[[key]] <- data.frame(MODEL = spec$name, FOLD = fold,
                                    fitted$best_tune)
      }
      if (isTRUE(config$output$save_fold_models)) {
        fold_models[[key]] <- fitted$fit
      }
    }
  }

  list(
    metrics = dplyr::bind_rows(metrics),
    predictions = dplyr::bind_rows(predictions),
    fit_diagnostics = dplyr::bind_rows(diagnostics),
    tuning = if (length(tuning) > 0) dplyr::bind_rows(tuning) else NULL,
    fold_models = fold_models
  )
}


#' Observation-level prediction table (probabilities AND hard classes).
build_prediction_table <- function(data, spec, probabilities, threshold,
                                   fold = NA) {
  table <- data.frame(
    MODEL = spec$name,
    FOLD = fold,
    ID = data[[INTERNAL_ID]],
    stringsAsFactors = FALSE
  )
  if (INTERNAL_X %in% names(data)) {
    table$X <- data[[INTERNAL_X]]
    table$Y <- data[[INTERNAL_Y]]
  }
  if (INTERNAL_BLOCK %in% names(data)) {
    table$BLOCK_ID <- data[[INTERNAL_BLOCK]]
  }
  table$observed <- data[[INTERNAL_RESPONSE]]
  table$predicted_probability <- probabilities
  table$predicted_class <- classify_probability(probabilities, threshold)
  table$threshold <- threshold
  table
}


# =============================================================================
# 10. Model comparison
# =============================================================================

#' Summarise fold-level CV metrics for each model.
#'
#' Models are listed in the configured order, not ranked. Pooled_OOF_AUC is
#' the AUC of all out-of-fold predictions combined; it mixes folds whose
#' predictions may be calibrated differently and is reported only as a
#' complement to the fold-wise mean.
compare_models <- function(cv_results, model_specs) {
  mean_na <- function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
  sd_na <- function(x) if (sum(!is.na(x)) < 2) NA_real_ else
    stats::sd(x, na.rm = TRUE)

  summary_table <- cv_results$metrics |>
    dplyr::group_by(dplyr::across(dplyr::all_of(c("MODEL", "MODEL_LABEL",
                                                  "FORMULA")))) |>
    dplyr::summarise(
      N_Folds = dplyr::n(),
      Mean_AUC = mean_na(.data$AUC),
      SD_AUC = sd_na(.data$AUC),
      Min_AUC = min(.data$AUC, na.rm = TRUE),
      Max_AUC = max(.data$AUC, na.rm = TRUE),
      Mean_Sensitivity = mean_na(.data$SENSITIVITY),
      SD_Sensitivity = sd_na(.data$SENSITIVITY),
      Mean_Specificity = mean_na(.data$SPECIFICITY),
      SD_Specificity = sd_na(.data$SPECIFICITY),
      Mean_Precision = mean_na(.data$PRECISION),
      N_Folds_Precision_Undefined = sum(is.na(.data$PRECISION)),
      Mean_F1 = mean_na(.data$F1),
      Mean_Balanced_Accuracy = mean_na(.data$BALANCED_ACCURACY),
      SD_Balanced_Accuracy = sd_na(.data$BALANCED_ACCURACY),
      Mean_Accuracy = mean_na(.data$ACCURACY),
      Mean_N_Terms = mean_na(.data$N_TERMS),
      .groups = "drop"
    )

  pooled <- cv_results$predictions |>
    dplyr::group_by(dplyr::across(dplyr::all_of("MODEL"))) |>
    dplyr::summarise(
      Pooled_OOF_AUC = compute_auc(.data$observed,
                                   .data$predicted_probability),
      .groups = "drop")

  summary_table |>
    dplyr::left_join(pooled, by = "MODEL") |>
    dplyr::rename(Model = "MODEL", Model_Label = "MODEL_LABEL",
                  Formula = "FORMULA") |>
    dplyr::mutate(Model = factor(.data$Model, levels = names(model_specs))) |>
    dplyr::arrange(.data$Model) |>
    dplyr::mutate(Model = as.character(.data$Model)) |>
    as.data.frame()
}


#' Paired, per-fold Delta AUC for each configured comparison.
#'
#' Delta AUC = AUC(model_a) - AUC(model_b), computed within each spatial fold.
#' The summary is descriptive (mean, SD, min, max, fold counts). No p-value is
#' produced: fold AUCs are not independent (training sets overlap by
#' (K-2)/(K-1) of the data), K is small, and spatial folds differ in difficulty,
#' so standard tests on fold-level differences are not valid.
#'
#' @return list(by_fold, summary)
compute_delta_auc <- function(cv_metrics, comparisons) {
  if (length(comparisons) == 0) {
    return(list(by_fold = NULL, summary = NULL))
  }
  auc_wide <- cv_metrics |>
    dplyr::select("MODEL", "FOLD", "AUC") |>
    tidyr::pivot_wider(names_from = "MODEL", values_from = "AUC")

  by_fold <- list()
  summary_rows <- list()
  for (comparison in comparisons) {
    auc_a <- auc_wide[[comparison$model_a]]
    auc_b <- auc_wide[[comparison$model_b]]
    delta <- auc_a - auc_b
    by_fold[[comparison$name]] <- data.frame(
      COMPARISON = comparison$name,
      MODEL_A = comparison$model_a,
      MODEL_B = comparison$model_b,
      FOLD = auc_wide$FOLD,
      AUC_A = auc_a,
      AUC_B = auc_b,
      DELTA_AUC = delta
    )
    valid <- delta[!is.na(delta)]
    n_a_higher <- sum(valid > 0)
    n_b_higher <- sum(valid < 0)
    mean_delta <- if (length(valid)) mean(valid) else NA_real_
    sd_delta <- if (length(valid) > 1) stats::sd(valid) else NA_real_
    min_delta <- if (length(valid)) min(valid) else NA_real_
    max_delta <- if (length(valid)) max(valid) else NA_real_
    summary_rows[[comparison$name]] <- data.frame(
      COMPARISON = comparison$name,
      DESCRIPTION = if (is.null(comparison$description)) "" else
        comparison$description,
      PRIMARY = isTRUE(comparison$primary),
      MODEL_A = comparison$model_a,
      MODEL_B = comparison$model_b,
      N_FOLDS = length(valid),
      MEAN_DELTA_AUC = mean_delta,
      SD_DELTA_AUC = sd_delta,
      MIN_DELTA_AUC = min_delta,
      MAX_DELTA_AUC = max_delta,
      N_FOLDS_A_HIGHER = n_a_higher,
      N_FOLDS_B_HIGHER = n_b_higher,
      N_FOLDS_TIED = sum(valid == 0),
      NOTE = sprintf(
        paste0("Mean Delta AUC (%s - %s) = %s (SD %s; range %s to %s); ",
               "%s had the higher AUC in %d of %d folds. Descriptive only: ",
               "fold-level differences are not independent and no ",
               "significance test is implied."),
        comparison$model_a, comparison$model_b,
        format_number(mean_delta, signed = TRUE), format_number(sd_delta),
        format_number(min_delta, signed = TRUE),
        format_number(max_delta, signed = TRUE),
        comparison$model_a, n_a_higher, length(valid))
    )
  }
  list(by_fold = dplyr::bind_rows(by_fold),
       summary = dplyr::bind_rows(summary_rows))
}


#' Choose the model to evaluate on the test set, using training-data CV only.
#'
#' If final_model$selected_model is set (recommended: decide before seeing the
#' CV results), that model is used. Otherwise the model with the highest mean
#' spatial-CV AUC is chosen, ties broken by fewer predictors and then by the
#' configured model order. The test data play no part in this choice.
select_final_model <- function(model_comparison, model_specs, config) {
  preselected <- config$final_model$selected_model
  if (!is.null(preselected)) {
    return(list(
      model = preselected,
      justification = sprintf(
        "Model '%s' was specified in final_model$selected_model.",
        preselected)
    ))
  }
  n_predictors <- vapply(model_specs[model_comparison$Model],
                         function(s) length(s$predictors), integer(1))
  order_index <- order(-model_comparison$Mean_AUC, n_predictors,
                       seq_len(nrow(model_comparison)), na.last = TRUE)
  chosen <- model_comparison$Model[order_index[1]]
  list(
    model = chosen,
    justification = sprintf(
      paste0("Model '%s' had the highest mean spatial-CV AUC (%s). Selection ",
             "rule: highest mean CV AUC; ties broken by fewer predictors. ",
             "Differences in mean AUC among models may be smaller than ",
             "fold-to-fold variation (see model_comparison.csv and ",
             "delta_auc.csv)."),
      chosen,
      format_number(model_comparison$Mean_AUC[order_index[1]]))
  )
}


# =============================================================================
# 11. Final models, variable importance
# =============================================================================

#' Refit every model on ALL training data.
#'
#' All five are refit so their variable importance and response curves can be
#' reported; only the selected model is evaluated on the test set (unless
#' evaluation$evaluate_all_models_on_test is true).
fit_final_models <- function(train_data, model_specs, config) {
  lapply(model_specs, function(spec) {
    log_message(sprintf("Final fit on all training data: %s", spec$name))
    fitted <- fit_mars_model(train_data, spec, config)
    fitted$spec <- spec
    fitted
  })
}


#' Variable importance for each final model (earth::evimp).
#'
#' evimp() tracks, over the backward-pruning sequence of nested subsets, how
#' often each predictor appears (nsubsets) and how much GCV / RSS improves in
#' subsets containing it. Values are scaled so the most important predictor
#' has 100 (sqrt. = FALSE: linear scaling).
#'
#' LIMITATIONS: importance is a property of this fitted model and these data,
#' not of the ecological system. It measures predictive association, NOT
#' causation. Correlated predictors (e.g. slope and moundness, both derived
#' from the same DTM) can share or trade importance, so a low value does not
#' mean a predictor is unrelated to salmonberry. In one-predictor models the
#' only predictor trivially has importance 100 (or 0 if unused). Importance
#' measures in-sample fit improvement and should be read alongside the
#' spatially cross-validated AUC.
extract_variable_importance <- function(final_models, config) {
  metric_column <- c(gcv = "IMPORTANCE_GCV", nsubsets = "IMPORTANCE_NSUBSETS",
                     rss = "IMPORTANCE_RSS")[[config$variable_importance$metric]]
  tables <- lapply(final_models, function(final) {
    importance <- unclass(earth::evimp(final$fit, trim = FALSE, sqrt. = FALSE))
    predictors <- sub("-unused$", "", rownames(importance))
    table <- data.frame(
      MODEL = final$spec$name,
      PREDICTOR = predictors,
      USED_IN_MODEL = importance[, "used"] == 1,
      IMPORTANCE_NSUBSETS = importance[, "nsubsets"],
      IMPORTANCE_GCV = importance[, "gcv"],
      IMPORTANCE_RSS = importance[, "rss"],
      stringsAsFactors = FALSE
    )
    table$IMPORTANCE <- table[[metric_column]]
    table$IMPORTANCE_METRIC <- config$variable_importance$metric
    rownames(table) <- NULL
    table[, c("MODEL", "PREDICTOR", "IMPORTANCE", "IMPORTANCE_METRIC",
              "USED_IN_MODEL", "IMPORTANCE_NSUBSETS", "IMPORTANCE_GCV",
              "IMPORTANCE_RSS")]
  })
  dplyr::bind_rows(tables)
}


# =============================================================================
# 12. Final test-set evaluation
# =============================================================================

#' Evaluate the selected model(s) on the untouched test set.
#'
#' The test CSV is read HERE for the first time in a full run, after all
#' modeling decisions have been fixed. Nothing computed from the test data is
#' fed back into any model, threshold, or selection.
#'
#' @return list(metrics, predictions, report)
evaluate_test_data <- function(final_models, selection, train_data, config) {
  log_message("Reading the held-out test data (first and only use).")
  raw_test <- load_data(config$data$test_path_resolved, config)
  validated <- validate_data(raw_test, config, role = "Test")
  test_data <- validated$data
  check_train_test_separation(train_data, test_data, config)

  threshold <- config$model$classification_threshold
  models_to_test <- if (isTRUE(config$evaluation$evaluate_all_models_on_test))
    names(final_models) else selection$model

  metrics <- list()
  predictions <- list()
  for (model_name in models_to_test) {
    final <- final_models[[model_name]]
    probabilities <- predict_probability(final$fit, test_data,
                                         final$spec$predictors)
    observed <- test_data[[INTERNAL_RESPONSE]]
    roc_curve <- pROC::roc(response = observed, predictor = probabilities,
                           levels = c(0, 1), direction = "<", quiet = TRUE)
    ci_values <- tryCatch(as.numeric(pROC::ci.auc(roc_curve, method = "delong")),
                          error = function(e) rep(NA_real_, 3))
    auc_ci <- ci_values[c(1, 3)]
    metrics[[model_name]] <- cbind(
      data.frame(
        MODEL = model_name,
        MODEL_LABEL = final$spec$label,
        FORMULA = final$spec$formula_text,
        EVALUATION_ROLE = if (model_name == selection$model)
          "selected_model" else "secondary_descriptive_only"
      ),
      compute_classification_metrics(observed, probabilities, threshold),
      data.frame(AUC_CI95_LOWER_DELONG = auc_ci[1],
                 AUC_CI95_UPPER_DELONG = auc_ci[2])
    )
    predictions[[model_name]] <- build_prediction_table(
      test_data, final$spec, probabilities, threshold)
  }
  list(
    metrics = dplyr::bind_rows(metrics),
    predictions = dplyr::bind_rows(predictions) |> dplyr::select(-"FOLD"),
    report = validated$report
  )
}


# =============================================================================
# 13. Figures
# =============================================================================

#' Shared ggplot theme for publication figures.
publication_theme <- function() {
  ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major = ggplot2::element_line(colour = "grey90",
                                               linewidth = 0.3),
      legend.position = "bottom",
      plot.title = ggplot2::element_text(face = "bold"),
      plot.caption = ggplot2::element_text(colour = "grey35", hjust = 0)
    )
}


#' Figure A: spatially cross-validated AUC by model.
plot_cv_auc <- function(cv_metrics, model_specs) {
  labels <- vapply(model_specs, `[[`, character(1), "label")
  plot_data <- cv_metrics |>
    dplyr::mutate(MODEL_LABEL = factor(.data$MODEL_LABEL, levels = labels))
  summary_data <- plot_data |>
    dplyr::group_by(dplyr::across(dplyr::all_of("MODEL_LABEL"))) |>
    dplyr::summarise(mean_auc = mean(.data$AUC, na.rm = TRUE),
                     sd_auc = stats::sd(.data$AUC, na.rm = TRUE),
                     .groups = "drop")

  ggplot2::ggplot(plot_data, ggplot2::aes(x = .data$MODEL_LABEL,
                                          y = .data$AUC)) +
    ggplot2::geom_hline(yintercept = 0.5, linetype = "dashed",
                        colour = "grey50") +
    ggplot2::geom_line(ggplot2::aes(group = .data$FOLD), colour = "grey80",
                       linewidth = 0.4) +
    ggplot2::geom_point(colour = "grey55", size = 2) +
    ggplot2::geom_pointrange(
      data = summary_data,
      ggplot2::aes(x = .data$MODEL_LABEL, y = .data$mean_auc,
                   ymin = .data$mean_auc - .data$sd_auc,
                   ymax = .data$mean_auc + .data$sd_auc),
      inherit.aes = FALSE, colour = SERIES_COLOURS[1], size = 0.6,
      linewidth = 0.9,
      position = ggplot2::position_nudge(x = 0.18)) +
    ggplot2::coord_cartesian(ylim = c(min(0.4, min(plot_data$AUC,
                                                   na.rm = TRUE)), 1)) +
    ggplot2::labs(
      title = "Spatially cross-validated discrimination",
      x = NULL, y = "ROC-AUC (validation fold)",
      caption = paste0(
        "Grey points: individual spatial folds (lines join the same fold ",
        "across models).\nBlue: mean ± SD across folds. Dashed line: ",
        "AUC = 0.5 (no discrimination).")) +
    publication_theme() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 20,
                                                       hjust = 1))
}


#' Supplementary figure: per-fold Delta AUC for each comparison.
plot_delta_auc <- function(delta_by_fold) {
  ggplot2::ggplot(delta_by_fold,
                  ggplot2::aes(x = .data$COMPARISON, y = .data$DELTA_AUC)) +
    ggplot2::geom_hline(yintercept = 0, colour = "grey50") +
    ggplot2::geom_point(colour = SERIES_COLOURS[1], size = 2.5) +
    ggplot2::stat_summary(fun = mean, fun.min = mean, fun.max = mean,
                          geom = "crossbar", width = 0.35,
                          colour = "grey20", linewidth = 0.4) +
    ggplot2::labs(
      title = "Paired per-fold difference in AUC",
      x = NULL, y = "ΔAUC (model A − model B)",
      caption = paste0("Points: spatial folds. Bar: mean. Positive values ",
                       "favour model A in that fold; descriptive only.")) +
    publication_theme()
}


#' Figures B and C: fitted probability of occurrence vs one predictor.
#'
#' For every final model containing the focal predictor, predictions are made
#' over the observed range of that predictor with all other predictors held
#' at their training-data medians (a conditional, not averaged, response).
#' Dashed vertical lines mark MARS knots on the focal predictor. Rugs show
#' training observations (top: presences; bottom: absences). These curves
#' describe fitted associations and must not be interpreted causally.
plot_response_curve <- function(focal_key, final_models, train_data, config) {
  focal_column <- config$variables$predictors[[focal_key]]
  if (is.null(focal_column)) {
    pipeline_warning(sprintf("Response curve skipped: unknown predictor '%s'.",
                             focal_key))
    return(NULL)
  }
  relevant <- Filter(function(m) focal_column %in% m$spec$predictors,
                     final_models)
  if (length(relevant) == 0) {
    return(NULL)
  }

  n_points <- config$plots$response_curve_points
  focal_range <- range(train_data[[focal_column]])
  grid_values <- seq(focal_range[1], focal_range[2], length.out = n_points)
  all_predictors <- all_model_predictor_columns(config)
  medians <- vapply(all_predictors,
                    function(col) stats::median(train_data[[col]]),
                    numeric(1))

  curve_rows <- list()
  knot_rows <- list()
  for (final in relevant) {
    new_data <- as.data.frame(as.list(medians))[rep(1, n_points), ,
                                                drop = FALSE]
    new_data[[focal_column]] <- grid_values
    curve_rows[[final$spec$name]] <- data.frame(
      MODEL_LABEL = final$spec$label,
      value = grid_values,
      probability = predict_probability(final$fit, new_data,
                                        final$spec$predictors))
    knots <- extract_knots(final$fit, focal_column)
    if (length(knots) > 0) {
      knot_rows[[final$spec$name]] <- data.frame(MODEL_LABEL = final$spec$label,
                                                 knot = knots)
    }
  }
  model_labels <- vapply(relevant, function(m) m$spec$label, character(1))
  curves <- dplyr::bind_rows(curve_rows) |>
    dplyr::mutate(MODEL_LABEL = factor(.data$MODEL_LABEL,
                                       levels = model_labels))
  knot_data <- dplyr::bind_rows(knot_rows)
  observations <- data.frame(
    value = train_data[[focal_column]],
    present = train_data[[INTERNAL_RESPONSE]] == 1)
  colours <- stats::setNames(
    SERIES_COLOURS[(seq_along(model_labels) - 1) %% length(SERIES_COLOURS) + 1],
    model_labels)

  plot <- ggplot2::ggplot() +
    ggplot2::geom_rug(data = observations[observations$present, ],
                      ggplot2::aes(x = .data$value), sides = "t",
                      alpha = 0.3, colour = "grey30") +
    ggplot2::geom_rug(data = observations[!observations$present, ],
                      ggplot2::aes(x = .data$value), sides = "b",
                      alpha = 0.3, colour = "grey30")
  if (nrow(knot_data) > 0) {
    knot_data$MODEL_LABEL <- factor(knot_data$MODEL_LABEL,
                                    levels = model_labels)
    plot <- plot +
      ggplot2::geom_vline(data = knot_data,
                          ggplot2::aes(xintercept = .data$knot,
                                       colour = .data$MODEL_LABEL),
                          linetype = "dashed", linewidth = 0.4, alpha = 0.7,
                          show.legend = FALSE)
  }
  plot +
    ggplot2::geom_line(data = curves,
                       ggplot2::aes(x = .data$value, y = .data$probability,
                                    colour = .data$MODEL_LABEL),
                       linewidth = 0.9) +
    ggplot2::scale_colour_manual(values = colours, name = NULL) +
    ggplot2::coord_cartesian(ylim = c(0, 1)) +
    ggplot2::labs(
      title = sprintf("Fitted probability of salmonberry vs %s", focal_column),
      x = focal_column,
      y = "Predicted probability of occurrence",
      caption = paste0(
        "Final models fit to all training data; other predictors held at ",
        "training medians.\nDashed lines: MARS knots. Rugs: training ",
        "presences (top) and absences (bottom).\nCurves show fitted ",
        "predictive associations, not causal effects.")) +
    publication_theme()
}


#' Build all figures. Returns a named list of ggplot objects.
create_plots <- function(cv_results, delta, final_models, train_data,
                         model_specs, config) {
  plots <- list(cv_auc_by_model = plot_cv_auc(cv_results$metrics, model_specs))
  if (!is.null(delta$by_fold)) {
    plots$delta_auc_by_fold <- plot_delta_auc(delta$by_fold)
  }
  for (key in config$plots$response_curve_predictors) {
    response_plot <- plot_response_curve(key, final_models, train_data, config)
    if (!is.null(response_plot)) {
      plots[[paste0("response_curve_", key)]] <- response_plot
    }
  }
  plots
}


# =============================================================================
# 14. Output
# =============================================================================

#' Create the output directory tree.
setup_output_directories <- function(config) {
  root <- config$output$directory_resolved
  subdirectories <- c("models", "predictions", "metrics",
                      "variable_importance", "plots", "diagnostics",
                      "session_info")
  paths <- stats::setNames(file.path(root, subdirectories), subdirectories)
  if (dir.exists(root) && length(list.files(root, recursive = TRUE)) > 0) {
    pipeline_warning(sprintf(
      "Output directory %s is not empty; files with the same names will be overwritten.",
      root))
  }
  for (path in paths) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
  c(root = root, paths)
}


#' Write a data frame as CSV, skipping NULL inputs.
write_table <- function(table, directory, file_name) {
  if (is.null(table)) {
    return(invisible(NULL))
  }
  readr::write_csv(table, file.path(directory, file_name), na = "NA")
  invisible(file.path(directory, file_name))
}


#' Save pre-modeling diagnostics (data validation and fold structure).
save_diagnostics <- function(train_validation, fold_summary, dirs) {
  writeLines(train_validation$report,
             file.path(dirs[["diagnostics"]], "data_validation_report.txt"))
  write_table(fold_summary, dirs[["diagnostics"]], "fold_summary.csv")
}


#' Save everything present in `results` to the output tree.
#'
#' Safe to call more than once: CV results are written before the test data
#' are read, so they persist even if a later stage fails.
save_results <- function(results, dirs, config) {
  if (!is.null(results$cv)) {
    write_table(results$cv$metrics, dirs[["metrics"]], "cv_metrics.csv")
    write_table(results$cv$predictions, dirs[["predictions"]],
                "fold_predictions.csv")
    write_table(results$cv$fit_diagnostics, dirs[["diagnostics"]],
                "cv_fit_diagnostics.csv")
    write_table(results$cv$tuning, dirs[["diagnostics"]],
                "cv_tuning_selected.csv")
    if (length(results$cv$fold_models) > 0) {
      saveRDS(results$cv$fold_models,
              file.path(dirs[["models"]], "cv_fold_models.rds"))
    }
  }
  write_table(results$model_comparison, dirs[["metrics"]],
              "model_comparison.csv")
  write_table(results$delta$by_fold, dirs[["metrics"]], "delta_auc.csv")
  write_table(results$delta$summary, dirs[["metrics"]],
              "delta_auc_summary.csv")
  if (!is.null(results$selection)) {
    writeLines(c(sprintf("Selected model: %s", results$selection$model),
                 results$selection$justification,
                 "The test data were not used for this selection."),
               file.path(dirs[["diagnostics"]], "model_selection.txt"))
  }

  if (!is.null(results$final_models)) {
    for (final in results$final_models) {
      saveRDS(final$fit, file.path(dirs[["models"]],
                                   sprintf("final_%s.rds", final$spec$name)))
    }
    terms <- dplyr::bind_rows(lapply(results$final_models, function(final) {
      extract_model_terms(final$fit, final$spec$name)
    }))
    write_table(terms, dirs[["models"]], "final_model_terms.csv")
    summaries <- unlist(lapply(results$final_models, function(final) {
      c(sprintf("===== %s: %s =====", final$spec$name,
                final$spec$formula_text),
        utils::capture.output(summary(final$fit)),
        if (length(final$warnings)) paste("Warnings:", final$warnings),
        "")
    }))
    writeLines(summaries, file.path(dirs[["models"]],
                                    "final_model_summaries.txt"))
    final_tuning <- dplyr::bind_rows(lapply(results$final_models,
                                            function(final) {
      if (is.null(final$best_tune)) NULL else
        data.frame(MODEL = final$spec$name, final$best_tune)
    }))
    if (nrow(final_tuning) > 0) {
      write_table(final_tuning, dirs[["diagnostics"]],
                  "final_tuning_selected.csv")
    }
  }
  write_table(results$variable_importance, dirs[["variable_importance"]],
              "variable_importance.csv")

  if (!is.null(results$test)) {
    write_table(results$test$metrics, dirs[["metrics"]], "test_metrics.csv")
    write_table(results$test$predictions, dirs[["predictions"]],
                "test_predictions.csv")
    writeLines(results$test$report,
               file.path(dirs[["diagnostics"]], "test_data_validation.txt"))
  }

  for (plot_name in names(results$plots)) {
    for (format in config$plots$formats) {
      ggplot2::ggsave(
        filename = file.path(dirs[["plots"]],
                             sprintf("%s.%s", plot_name, format)),
        plot = results$plots[[plot_name]],
        width = config$plots$width_in, height = config$plots$height_in,
        dpi = config$plots$dpi, units = "in")
    }
  }
  invisible(TRUE)
}


#' Record configuration, input checksums, and the software environment.
save_session_info <- function(config, dirs, stage) {
  session_dir <- dirs[["session_info"]]
  file.copy(config$meta$config_path,
            file.path(session_dir, "config_original.yaml"), overwrite = TRUE)
  used <- config
  used$meta$raw_config <- NULL
  yaml::write_yaml(used, file.path(session_dir, "config_used.yaml"))

  input_files <- c(train = config$data$train_path_resolved)
  if (stage == "complete" && isTRUE(config$evaluation$run_test_evaluation)) {
    input_files <- c(input_files, test = config$data$test_path_resolved)
  }
  package_versions <- vapply(REQUIRED_PACKAGES, function(p) {
    as.character(utils::packageVersion(p))
  }, character(1))
  manifest <- list(
    run_completed = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
    stage = stage,
    r_version = R.version.string,
    seed = config$model$seed,
    tuning_method = config$model$tuning$method,
    classification_threshold = config$model$classification_threshold,
    input_files = as.list(input_files),
    input_md5 = as.list(unname(tools::md5sum(input_files))) |>
      stats::setNames(names(input_files)),
    package_versions = as.list(package_versions)
  )
  yaml::write_yaml(manifest, file.path(session_dir, "run_manifest.yaml"))
  writeLines(utils::capture.output(utils::sessionInfo()),
             file.path(session_dir, "sessionInfo.txt"))
  invisible(TRUE)
}


# =============================================================================
# 15. Console reporting
# =============================================================================

#' Print the key results to the console.
print_run_summary <- function(results) {
  cat("\n===== Spatial cross-validation: model comparison =====\n")
  print(results$model_comparison[, c("Model", "Mean_AUC", "SD_AUC",
                                     "Mean_Sensitivity", "Mean_Specificity",
                                     "Mean_F1", "Mean_Balanced_Accuracy")],
        digits = 3, row.names = FALSE)
  if (!is.null(results$delta$summary)) {
    cat("\n===== Paired Delta AUC (descriptive) =====\n")
    for (note in results$delta$summary$NOTE) {
      cat(strwrap(note, width = 78, prefix = "  "), sep = "\n")
    }
  }
  cat("\n===== Selected model =====\n")
  cat(strwrap(results$selection$justification, width = 78, prefix = "  "),
      sep = "\n")
  if (!is.null(results$test)) {
    cat("\n===== Held-out test set =====\n")
    print(results$test$metrics[, c("MODEL", "EVALUATION_ROLE", "N", "AUC",
                                   "AUC_CI95_LOWER_DELONG",
                                   "AUC_CI95_UPPER_DELONG", "SENSITIVITY",
                                   "SPECIFICITY", "F1", "BALANCED_ACCURACY")],
          digits = 3, row.names = FALSE)
  }
  cat("\n")
}


# =============================================================================
# 16. Orchestration
# =============================================================================

#' Run the complete pipeline.
#'
#' Stages: configuration -> file checks -> training-data validation -> fold
#' validation -> spatial CV -> comparison / Delta AUC -> model selection ->
#' final fits -> variable importance -> figures -> save -> test evaluation ->
#' save. Stops before any modeling if data are absent or invalid.
#'
#' @param config_path Path to the YAML configuration.
#' @param validate_only If TRUE, check configuration and data, then stop
#'   without fitting any model.
#' @return Invisibly, a list of all results.
run_pipeline <- function(config_path = "config.yaml", validate_only = FALSE) {
  check_required_packages()
  config <- load_config(config_path)
  log_message("Configuration loaded: ", config$meta$config_path)
  set.seed(config$model$seed)

  run_test <- isTRUE(config$evaluation$run_test_evaluation)
  validate_file_paths(config, include_test = run_test)
  model_specs <- define_models(config)

  train_raw <- load_data(config$data$train_path_resolved, config)
  train_validation <- validate_data(train_raw, config, role = "Training")
  train_data <- train_validation$data
  fold_summary <- validate_spatial_folds(train_data, config)
  cat(paste0("  ", train_validation$report, "\n"), sep = "")
  cat("  Spatial folds:\n")
  print(as.data.frame(fold_summary), row.names = FALSE)

  if (validate_only) {
    if (!is.null(config$data$test_path_resolved) &&
        file.exists(config$data$test_path_resolved)) {
      test_validation <- validate_data(
        load_data(config$data$test_path_resolved, config), config,
        role = "Test")
      check_train_test_separation(train_data, test_validation$data, config)
      cat(paste0("  ", test_validation$report, "\n"), sep = "")
    }
    log_message("Validation passed. No models were fitted (--validate-only).")
    return(invisible(list(config = config, fold_summary = fold_summary)))
  }

  dirs <- setup_output_directories(config)
  save_diagnostics(train_validation, fold_summary, dirs)

  results <- list()
  results$cv <- run_spatial_cv(train_data, model_specs, config)
  results$model_comparison <- compare_models(results$cv, model_specs)
  results$delta <- compute_delta_auc(results$cv$metrics, config$comparisons)
  results$selection <- select_final_model(results$model_comparison,
                                          model_specs, config)
  results$final_models <- fit_final_models(train_data, model_specs, config)
  results$variable_importance <- extract_variable_importance(
    results$final_models, config)
  results$plots <- create_plots(results$cv, results$delta,
                                results$final_models, train_data,
                                model_specs, config)
  save_results(results, dirs, config)
  save_session_info(config, dirs, stage = "cross_validation")
  log_message("Cross-validation results saved to ", dirs[["root"]])

  if (run_test) {
    results$test <- evaluate_test_data(results$final_models,
                                       results$selection, train_data, config)
    save_results(list(test = results$test), dirs, config)
  } else {
    log_message("Test evaluation disabled (evaluation$run_test_evaluation: ",
                "false). The test data were not read.")
  }
  save_session_info(config, dirs, stage = "complete")

  print_run_summary(results)
  log_message("Run complete. Outputs in ", dirs[["root"]])
  invisible(results)
}


#' Command-line entry point.
main <- function() {
  options(warn = 1)
  cli <- parse_command_line()
  tryCatch(
    run_pipeline(cli$config_path, validate_only = cli$validate_only),
    error = function(e) {
      cat(conditionMessage(e), "\n", file = stderr())
      quit(save = "no", status = 1)
    }
  )
  invisible(NULL)
}


# Run only when executed with Rscript, not when source()d.
if (sys.nframe() == 0L) {
  main()
}
