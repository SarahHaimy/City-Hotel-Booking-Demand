# CLI Reference (`main.py`)

Unified command-line interface supporting both an **interactive menu** (with per-level Help) and **direct one-line subcommands** for training, tuning, evaluating, clustering, predicting, and launching the Streamlit ML Studio.

```bash
# Launch interactive menu (Category -> Model -> Action -> Options)
python main.py
python main.py menu

# Or invoke a direct subcommand
python main.py <command> [options]
```

Run `python main.py -h` or `python main.py <command> -h` at any time to view built-in help.

---

## Run Naming & Overwrite Semantics

- **Automatic timestamp defaults (no accidental overwrites):** Omitting `--run-name` on `train`, `tune`, or `cluster` automatically generates a unique timestamped run name (e.g., `baseline_20261126_170512`), so consecutive training runs never overwrite each other.
- **Explicit `--run-name` (clean scoped replacement):** Passing `--run-name <name>` saves the run under `artifacts/<task>/<model_id>/<name>/` (or `artifacts/user/<task>_<model_id>_<name>/` when `--data` is used). Re-running with the same `--run-name` cleanly wipes and replaces **only** that run's directory so no stale files remain.
- **Required on `evaluate`:** `evaluate` always requires an explicit `--run-name <name>` (or `--run-name all` to evaluate every saved run for that task/model).

---

## Command Summary

| Command | Purpose |
|---|---|
| *(no args)* / `menu` | Launch the interactive CLI menu with in-menu Help |
| `gui` | Launch the interactive Streamlit ML Studio |
| `list-models` | List all registered models across tracks |
| `analyze` | Run exploratory data analysis (writes `outputs/eda/`) |
| `build-artifacts` | Build official baseline, tuned, and PCA/LDA artifact bundles |
| `train` | Train a supervised regression or classification model |
| `tune` | Run cross-validated hyperparameter optimization (supervised) |
| `evaluate` | Evaluate a named saved run on `val`, `test`, or a custom CSV/XLSX |
| `cluster` | Fit or optimize an unsupervised clustering algorithm |
| `predict` | Run deployed inference pipelines on a preset booking or batch file |

---

## 1. Interactive Menu (`python main.py` or `python main.py menu`)

Guides you step-by-step through:
1. **Category Menu** (`Regression`, `Classification`, `Clustering`, `EDA`, `Predict`, `GUI`)
2. **Model Menu** (dynamically loaded from the model registry)
3. **Action Menu** (only actions supported by that model — e.g., `cluster` only for K-Means)
4. **Options & Run Name** (including model-specific options like `--variants` or `--calibrate`)

- Type `h` at any menu or prompt for contextual help explaining choices, inputs, outputs, and assumptions, then return to the same prompt.
- Invalid or conflicting selections reprompt automatically.
- Before executing, the menu prints the equivalent direct `python main.py ...` command.

---

## 2. `list-models`

List registered models from the central registry (`src/registry/`).

| Flag | Values | Default | Description |
|---|---|---|---|
| `--task` | `regression`, `classification`, `clustering` | all tasks | Filter models by analytical track |

```bash
python main.py list-models
python main.py list-models --task regression
```

---

## 3. `gui`

Launch the multipage Streamlit ML Studio (`app/streamlit_app.py`).

| Flag | Type | Default | Description |
|---|---|---|---|
| `--port` | `int` | `8501` | Port to serve Streamlit on |
| `--headless` | flag | `False` | Run without opening a browser window |

```bash
python main.py gui
python main.py gui --port 8502 --headless
```

---

## 4. `analyze`

Run the automated EDA script (`src/eda/run_eda1.py`) and save tables/figures under `outputs/eda/`.

```bash
python main.py analyze
```

---

## 5. `build-artifacts`

Build official artifact bundles under `artifacts/<task>/<model_id>/<variant>/`.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--fast` | flag | `False` | Use validated tuned presets instead of running full CV grid search |
| `--core-only` | flag | `False` | Build only the 3 Milestone 1 core models |
| `--resume` | flag | `False` | Skip bundles that already exist on disk |

```bash
python main.py build-artifacts
python main.py build-artifacts --fast
python main.py build-artifacts --core-only --resume
```

---

## 6. `train`

Train a supervised regression or classification model and save its `ArtifactBundle`.

| Flag | Values | Default | Description |
|---|---|---|---|
| `--task` *(required)* | `regression`, `classification` | — | Target task |
| `--model` *(required)* | Registered `model_id` or alias | — | Model architecture to train |
| `--mode` | `baseline`, `tuned_preset` | `baseline` | Hyperparameter preset |
| `--feature-space` | `original`, `pca`, `lda` | `original` | Feature representation |
| `--pca` | `float` | `0.90` | PCA explained variance ratio (used when `--feature-space pca`) |
| `--data` | file path | official splits | Custom CSV/XLSX (auto-split 70/15/15, saved under `artifacts/user/`) |
| `--run-name` | string (`[A-Za-z0-9._-]`) | timestamp | Explicit run name; reusing a name cleanly replaces that run |

**Model-specific options** (validated per model; passing an unsupported or conflicting option exits with an error):

| Model | Flag | Type | Default | Description |
|---|---|---|---|---|
| `linear_regression` | `--variants` | flag | `False` | Run sequential 5-stage ablation tests on train/val and fit the winning config (saves variant tables in `extras/`) |
| `linear_regression` | `--with-optional` | flag | `False` | Include optional Test 6 (`night_features`); requires `--variants` |
| `linear_regression` | `--tol` | `float` | `0.005` | Relative RMSE improvement tolerance for winner selection; requires `--variants` |
| `gaussian_nb` | `--calibrate` | flag | `False` | Wrap `GaussianNB` in 5-fold Sigmoid `CalibratedClassifierCV` |
| `gaussian_nb` | `--var-smoothing` | `float` | preset | Override `var_smoothing` (e.g. `1e-5`) |
| `gaussian_nb` | `--threshold` | `float` | preset | Override classification decision threshold (`0.0`–`1.0`) |

```bash
# Auto-timestamped baseline run (never overwrites previous runs)
python main.py train --task regression --model linear_regression

# Named Linear Regression run with sequential variant ablation
python main.py train --task regression --model linear_regression --variants --run-name test_run23
python main.py train --task regression --model linear_regression --variants --with-optional --tol 0.005 --run-name ablation_full

# Named Gaussian Naive Bayes run with calibration & custom smoothing
python main.py train --task classification --model gaussian_nb --calibrate --var-smoothing 1e-5 --run-name gnb_cal

# Train on a custom dataset
python main.py train --task regression --model linear_regression --data path/to/bookings.csv --run-name custom_exp1
```

---

## 7. `tune`

Run cross-validated hyperparameter optimization strictly on the training split (never touches the test split) and automatically tune the classification decision threshold on validation.

| Flag | Values | Default | Description |
|---|---|---|---|
| `--task` *(required)* | `regression`, `classification` | — | Target task |
| `--model` *(required)* | Registered `model_id` or alias | — | Model architecture to optimize |
| `--method` | `random`, `grid` | `random` | `RandomizedSearchCV` vs exhaustive `GridSearchCV` |
| `--cv` | `int` | `5` | Number of cross-validation folds |
| `--n-iter` | `int` | `10` | Max candidate trials (for `--method random`) |
| `--feature-space` | `original`, `pca`, `lda` | `original` | Feature representation |
| `--run-name` | string | timestamp | Explicit run name; reusing a name cleanly replaces that run |

```bash
python main.py tune --task regression --model linear_regression --run-name lr_cv5
python main.py tune --task classification --model gaussian_nb --method grid --cv 5 --run-name gnb_grid
```

---

## 8. `evaluate`

Evaluate a saved run on the validation split, unseen test split, or an external labeled CSV/XLSX without retraining.

| Flag | Values | Default | Description |
|---|---|---|---|
| `--task` *(required)* | `regression`, `classification` | — | Target task |
| `--run-name` *(required)* | Saved run name or `all` | — | Explicit run name to evaluate |
| `--model` | `all`, `model_id`, or alias | `all` | Narrow evaluation to one model |
| `--split` | `val`, `test` | `val` | Official split to score (ignored if `--data` is set) |
| `--data` | file path | `None` | Optional external labeled CSV/XLSX to evaluate on |

```bash
python main.py evaluate --task regression --run-name test_run23 --split test
python main.py evaluate --task classification --model gaussian_nb --run-name gnb_cal --split val
python main.py evaluate --task classification --run-name all --data data/samples/sample_bookings_labeled.csv
```

---

## 9. `cluster`

Fit or optimize an unsupervised clustering algorithm on cleaned City Hotel bookings.

| Flag | Values | Default | Description |
|---|---|---|---|
| `--algorithm` | `kmeans`, `gmm`, `agglomerative`, `dbscan` (or alias) | `kmeans` | Registered clustering algorithm |
| `--mode` | `baseline`, `tune` | `baseline` | Fit single configuration or optimize Silhouette score |
| `--k` | `int` | `None` | Override `n_clusters` / `n_components` |
| `--pca` | `none` or float (e.g. `0.90`) | `0.90` | PCA variance retention ratio, or `none` for original space |
| `--run-name` | string | timestamp | Explicit run name; reusing a name cleanly replaces that run |
| `--init` | `k-means++`, `random` | `k-means++` | `[kmeans]` Centroid initialization strategy |
| `--n-init` | `int` | `10` | `[kmeans]` Number of centroid initializations |

```bash
python main.py cluster --algorithm kmeans --pca 0.90 --run-name km_pca90
python main.py cluster --algorithm kmeans --k 4 --pca none --init random --n-init 15 --run-name km_k4_nopca
python main.py cluster --algorithm kmeans --mode tune --run-name km_tuned
```

---

## 10. `predict`

Run inference using the currently deployed bundles (`artifacts/deployed.json`) without retraining.

| Flag | Values | Default | Description |
|---|---|---|---|
| `--task` | `regression`, `classification`, `all` | `all` | Which deployed pipeline(s) to run |
| `--preset` | Built-in preset name (see below) | First preset | Single-booking preset (ignored when `--input` is given) |
| `--input` | file path | `None` | Batch CSV/XLSX file to score |
| `--output` | file path | prints top 15 rows | Destination CSV path for batch predictions |

**Available `--preset` values:**
- `"Summer Couple Leisure (Online TA)"`
- `"Last-Minute Business Solo (Corporate)"`
- `"Family Holiday Suite (Direct)"`
- `"High-Risk Long-Lead Group Booking"`

```bash
python main.py predict --task all
python main.py predict --task classification --preset "High-Risk Long-Lead Group Booking"
python main.py predict --task all --input data/samples/sample_bookings_unlabeled.csv --output predictions.csv
```

---

## Model Aliases

`main.py` resolves shorthand model names to canonical `model_id`s:

| Task | Alias | Resolves To |
|---|---|---|
| `regression` | `ols` | `linear_regression` |
| `regression` | `rf`, `random_forest` | `random_forest_reg` |
| `regression` | `xgboost`, `gradient_boosting` | `hist_gb_reg` |
| `regression` | `decision_tree` | `decision_tree_reg` |
| `regression` | `knn` | `knn_reg` |
| `regression` | `mlp` | `mlp_reg` |
| `classification` | `naive_bayes` | `gaussian_nb` |
| `classification` | `rf`, `random_forest` | `random_forest_clf` |
| `classification` | `xgboost`, `gradient_boosting` | `hist_gb_clf` |
| `classification` | `decision_tree` | `decision_tree_clf` |
| `classification` | `knn` | `knn_clf` |
| `classification` | `svm` | `rbf_svm` |
| `classification` | `mlp` | `mlp_clf` |
| `clustering` | `hierarchical` | `agglomerative` |
