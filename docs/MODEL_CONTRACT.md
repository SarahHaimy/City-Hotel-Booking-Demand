# Model Integration Contract (GUI + CLI)

Any regression, classification, or clustering model registered via `ModelSpec` automatically appears across **Model Catalog**, **Predict**, **Batch & Evaluate**, **Train Studio**, **Tuning Lab**, **Compare**, and **`main.py`** without editing GUI code.

## 1. Registering a New Model (~15 lines)

Add your builder (must use top-level functions — no lambdas — so `joblib` can pickle the fitted pipeline) and call `register_model(...)` inside `src/regression/models.py`, `src/classification/models.py`, or `src/clustering/models.py`:

```python
from src.registry import ModelSpec, ParamSpec, register_model

def build_my_model(params: dict, feature_space: str = "original", n_components=None):
    # Return an unfitted sklearn Pipeline (or TransformedTargetRegressor)
    # accepting raw H2 columns in .fit(X, y) and .predict(X)
    ...

register_model(
    ModelSpec(
        model_id="my_model",
        task="regression",                 # "regression" | "classification" | "clustering"
        display_name="My Custom Regressor",
        family="ensemble",
        description="Short 1–2 sentence summary shown on catalog cards.",
        assumptions_and_pros="Key assumptions and strengths/trade-offs.",
        required_inputs=RAW_INPUT_COLUMNS,
        forbidden_inputs=FORBIDDEN_COLUMNS,
        params=[
            ParamSpec("n_estimators", "Trees", "int", 200, low=50, high=500, step=50),
        ],
        baseline_params={"n_estimators": 100},
        tuned_params={"n_estimators": 300},
        search_space={"regressor__regressor__n_estimators": [100, 200, 300]},
        default_scoring="neg_root_mean_squared_error",
        build_pipeline=build_my_model,
    )
)
```

## 2. Training & Saving Official Artifacts

Run from the CLI (or via **Train Studio** in the GUI):

```bash
# Baseline + tuned bundles
python main.py train --task regression --model my_model
python main.py tune  --task regression --model my_model
```

This writes `artifacts/<task>/<model_id>/<variant>/`:
- `pipeline.joblib` — fitted preprocessing + dimensionality reduction + estimator
- `model_card.json` — hyperparameters, CV/val/test metrics, data hash, sklearn version, git SHA
- `predictions_val.csv` / `predictions_test.csv` — used for ROC, PR, calibration, parity, and residual overlays
- `cv_scores.json` & `tuning_results.csv` — used by Compare boxplots and Tuning Lab
- `extras/` — feature weights, threshold sweeps, segment error tables, cluster profiles, 2D PCA coordinates
