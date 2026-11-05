# City Hotel Booking Demand: Machine Learning & Predictive Modeling

An end-to-end machine learning repository analyzing booking demand patterns, room pricing, and cancellation behavior on the **City Hotel (H2)** dataset. The project implements reproducible, leakage-free data pipelines alongside three core analytical tracks: **ADR (Average Daily Rate) regression**, **cancellation classification**, and **unsupervised customer segmentation (clustering)**. 

> Note:
> **Project Status**: **Development is currently in progress for each model track.** Current implementations establish rigorous preprocessed baselines, feature engineering benchmarks, and hyperparameter explorations. Advanced architectures, expanded tuning grids, and ensemble methods are under active development.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13+ |
| Core ML Framework | scikit-learn (`1.9.1`) |
| Data Processing | pandas (`3.1.0`), numpy (`2.5.4`), scipy (`1.18.1`) |
| Model Serialization | joblib (`1.6.0`) |
| Data Visualization | matplotlib (`3.11.2`), seaborn (`0.13.2`) |
| Testing | pytest (`7.4.2`) |
| Architecture Pattern | Scikit-learn `Pipeline` & `ColumnTransformer` with zero data leakage |

---

## Core Tracks & Pipeline Architecture

### 1. Data Cleaning & Anti-Leakage Pipeline
- **Strict Separation of Concerns**: Preprocessing is partitioned into stateless structural cleaning (executed before splitting) and data-dependent transformations (fit exclusively on the training split).
- **Stratified Partitioning**: Disjoint 70% train / 15% validation / 15% test splits stratified on `IsCanceled` to preserve outcome distributions across sets.
- **Leakage Guards**: Automated assertions prohibit downstream outcome proxies (`DepositType`, `ReservationStatus`, `AssignedRoomType`, `DaysInWaitingList`) from entering predictive feature matrices.
- **Normalization & Cleansing**: Whitespace stripping, explicit type coercion (`Agent`/`Company` to nullable string IDs), guest count validation, zero-night filtering, and dropping sparse fields (`Company` ~95% missing).

### 2. Regression: ADR Prediction (`TARGET = "ADR"`)
*Status: In Progress — Linear baseline & feature ablation complete; non-linear models in progress.*
- **Objective**: Forecast the Average Daily Rate (ADR) of valid bookings at reservation time to support revenue management.
- **Architecture**: `TransformedTargetRegressor` applying a `log1p` target transform and wrapping a composite feature pipeline into `LinearRegression`.
- **Ablation & Variant Testing**: Iterative validation against 6 hypothesis dimensions:
  1. *Target Transformation*: `log_adr` vs raw ADR
  2. *Outlier Filtering*: IQR bounds vs retaining full spectrum
  3. *Seasonality Representation*: One-hot monthly encoding vs harmonic sine/cosine cyclic features
  4. *Agent Grouping*: Minimum frequency thresholding (`min_freq = 10`)
  5. *Customer Flags*: Prior cancellation & repeated guest indicators
  6. *Stay Duration Dynamics*: Log total nights and weekend share ratios
- **Current Performance**: Validation RMSE of **20.23** ($R^2 = 0.736$), 5-fold CV RMSE of **20.40**, and Test RMSE of **20.42** (MAE **13.84**, $R^2 = 0.740$) across 7,848 test samples.

### 3. Classification: Booking Cancellation (`TARGET = "IsCanceled"`)
*Status: In Progress — Gaussian Naive Bayes baseline, hyperparameter tuning & calibration complete; tree-based models in progress.*
- **Objective**: Predict whether a booking will be canceled (`0 = No`, `1 = Yes`) using only reservation-time attributes.
- **Architecture**: `GaussianNB` paired with one-hot categorical encoding and numerical pass-through features.
- **Variance Smoothing & Calibration**:
  - Grid search over variance smoothing parameters (`var_smoothing = 1e-5` selected).
  - Sigmoid probability calibration via 5-fold `CalibratedClassifierCV`.
  - Decision threshold optimization ($0.30 - 0.90$) to balance recall and minimize false cancellation alarms across market segments.
- **Current Performance**: Tuned validation F1 score of **~60%** with calibrated decision boundary; error analysis by `MarketSegment` reveals distinct false alarm tendencies under traditional 0.5 cutoffs.

### 4. Clustering: Customer Segmentation
*Status: In Progress — K-Means baseline with PCA reduction complete; alternate $k$ and cluster algorithms in progress.*
- **Objective**: Discover distinct behavioral cohorts of hotel guests to enable targeted marketing, dynamic cancellation policies, and operational planning.
- **Architecture**: Unsupervised `Pipeline` integrating `log1p` transformations, `StandardScaler`, frequency-filtered `OneHotEncoder`, **PCA (90% variance retention, 7 components)**, and Lloyd's `KMeans`.
- **Validation & Profiling**:
  - Evaluation via Silhouette Score (**0.180**), Davies-Bouldin Index (**1.763**), and Calinski-Harabasz Index (**11,518.79**).
  - Post-hoc leakage checks verifying cluster separation and examining correlated cancellation propensities (e.g., Cluster 0: 29.8%, Cluster 1: 16.1%, Cluster 2: 40.0%).
  - Customer persona mapping (e.g., short-stay transient business travelers vs high-lead leisure groups).

---

## Project Structure

```text
City-Hotel-Booking-Demand/
├── artifacts/                                # Serialized pipelines and model metadata
│   ├── kmeans_baseline_pca90_pipeline.joblib # Serialized K-Means + PCA pipeline
│   └── regression/
│       └── linear_regression/
│           ├── linear_regression.joblib      # Serialized scikit-learn ADR pipeline
│           ├── selected_config.json          # Optimal variant hyperparameter config
│           └── training_summary.json         # Training parameters and validation scores
├── data/
│   ├── H2.csv                                # Raw City Hotel dataset
│   ├── bayes/                                # Data splits prepared for classification
│   ├── preprocessed/                         # Preprocessed train/val/test splits
│   └── processed/                            # Cleaned, stratified splits (70/15/15)
├── miscellaneous/
│   └── EDA Report_ City Hotel Booking Demand.pdf # Comprehensive domain & EDA report
├── outputs/                                  # Evaluation metrics, tables, and diagnostic plots
│   ├── clustering/
│   │   ├── figures/                          # PCA scatter plots and cluster visualizations
│   │   └── tables/                           # Metrics, cluster profiles, and cancellation rates
│   ├── eda/
│   │   ├── figures/                          # Distributions, correlation heatmaps, missingness
│   │   └── tables/                           # Cardinality, data types, statistical summaries
│   └── regression/
│       └── linear_regression/
│           ├── coefficients.csv              # Model weights and feature attributions
│           ├── test_metrics.csv              # Test evaluation metrics (MAE, RMSE, R2)
│           ├── test_predicted_vs_actual.png  # Diagnostic parity plot on test split
│           ├── test_residuals_vs_predicted.png # Residual analysis plot
│           └── variant_summary.csv           # Ablation comparison across all test phases
├── src/
│   ├── classification/                       # Classification track (cancellation prediction)
│   │   ├── bayes_preprocessing.py            # Feature definitions and encoding functions
│   │   ├── final_gaussian_nb.py              # Evaluated model script with segment error analysis
│   │   ├── gaussian_nb_baseline.py           # Baseline Naive Bayes implementation
│   │   ├── gaussian_nb_calibration.py        # Sigmoid calibration and threshold tuning
│   │   ├── gaussian_nb_experiments.py        # Hyperparameter search over var_smoothing
│   │   ├── gaussian_nb_feature_experiments.py# Feature subsetting & ablation tests
│   │   └── prepare_bayes_data.py             # Data preparation script for Bayes models
│   ├── clustering/                           # Clustering track (customer segmentation)
│   │   └── k_means/
│   │       ├── audit_functions.py            # Feature audits and variance checks
│   │       ├── features.py                   # Cluster feature schemas and leakage assertions
│   │       ├── learn.py                      # K-Means training, PCA fitting, and profiling
│   │       ├── pipeline.py                   # Preprocessing + PCA + KMeans pipeline builder
│   │       └── preprocessing.py              # ColumnTransformer configuration
│   ├── data/                                 # Data loading, cleaning, and partitioning
│   │   ├── data_cleaning.py                  # Stateless row-level cleaning logic
│   │   ├── data_split.py                     # Stratified train/val/test split generator
│   │   └── loader.py                         # H2 dataset loader and format sanitizer
│   ├── eda/
│   │   └── run_eda1.py                       # Exploratory Data Analysis runner
│   ├── evaluation/                           # Shared metrics and scoring routines
│   ├── inference/                            # End-to-end inference utilities
│   ├── regression/                           # Regression track (ADR prediction)
│   │   └── linear_regression/
│   │       ├── features.py                   # Feature engineering and RegressionConfig
│   │       ├── linear_regression.py          # TransformedTargetRegressor pipeline & diagnostics
│   │       ├── preprocessing.py              # ColumnTransformer & IQR bounds
│   │       ├── test.py                       # Test split evaluation script
│   │       └── train.py                      # Variant testing and final model training
│   └── tuning/                               # Search and optimization routines
├── tests/
│   ├── test_data_pipelines.py                # Unit tests for loading, cleaning, and splitting
│   └── test_pipelines.py                     # Pipeline integration tests
├── requirements.txt                          # Project dependencies
└── README.md                                 # Project documentation
```

---

## Execution Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/SarahHaimy/City-Hotel-Booking-Demand.git
cd City-Hotel-Booking-Demand

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Exploratory Data Analysis (EDA)

Run the automated EDA suite to generate summary statistics, distribution plots, and correlation matrices under `outputs/eda/`:

```bash
python src/eda/run_eda1.py
```

### 3. Data Ingestion & Splitting

Clean raw bookings and generate stratified 70/15/15 partitions:

```bash
python -c "from src.data.loader import load_h2; from src.data.data_cleaning import clean_data; from src.data.data_split import split_data, save_splits; df, _ = clean_data(load_h2()); tr, va, te = split_data(df); save_splits(tr, va, te, 'data/processed')"
```

### 4. Running the Regression Track (ADR)

```bash
# Run ablation variant tests (Tests 1-5) and select optimal feature config
python src/regression/linear_regression/train.py --variants

# Train the final pipeline using the selected configuration
python src/regression/linear_regression/train.py

# Evaluate on the unseen test split and generate parity/residual plots
python src/regression/linear_regression/test.py
```

### 5. Running the Classification Track (Cancellation)

```bash
# Prepare classification-specific datasets
python src/classification/prepare_bayes_data.py

# Run baseline Gaussian Naive Bayes
python src/classification/gaussian_nb_baseline.py

# Perform variance smoothing and decision threshold experiments
python src/classification/gaussian_nb_experiments.py

# Train calibrated model and evaluate segment-level false alarms
python src/classification/final_gaussian_nb.py
```

### 6. Running the Clustering Track (K-Means)

```bash
# Train K-Means (k=3) with 90% PCA variance retention and generate profile tables
python -m src.clustering.k_means.learn --pca 0.90 --k 3
```

### 7. Running Tests

Execute the unit test suite validating data integrity, duplicate handling, and partition isolation:

```bash
python -m pytest tests/test_data_pipelines.py -v
```

---

## Current Experimental Results

### Regression: ADR Prediction Benchmark

| Stage | Model Configuration | Metric | Score | Sample Size ($N$) |
|---|---|---|---|---|
| **Validation** | Baseline (`LinearRegression`, raw target) | RMSE | 20.48 | 7,848 |
| **Validation** | Variant Winner (`log1p` target, agent grouping, repeated guest flag) | RMSE | **20.23** | 7,848 |
| **Validation** | Variant Winner ($R^2$ Score) | $R^2$ | **0.736** | 7,848 |
| **Cross-Validation**| 5-Fold Cross-Validation | CV RMSE | **20.40** | 36,622 (Train) |
| **Test (Final)** | Linear Regression Pipeline on Unseen Test Split | MAE / RMSE / $R^2$ | **13.84** / **20.42** / **0.740** | 7,848 (Test) |

### Classification: Cancellation Prediction (Gaussian Naive Bayes)

| Variant | var_smoothing | Threshold | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline GNB | `1e-9` (default) | 0.50 | 56.40% | 61.20% | 58.70% | 0.7780 |
| Smoothed GNB | `1e-5` (tuned) | 0.50 | 58.82% | 60.91% | 59.84% | 0.7812 |
| Tuned Operating Point | `1e-5` (tuned) | 0.60 | **65.12%** | **52.40%** | **58.07%** | **0.7812** |
| Calibrated (Sigmoid) | `1e-5` (5-fold CV)| 0.50 | 62.45% | 56.10% | 59.11% | 0.7815 |

### Clustering: Customer Segmentation Benchmark

| Configuration | Principal Components | Variance Explained | Silhouette Score | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|---|
| K-Means ($k=3$) + PCA | 7 components | 90.63% | **0.180** | **1.763** | **11,518.79** |

*Cluster Cohort Summary:*
- **Cluster 0 (41.5%)**: High-value leisure bookings (Average ADR \$134.12, 2.45 guests, 89-day lead time, 29.8% cancel rate).
- **Cluster 1 (22.2%)**: Short-lead, short-stay bookings (10.6-day lead time, 1.67 nights, 16.1% cancel rate — lowest cancellation risk).
- **Cluster 2 (36.3%)**: Long-lead, budget transient bookings (108.3-day lead time, 3.81 nights, ADR \$94.70, 40.0% cancel rate — highest cancellation risk).

---

## Model Development Status & Roadmap

Each analytical track is in active development with the following roadmap milestones:

### Data & Infrastructure
- [x] Ingestion and whitespace sanitization pipeline
- [x] Leakage-free structural cleaning and column mapping
- [x] Stratified 70/15/15 train/validation/test split generation
- [x] Unit test suite for data loaders and cleaning transforms (`tests/test_data_pipelines.py`)
- [ ] Automated end-to-end execution pipeline via `main.py` CLI / `config.yaml`

### Regression Track (ADR)
- [x] Baseline linear regression model implementation
- [x] 6-stage systematic variant and ablation test harness
- [x] Target transform (`log1p`/`expm1`) and residual evaluation
- [x] Diagnostic evaluation (predicted vs actual, residual plots, coefficient audit)
- [ ] *In Progress*: Regularized regression baselines (Ridge, Lasso, ElasticNet)
- [ ] *In Progress*: Non-linear ensemble models (Random Forest Regressor, LightGBM / XGBoost)
- [ ] *In Progress*: Hyperparameter grid / randomized search integration

### Classification Track (Cancellation)
- [x] Baseline Gaussian Naive Bayes classifier
- [x] Variance smoothing optimization (`var_smoothing`)
- [x] Sigmoid probability calibration (`CalibratedClassifierCV`)
- [x] Threshold decision tuning and false alarm rate audit across market segments
- [ ] *In Progress*: Tree-based and linear classifiers (Logistic Regression, Random Forest, HistGradientBoosting)
- [ ] *In Progress*: Class imbalance handling (cost-sensitive learning / SMOTE)
- [ ] *In Progress*: Full ROC-PR / ROC-AUC curve diagnostics and deployment threshold selection

### Clustering Track (Segmentation)
- [x] Leakage-free clustering feature set definition & audit
- [x] Scaled numerical + one-hot categorical preprocessing pipeline
- [x] PCA dimensionality reduction (90% variance thresholding)
- [x] K-Means ($k=3$) baseline fitting, inertia evaluation, and cluster profiling
- [ ] *In Progress*: Multi-k evaluation via Elbow curve and Silhouette analysis ($k \in [2, 8]$)
- [ ] *In Progress*: Alternative clustering algorithms (DBSCAN, Agglomerative Hierarchical Clustering)
- [ ] *In Progress*: Business persona translation and cluster-specific booking strategies
