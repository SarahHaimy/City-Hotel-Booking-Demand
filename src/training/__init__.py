"""Training, hyperparameter optimization, clustering, and background job execution services."""
from src.training.jobs import JobCancelledError, JobRecord, JobRunner, get_job_runner
from src.training.service import (
    build_official_artifacts,
    compute_pca_lda_diagnostics,
    evaluate_bundle_on_data,
    extract_feature_weights,
    generate_sample_csvs,
    prepare_user_splits,
    train_clustering_model,
    train_supervised_model,
)

__all__ = [
    "JobCancelledError",
    "JobRecord",
    "JobRunner",
    "build_official_artifacts",
    "compute_pca_lda_diagnostics",
    "evaluate_bundle_on_data",
    "extract_feature_weights",
    "generate_sample_csvs",
    "get_job_runner",
    "prepare_user_splits",
    "train_clustering_model",
    "train_supervised_model",
]
