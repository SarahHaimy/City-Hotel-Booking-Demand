"""Shared evaluation metrics, Plotly diagnostic figures, and comparison report builders."""
from src.evaluation.metrics import (
    METRIC_META,
    TASK_METRICS,
    compute_classification_metrics,
    compute_clustering_metrics,
    compute_regression_metrics,
    format_metric,
    segment_error_table,
    sweep_thresholds,
)
from src.evaluation.report import build_comparison_html, build_comparison_markdown
from src.evaluation.summary import generate_comparison_summary

__all__ = [
    "METRIC_META",
    "TASK_METRICS",
    "build_comparison_html",
    "build_comparison_markdown",
    "compute_classification_metrics",
    "compute_clustering_metrics",
    "compute_regression_metrics",
    "format_metric",
    "generate_comparison_summary",
    "segment_error_table",
    "sweep_thresholds",
]
