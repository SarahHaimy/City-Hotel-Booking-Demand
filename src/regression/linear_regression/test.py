"""
test.py

Final evaluation of the trained linear regression ADR baseline on
data/preprocessed/test.csv. Run it once, after the variants are chosen and the
final model is trained. Variant selection must never use these numbers.

Usage (from the repo root):

    python src/regression/linear_regression/test.py

Reads the pipeline saved by train.py, scores every paying booking (ADR > 0) in
the test split, and writes metrics and diagnostic plots to
outputs/regression/linear_regression/<run_name>/. Pass --run-name to select a
specific training run; otherwise the most recently modified artifact run is
used. When the selected config uses an IQR filter, a secondary set of metrics
is added for test rows inside the training limits.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.regression.linear_regression.features import TARGET  # noqa: E402
from src.regression.linear_regression.linear_regression import (  # noqa: E402
    ARTIFACT_DIR,
    evaluate,
    get_run_dir,
    get_latest_run_name,
    load_model,
    load_split,
    load_summary,
    save_diagnostic_plots,
)
from src.regression.linear_regression.preprocessing import paying_only  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=int, default=None,
                        help="legacy numeric run ID")
    parser.add_argument("--run-name", default=None,
                        help="name of the training run to evaluate (default: latest artifact run)")
    args = parser.parse_args(argv)

    if args.run is not None and args.run_name is not None:
        parser.error("--run and --run-name cannot be used together")

    run_id = args.run
    run_name = args.run_name
    if run_id is None and run_name is None:
        run_name = get_latest_run_name(ARTIFACT_DIR)

    summary = load_summary(run_id=run_id, run_name=run_name)
    model = load_model(run_id=run_id, run_name=run_name)
    limits = tuple(summary["iqr_limits"]) if summary.get("iqr_limits") else None

    if run_name is not None:
        run_dir = get_run_dir(run_name=run_name)
    elif run_id is not None:
        run_dir = get_run_dir(run_id)
    elif summary.get("output_dir") and Path(summary["output_dir"]).exists():
        run_dir = Path(summary["output_dir"])
    elif summary.get("run_id") is not None:
        run_dir = get_run_dir(summary["run_id"])
    else:
        run_dir = get_run_dir()

    test_all = load_split("test")
    test = paying_only(test_all)

    metrics, pred = evaluate(model, test, limits)
    metrics["n_rows_dropped_zero_adr"] = int(len(test_all) - len(test))

    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(run_dir / "test_metrics.csv", index=False)
    pd.DataFrame({"actual": test[TARGET].to_numpy(), "predicted": pred}).to_csv(
        run_dir / "test_predictions.csv", index=False
    )
    save_diagnostic_plots(test[TARGET], pred, run_dir, "test")

    print("Linear regression baseline, test set (original ADR scale)")
    print(f"Config: {summary['config']}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"Saved metrics and plots to {run_dir}")


if __name__ == "__main__":
    main()
