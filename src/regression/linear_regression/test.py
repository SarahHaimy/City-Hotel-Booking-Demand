"""
test.py

Final evaluation of the trained linear regression ADR baseline on
data/preprocessed/test.csv. Run it once, after the variants are chosen and the
final model is trained. Variant selection must never use these numbers.

Usage (from the repo root):

    python src/regression/linear_regression/test.py

Reads the pipeline saved by train.py, scores every paying booking (ADR > 0) in
the test split, and writes metrics and diagnostic plots to
outputs/regression/linear/. When the selected config uses an IQR filter, a
secondary set of metrics is added for test rows inside the training limits.
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
    OUTPUT_DIR,
    evaluate,
    load_model,
    load_split,
    load_summary,
    save_diagnostic_plots,
)
from src.regression.linear_regression.preprocessing import paying_only  # noqa: E402


def main(argv=None) -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)

    summary = load_summary()
    model = load_model()
    limits = tuple(summary["iqr_limits"]) if summary.get("iqr_limits") else None

    test_all = load_split("test")
    test = paying_only(test_all)

    metrics, pred = evaluate(model, test, limits)
    metrics["n_rows_dropped_zero_adr"] = int(len(test_all) - len(test))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)
    pd.DataFrame({"actual": test[TARGET].to_numpy(), "predicted": pred}).to_csv(
        OUTPUT_DIR / "test_predictions.csv", index=False
    )
    save_diagnostic_plots(test[TARGET], pred, OUTPUT_DIR, "test")

    print("Linear regression baseline, test set (original ADR scale)")
    print(f"Config: {summary['config']}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"Saved metrics and plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
