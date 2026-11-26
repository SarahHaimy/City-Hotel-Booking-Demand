"""Unified CLI & GUI launcher for City Hotel Booking Demand (CSE382).

Examples (matching Specification Section 14):
    python main.py gui
    python main.py build-artifacts
    python main.py analyze
    python main.py train --task regression --model random_forest
    python main.py train --task classification --model xgboost
    python main.py tune  --task regression --model linear_regression
    python main.py evaluate --task classification --model all
    python main.py cluster --algorithm kmeans
    python main.py predict --task regression
    python main.py predict --task classification
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schema import BOOKING_PRESETS  # noqa: E402
from src.data.validation import ensure_task_splits, load_uploaded_table  # noqa: E402
from src.evaluation.metrics import TASK_METRICS  # noqa: E402
from src.inference.predict import predict_batch_dataframe, predict_single_booking  # noqa: E402
from src.registry.artifacts import get_store  # noqa: E402
from src.registry.registry import list_specs  # noqa: E402
from src.training.service import (  # noqa: E402
    build_official_artifacts,
    evaluate_bundle_on_data,
    prepare_user_splits,
    train_clustering_model,
    train_supervised_model,
)

MODEL_ALIASES: dict[tuple[str, str], str] = {
    ("regression", "random_forest"): "random_forest_reg",
    ("regression", "rf"): "random_forest_reg",
    ("regression", "xgboost"): "hist_gb_reg",
    ("regression", "gradient_boosting"): "hist_gb_reg",
    ("regression", "decision_tree"): "decision_tree_reg",
    ("regression", "knn"): "knn_reg",
    ("regression", "mlp"): "mlp_reg",
    ("regression", "ols"): "linear_regression",
    ("classification", "random_forest"): "random_forest_clf",
    ("classification", "rf"): "random_forest_clf",
    ("classification", "xgboost"): "hist_gb_clf",
    ("classification", "gradient_boosting"): "hist_gb_clf",
    ("classification", "decision_tree"): "decision_tree_clf",
    ("classification", "knn"): "knn_clf",
    ("classification", "svm"): "rbf_svm",
    ("classification", "mlp"): "mlp_clf",
    ("classification", "naive_bayes"): "gaussian_nb",
    ("clustering", "hierarchical"): "agglomerative",
}


def _resolve_model(task: str, model_id: str) -> str:
    return MODEL_ALIASES.get((task, model_id.lower()), model_id)


def cmd_gui(args: argparse.Namespace) -> int:
    app_path = ROOT / "app" / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(args.port)]
    if args.headless:
        cmd.extend(["--server.headless", "true"])
    return subprocess.call(cmd, cwd=ROOT)


def cmd_analyze(_args: argparse.Namespace) -> int:
    eda_script = ROOT / "src" / "eda" / "run_eda1.py"
    return subprocess.call([sys.executable, str(eda_script)], cwd=ROOT)


def cmd_list_models(args: argparse.Namespace) -> int:
    specs = list_specs(args.task)
    rows = [
        {
            "task": s.task,
            "model_id": s.model_id,
            "display_name": s.display_name,
            "family": s.family,
            "cost": s.cost_hint,
            "origin": s.track_origin,
        }
        for s in specs
    ]
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    mid = _resolve_model(args.task, args.model)
    splits_arg = None
    source = "project"
    ds_name = "H2 City Hotel (official splits)"
    if args.data:
        df_u = load_uploaded_table(args.data)
        tr, va, te, _ = prepare_user_splits(df_u, task=args.task)
        splits_arg = (tr, va, te)
        source = "user"
        ds_name = Path(args.data).name

    bundle = train_supervised_model(
        task=args.task,
        model_id=mid,
        mode=args.mode,
        feature_space=args.feature_space,
        pca_components=args.pca,
        evaluate_test=True,
        splits=splits_arg,
        dataset_name=ds_name,
        source=source,
        progress_cb=lambda p, m: print(f"[{p * 100:5.1f}%] {m}"),
    )
    print(f"\nSaved bundle: {bundle.card.bundle_id}")
    print(json.dumps(bundle.card.metrics, indent=2))
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    mid = _resolve_model(args.task, args.model)
    bundle = train_supervised_model(
        task=args.task,
        model_id=mid,
        mode="tune",
        search_method=args.method,
        cv_folds=args.cv,
        n_iter=args.n_iter,
        feature_space=args.feature_space,
        tune_threshold=(args.task == "classification"),
        evaluate_test=True,
        progress_cb=lambda p, m: print(f"[{p * 100:5.1f}%] {m}"),
    )
    print(f"\nSaved tuned bundle: {bundle.card.bundle_id}")
    print("Best params:", json.dumps(bundle.card.params, indent=2))
    print("Metrics:", json.dumps(bundle.card.metrics, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    store = get_store()
    bundles = store.list_bundles(task=args.task)
    if args.model != "all":
        mid = _resolve_model(args.task, args.model)
        bundles = [b for b in bundles if b.card.model_id == mid]
    if not bundles:
        print(f"No saved bundles found for task={args.task}, model={args.model}. Run `build-artifacts` or `train` first.")
        return 1

    if args.data:
        eval_df = load_uploaded_table(args.data)
    else:
        _, val_df, test_df = ensure_task_splits(args.task)
        eval_df = test_df if args.split == "test" else val_df

    rows = []
    for b in bundles:
        ev = evaluate_bundle_on_data(b, eval_df)
        rows.append(
            {
                "bundle_id": b.card.bundle_id,
                "variant": b.card.variant,
                "space": b.card.feature_space,
                "deployed": "*" if b.card.is_deployed else "",
                **{k: round(ev["metrics"][k], 4) if isinstance(ev["metrics"].get(k), float) else ev["metrics"].get(k) for k in TASK_METRICS[args.task]},
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


def cmd_cluster(args: argparse.Namespace) -> int:
    mid = _resolve_model("clustering", args.algorithm)
    fs = "original" if str(args.pca).lower() == "none" else "pca"
    pca_v = None if fs == "original" else float(args.pca)
    custom_p = {"n_clusters": args.k, "n_components": args.k} if args.k else None
    mode = "custom" if custom_p and args.mode == "baseline" else args.mode

    bundle = train_clustering_model(
        model_id=mid,
        mode=mode,
        custom_params=custom_p,
        feature_space=fs,
        pca_components=pca_v,
        progress_cb=lambda p, m: print(f"[{p * 100:5.1f}%] {m}"),
    )
    print(f"\nSaved clustering bundle: {bundle.card.bundle_id}")
    print(json.dumps(bundle.card.metrics, indent=2))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    store = get_store()
    if args.input:
        df_in = load_uploaded_table(args.input)
        reg_b = store.get_deployed("regression")
        clf_b = store.get_deployed("classification")
        clu_b = store.get_deployed("clustering")
        scored = predict_batch_dataframe(
            df_in,
            reg_bundle_id=reg_b.card.bundle_id if reg_b and args.task in ("regression", "all") else None,
            clf_bundle_id=clf_b.card.bundle_id if clf_b and args.task in ("classification", "all") else None,
            clu_bundle_id=clu_b.card.bundle_id if clu_b and args.task == "all" else None,
            store=store,
        )
        if args.output:
            scored.to_csv(args.output, index=False)
            print(f"Wrote {len(scored):,} predictions to {args.output}")
        else:
            print(scored.head(15).to_string(index=False))
        return 0

    preset_name = args.preset or next(iter(BOOKING_PRESETS.keys()))
    booking = BOOKING_PRESETS.get(preset_name, next(iter(BOOKING_PRESETS.values())))
    res = predict_single_booking(booking, store=store)
    print(f"Preset booking: {preset_name}")
    if args.task in ("regression", "all") and res["regression"]:
        r = res["regression"]
        print(f"  Predicted ADR       : €{r['pred_adr']:.2f}  [{r['model_name']}]")
    if args.task in ("classification", "all") and res["classification"]:
        c = res["classification"]
        print(f"  Cancel Probability  : {c['cancel_proba']:.2%} -> {c['decision_label']} (thr={c['threshold']:.2f})  [{c['model_name']}]")
    if args.task == "all" and res["clustering"]:
        k = res["clustering"]
        print(f"  Assigned Segment    : Cluster {k['cluster']}  [{k['model_name']}]")
    return 0


def cmd_build_artifacts(args: argparse.Namespace) -> int:
    built = build_official_artifacts(
        core_only=args.core_only,
        run_grid_search=not args.fast,
        skip_existing=args.resume,
        progress_cb=lambda p, m: print(f"[{p * 100:5.1f}%] {m}", flush=True),
    )
    print(f"\nSuccessfully built {len(built)} official artifact bundles.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="City Hotel Booking Demand — CLI & ML Studio")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gui", help="Launch the interactive Streamlit ML Studio")
    g.add_argument("--port", type=int, default=8501)
    g.add_argument("--headless", action="store_true")
    g.set_defaults(func=cmd_gui)

    a = sub.add_parser("analyze", help="Run exploratory data analysis (writes outputs/eda/)")
    a.set_defaults(func=cmd_analyze)

    lm = sub.add_parser("list-models", help="List all registered models across tracks")
    lm.add_argument("--task", choices=["regression", "classification", "clustering"], default=None)
    lm.set_defaults(func=cmd_list_models)

    tr = sub.add_parser("train", help="Train a supervised regression or classification model")
    tr.add_argument("--task", choices=["regression", "classification"], required=True)
    tr.add_argument("--model", required=True, help="Model ID (e.g. linear_regression, random_forest, xgboost, gaussian_nb)")
    tr.add_argument("--mode", choices=["baseline", "tuned_preset"], default="baseline")
    tr.add_argument("--feature-space", choices=["original", "pca", "lda"], default="original")
    tr.add_argument("--pca", type=float, default=0.90)
    tr.add_argument("--data", default=None, help="Optional path to custom CSV/XLSX to train on")
    tr.set_defaults(func=cmd_train)

    tu = sub.add_parser("tune", help="Run CV hyperparameter optimization for a supervised model")
    tu.add_argument("--task", choices=["regression", "classification"], required=True)
    tu.add_argument("--model", required=True)
    tu.add_argument("--method", choices=["random", "grid"], default="random")
    tu.add_argument("--cv", type=int, default=5)
    tu.add_argument("--n-iter", type=int, default=10)
    tu.add_argument("--feature-space", choices=["original", "pca", "lda"], default="original")
    tu.set_defaults(func=cmd_tune)

    ev = sub.add_parser("evaluate", help="Evaluate saved models on val, test, or custom CSV")
    ev.add_argument("--task", choices=["regression", "classification"], required=True)
    ev.add_argument("--model", default="all")
    ev.add_argument("--split", choices=["val", "test"], default="val")
    ev.add_argument("--data", default=None)
    ev.set_defaults(func=cmd_evaluate)

    cl = sub.add_parser("cluster", help="Run or optimize an unsupervised clustering algorithm")
    cl.add_argument("--algorithm", default="kmeans", help="kmeans | gmm | agglomerative | dbscan")
    cl.add_argument("--mode", choices=["baseline", "tune"], default="baseline")
    cl.add_argument("--k", type=int, default=None)
    cl.add_argument("--pca", default="0.90", help="'none' or float variance ratio like 0.90")
    cl.set_defaults(func=cmd_cluster)

    pr = sub.add_parser("predict", help="Run saved inference pipelines on a preset or CSV file")
    pr.add_argument("--task", choices=["regression", "classification", "all"], default="all")
    pr.add_argument("--preset", choices=list(BOOKING_PRESETS.keys()), default=None)
    pr.add_argument("--input", default=None, help="Batch CSV/XLSX file to score")
    pr.add_argument("--output", default=None, help="Output CSV path for batch predictions")
    pr.set_defaults(func=cmd_predict)

    ba = sub.add_parser("build-artifacts", help="Build all official baseline, tuned, and PCA/LDA bundles")
    ba.add_argument("--fast", action="store_true", help="Use validated tuned presets instead of full CV search")
    ba.add_argument("--core-only", action="store_true", help="Build only the 3 Milestone 1 models")
    ba.add_argument("--resume", action="store_true", help="Skip bundles already present on disk")
    ba.set_defaults(func=cmd_build_artifacts)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
