"""Saved-artifact inference service: single booking, batch scoring, all-models comparison, and what-if sweeps."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.validation import normalize_columns
from src.registry.artifacts import ArtifactBundle, ArtifactStore, get_store


def _prepare_frame(data: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()
    norm, _, _ = normalize_columns(df)
    return norm


def predict_with_bundle(
    bundle: ArtifactBundle,
    df: pd.DataFrame,
    threshold_override: float | None = None,
) -> dict[str, Any]:
    """Run a single loaded ArtifactBundle on a prepared DataFrame without retraining."""
    pipe = bundle.load_pipeline()
    task = bundle.card.task

    if task == "regression":
        preds = np.asarray(pipe.predict(df), dtype=float)
        val_rmse = bundle.card.metrics.get("val", {}).get("RMSE")
        return {"pred_adr": preds, "val_rmse": float(val_rmse) if val_rmse is not None else None}

    if task == "classification":
        thr = float(threshold_override if threshold_override is not None else (bundle.card.threshold or 0.5))
        if hasattr(pipe, "predict_proba"):
            proba = np.asarray(pipe.predict_proba(df)[:, 1], dtype=float)
            preds = (proba >= thr).astype(int)
        else:
            preds = np.asarray(pipe.predict(df), dtype=int)
            proba = preds.astype(float)
        return {"pred_canceled": preds, "cancel_proba": proba, "threshold": thr}

    if not bundle.card.supports_predict_new:
        raise ValueError(f"Clustering algorithm '{bundle.card.display_name}' does not support out-of-sample predict().")
    labels = np.asarray(pipe.predict(df), dtype=int)
    personas = bundle.read_json("personas.json") or []
    persona_map = {int(p["cluster"]): p for p in personas if "cluster" in p}
    return {"cluster": labels, "persona_map": persona_map}


def predict_single_booking(
    booking: dict[str, Any],
    reg_bundle_id: str | None = None,
    clf_bundle_id: str | None = None,
    clu_bundle_id: str | None = None,
    threshold_override: float | None = None,
    store: ArtifactStore | None = None,
) -> dict[str, Any]:
    """Predict ADR, cancellation risk, and guest segment for one booking using saved artifacts."""
    store = store or get_store()
    df = _prepare_frame(booking)
    out: dict[str, Any] = {"input": df.iloc[0].to_dict(), "regression": None, "classification": None, "clustering": None}

    reg_b = store.get_bundle(reg_bundle_id) if reg_bundle_id else store.get_deployed("regression")
    if reg_b is not None:
        r = predict_with_bundle(reg_b, df)
        adr_val = float(r["pred_adr"][0])
        # Populate ADR for clustering if user left it unset
        if "ADR" not in df.columns or pd.isna(df["ADR"].iloc[0]):
            df["ADR"] = adr_val
        out["regression"] = {
            "bundle_id": reg_b.card.bundle_id,
            "model_name": f"{reg_b.card.display_name} ({reg_b.card.variant})",
            "pred_adr": adr_val,
            "val_rmse": r["val_rmse"],
            "interval_low": max(0.0, adr_val - r["val_rmse"]) if r["val_rmse"] else None,
            "interval_high": (adr_val + r["val_rmse"]) if r["val_rmse"] else None,
        }

    clf_b = store.get_bundle(clf_bundle_id) if clf_bundle_id else store.get_deployed("classification")
    if clf_b is not None:
        c = predict_with_bundle(clf_b, df, threshold_override=threshold_override)
        prob = float(c["cancel_proba"][0])
        lbl = int(c["pred_canceled"][0])
        out["classification"] = {
            "bundle_id": clf_b.card.bundle_id,
            "model_name": f"{clf_b.card.display_name} ({clf_b.card.variant})",
            "cancel_proba": prob,
            "threshold": c["threshold"],
            "pred_canceled": lbl,
            "decision_label": "Likely to Cancel" if lbl == 1 else "Likely to Check-In",
        }

    clu_b = store.get_bundle(clu_bundle_id) if clu_bundle_id else store.get_deployed("clustering")
    if clu_b is not None and clu_b.card.supports_predict_new:
        try:
            k = predict_with_bundle(clu_b, df)
            cid = int(k["cluster"][0])
            out["clustering"] = {
                "bundle_id": clu_b.card.bundle_id,
                "model_name": f"{clu_b.card.display_name} ({clu_b.card.variant})",
                "cluster": cid,
                "persona": k["persona_map"].get(cid),
            }
        except Exception:
            out["clustering"] = None

    return out


def predict_across_all_models(
    booking: dict[str, Any],
    task: str,
    threshold_override: float | None = None,
    store: ArtifactStore | None = None,
) -> pd.DataFrame:
    """Score the same single booking across every available bundle of a task for side-by-side consensus."""
    store = store or get_store()
    df = _prepare_frame(booking)
    rows: list[dict[str, Any]] = []
    for b in store.list_bundles(task=task):
        try:
            res = predict_with_bundle(b, df, threshold_override=threshold_override)
        except Exception:
            continue
        if task == "regression":
            rows.append(
                {
                    "Model": b.card.display_name,
                    "Variant": b.card.variant,
                    "Feature Space": b.card.feature_space,
                    "Source": b.card.source,
                    "Deployed": "Active" if b.card.is_deployed else "",
                    "Predicted ADR (EUR)": round(float(res["pred_adr"][0]), 2),
                    "Val RMSE (EUR)": round(float(res["val_rmse"]), 2) if res["val_rmse"] else None,
                }
            )
        elif task == "classification":
            rows.append(
                {
                    "Model": b.card.display_name,
                    "Variant": b.card.variant,
                    "Feature Space": b.card.feature_space,
                    "Source": b.card.source,
                    "Deployed": "Active" if b.card.is_deployed else "",
                    "Cancel Prob": round(float(res["cancel_proba"][0]), 4),
                    "Threshold": round(float(res["threshold"]), 2),
                    "Prediction": "Cancel (1)" if int(res["pred_canceled"][0]) == 1 else "Stay (0)",
                }
            )
        elif task == "clustering" and b.card.supports_predict_new:
            cid = int(res["cluster"][0])
            p = res["persona_map"].get(cid, {})
            rows.append(
                {
                    "Model": b.card.display_name,
                    "Variant": b.card.variant,
                    "Feature Space": b.card.feature_space,
                    "Assigned Cluster": cid,
                    "Segment Persona": p.get("title", f"Cluster {cid}"),
                }
            )
    return pd.DataFrame(rows)


def predict_batch_dataframe(
    df: pd.DataFrame,
    reg_bundle_id: str | None = None,
    clf_bundle_id: str | None = None,
    clu_bundle_id: str | None = None,
    threshold_override: float | None = None,
    store: ArtifactStore | None = None,
) -> pd.DataFrame:
    """Append batch predictions (`pred_ADR`, `p_IsCanceled`, `pred_IsCanceled`, `pred_Cluster`) to a DataFrame."""
    store = store or get_store()
    norm = _prepare_frame(df)
    out = df.copy()

    if reg_bundle_id:
        rb = store.get_bundle(reg_bundle_id)
        r = predict_with_bundle(rb, norm)
        out["pred_ADR"] = np.round(r["pred_adr"], 2)
        if "ADR" not in norm.columns:
            norm["ADR"] = out["pred_ADR"]

    if clf_bundle_id:
        cb = store.get_bundle(clf_bundle_id)
        c = predict_with_bundle(cb, norm, threshold_override=threshold_override)
        out["p_IsCanceled"] = np.round(c["cancel_proba"], 4)
        out["pred_IsCanceled"] = c["pred_canceled"]

    if clu_bundle_id:
        kb = store.get_bundle(clu_bundle_id)
        if kb.card.supports_predict_new:
            k = predict_with_bundle(kb, norm)
            out["pred_Cluster"] = k["cluster"]

    return out


def what_if_sweep(
    booking: dict[str, Any],
    feature_name: str,
    values: list[Any],
    reg_bundle_id: str | None = None,
    clf_bundle_id: str | None = None,
    store: ArtifactStore | None = None,
) -> pd.DataFrame:
    """Vary a single input feature across `values` while holding all other booking fields fixed."""
    store = store or get_store()
    rows = []
    for v in values:
        cand = dict(booking)
        cand[feature_name] = v
        rows.append(cand)
    grid_df = _prepare_frame(pd.DataFrame(rows))
    res_df = pd.DataFrame({feature_name: values})

    rb = store.get_bundle(reg_bundle_id) if reg_bundle_id else store.get_deployed("regression")
    if rb is not None:
        try:
            res_df["Predicted ADR (€)"] = predict_with_bundle(rb, grid_df)["pred_adr"]
        except Exception:
            pass

    cb = store.get_bundle(clf_bundle_id) if clf_bundle_id else store.get_deployed("classification")
    if cb is not None:
        try:
            res_df["Cancel Probability"] = predict_with_bundle(cb, grid_df)["cancel_proba"]
        except Exception:
            pass
    return res_df
