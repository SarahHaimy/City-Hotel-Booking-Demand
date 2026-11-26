"""Registered clustering algorithm adapter wrapping the implemented K-Means + PCA pipeline.

When teammates implement additional Milestone 2 clustering algorithms, register them here
following `docs/MODEL_CONTRACT.md` and they will appear across the GUI automatically.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.clustering.k_means.features import FEATURE_SET, FORBIDDEN
from src.clustering.k_means.preprocessing import build_preprocessor
from src.config import SEED
from src.registry.registry import register_model
from src.registry.specs import FeatureSpace, ModelOptionSpec, ModelSpec, ParamSpec

CLUSTER_RAW_INPUTS = [
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "ADR",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
    "IsRepeatedGuest",
    "MarketSegment",
    "CustomerType",
    "Meal",
]


def engineer_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic row-wise feature builder matching `src/clustering/k_means/features.py`.

    Accepts raw H2 frames (`Babies`), cleaned frames (`HasBabies`), or pre-engineered frames.
    """
    out = df.copy()
    if "HasBabies" in out.columns:
        has_babies = pd.to_numeric(out["HasBabies"], errors="coerce").fillna(0)
    elif "Babies" in out.columns:
        has_babies = (pd.to_numeric(out["Babies"], errors="coerce").fillna(0) > 0).astype(int)
    else:
        has_babies = pd.Series(0, index=out.index)

    children = pd.to_numeric(out["Children"], errors="coerce").fillna(0)
    if "TotalNights" not in out.columns:
        wknd = pd.to_numeric(out["StaysInWeekendNights"], errors="coerce").fillna(0)
        wk = pd.to_numeric(out["StaysInWeekNights"], errors="coerce").fillna(0)
        out["TotalNights"] = (wknd + wk).clip(lower=0)
    if "TotalGuests" not in out.columns:
        adults = pd.to_numeric(out["Adults"], errors="coerce").fillna(1)
        out["TotalGuests"] = (adults + children + has_babies).clip(lower=1)
    if "HasChildren" not in out.columns:
        out["HasChildren"] = ((children + has_babies) > 0).astype(int)
    if "HasParking" not in out.columns:
        park = pd.to_numeric(out.get("RequiredCarParkingSpaces", 0), errors="coerce").fillna(0)
        out["HasParking"] = (park > 0).astype(int)

    for col in ("LeadTime", "ADR", "TotalOfSpecialRequests"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).clip(lower=0)
    out["IsRepeatedGuest"] = (pd.to_numeric(out["IsRepeatedGuest"], errors="coerce").fillna(0) > 0).astype(int)

    for col in FEATURE_SET["cat"]:
        s = out[col].astype("string").str.strip()
        is_na = s.isna() | s.str.upper().isin(["NULL", "NAN", ""])
        out[col] = s.mask(is_na, "Unknown").astype(object)

    cols = FEATURE_SET["num"] + FEATURE_SET["bin"] + FEATURE_SET["cat"]
    leaked = FORBIDDEN & set(cols)
    if leaked:
        raise AssertionError(f"Forbidden leakage columns in clustering features: {leaked}")
    return out[cols]


def build_kmeans_pipeline(
    params: dict[str, Any],
    feature_space: FeatureSpace = "pca",
    n_components: float | int | None = 0.90,
) -> Pipeline:
    est = KMeans(
        n_clusters=int(params.get("n_clusters", 3)),
        init=str(params.get("init", "k-means++")),
        n_init=int(params.get("n_init", 10)),
        max_iter=int(params.get("max_iter", 300)),
        algorithm=str(params.get("algorithm", "lloyd")),
        random_state=SEED,
    )
    steps: list[tuple[str, Any]] = [
        ("engineer", FunctionTransformer(engineer_cluster_features, validate=False)),
        ("prep", build_preprocessor(FEATURE_SET)),
    ]
    if feature_space == "pca" and n_components is not None:
        steps.append(("pca", PCA(n_components=n_components, random_state=SEED)))
    steps.append(("clusterer", est))
    return Pipeline(steps)


def describe_cluster_personas(
    df_profile: pd.DataFrame,
    df_cancel: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Derive human-readable guest segment personas from empirical cluster centroids."""
    if df_profile is None or df_profile.empty:
        return []
    cancel_map: dict[int, float] = {}
    if df_cancel is not None and not df_cancel.empty:
        c_col = "cluster" if "cluster" in df_cancel.columns else df_cancel.columns[0]
        m_col = "mean" if "mean" in df_cancel.columns else "cancel_rate"
        if m_col in df_cancel.columns:
            cancel_map = {int(r[c_col]): float(r[m_col]) for _, r in df_cancel.iterrows()}

    med_lead = float(df_profile["LeadTime"].median()) if "LeadTime" in df_profile else 50.0
    med_adr = float(df_profile["ADR"].median()) if "ADR" in df_profile else 105.0
    med_nights = float(df_profile["TotalNights"].median()) if "TotalNights" in df_profile else 2.5

    personas: list[dict[str, Any]] = []
    for idx, row in df_profile.iterrows():
        cid = int(row["cluster"]) if "cluster" in df_profile.columns else int(idx)
        lead = float(row.get("LeadTime", 0))
        adr = float(row.get("ADR", 0))
        nights = float(row.get("TotalNights", 0))
        guests = float(row.get("TotalGuests", 0))
        req = float(row.get("TotalOfSpecialRequests", 0))
        kids = float(row.get("HasChildren", 0))
        rep = float(row.get("IsRepeatedGuest", 0))
        pct = float(row.get("size_pct", 0))
        seg = str(row.get("MarketSegment_mode", "Online TA"))

        tags = []
        tags.append("Short-Lead" if lead < max(25.0, med_lead * 0.55) else ("Long-Lead" if lead > med_lead * 1.1 else "Mid-Lead"))
        tags.append("Premium-ADR" if adr > med_adr * 1.12 else ("Budget-Conscious" if adr < med_adr * 0.92 else "Standard-Rate"))
        if kids >= 0.15 or guests >= 2.25:
            tags.append("Family / Leisure Party")
        elif rep >= 0.08 or nights <= 2.0:
            tags.append("Short-Stay Business / Repeat")
        elif nights > med_nights * 1.15:
            tags.append("Extended-Stay Tourists")
        else:
            tags.append(f"{seg} Travelers")

        title = " · ".join(tags[:3])
        summary = (
            f"{pct:.1f}% of bookings — avg lead time {lead:.0f} days, {nights:.1f} nights, "
            f"{guests:.2f} guests, €{adr:.0f}/night ADR, {req:.2f} special requests (primary channel: {seg})."
        )
        c_rate = cancel_map.get(cid)
        personas.append(
            {
                "cluster": cid,
                "title": title,
                "size": int(row.get("size", 0)),
                "size_pct": pct,
                "posthoc_cancel_rate": c_rate,
                "summary": summary,
            }
        )
    return personas


register_model(
    ModelSpec(
        model_id="kmeans",
        task="clustering",
        display_name="K-Means Clustering",
        family="centroid",
        description=(
            "Implemented unsupervised segmentation pipeline (`src/clustering/k_means/`) integrating "
            "`log1p` scaling, `StandardScaler`, `OneHotEncoder(min_frequency=0.01)`, optional 90% variance "
            "PCA (7 components), and Lloyd's K-Means."
        ),
        assumptions_and_pros=(
            "Scales linearly to all ~52k cleaned City Hotel bookings and supports instant out-of-sample "
            "segment assignment on new reservations."
        ),
        required_inputs=list(CLUSTER_RAW_INPUTS),
        forbidden_inputs=sorted(FORBIDDEN),
        params=[
            ParamSpec("n_clusters", "Number of Clusters (k)", "int", 3, low=2, high=10, step=1),
            ParamSpec("init", "Initialization", "choice", "k-means++", choices=("k-means++", "random")),
            ParamSpec("n_init", "Restarts (n_init)", "int", 10, low=3, high=30, step=1),
            ParamSpec("max_iter", "Max Iterations", "int", 300, low=100, high=1000, step=50),
        ],
        baseline_params={"n_clusters": 3, "init": "k-means++", "n_init": 10, "max_iter": 300},
        tuned_params={"n_clusters": 4, "init": "k-means++", "n_init": 15, "max_iter": 300},
        search_space={"n_clusters": [2, 3, 4, 5, 6, 7, 8]},
        default_scoring="silhouette",
        build_pipeline=build_kmeans_pipeline,
        supports_predict_new=True,
        supports_pca=True,
        cost_hint="fast",
        track_origin="milestone_1",
        supported_actions=("cluster",),
        cli_options=[
            ModelOptionSpec(
                flag="--init",
                dest="init",
                kind="str",
                default="k-means++",
                choices=("k-means++", "random"),
                help="K-Means centroid initialization strategy.",
            ),
            ModelOptionSpec(
                flag="--n-init",
                dest="n_init",
                kind="int",
                default=10,
                help="Number of K-Means centroid initializations.",
            ),
        ],
    )
)
