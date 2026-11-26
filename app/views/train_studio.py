"""Train Studio page — guided wizard to train or tune any registered model on uploaded user data or H2 splits (emoji-free)."""
from __future__ import annotations

import streamlit as st

from app.cache import get_cached_runner
from app.components.job_monitor import render_live_job_monitor
from app.components.param_editor import render_param_editor, render_search_space_editor
from app.components.style import family_pill_html, glass_container, page_header, pill_html, run_page_safely, section_header
from app.components.uploader import render_data_uploader
from src.registry.registry import get_spec, list_specs
from src.training.service import (
    prepare_user_splits,
    train_clustering_model,
    train_supervised_model,
)


def _render() -> None:
    page_header(
        "Train Studio — Custom & Project Model Training",
        "Upload your own booking CSV/XLSX (or use official H2 splits), configure hyperparameters or CV search, and persist reusable model bundles",
        pill_html("Persisted to artifacts/user/", "user"),
    )

    section_header(
        "Choose Task & Model Architecture",
        "Select the analytical objective, registered estimator family, and feature representation.",
        kicker="Step 1 of 4",
    )
    with glass_container("reeded"):
        s1, s2, s3 = st.columns([1.3, 1.8, 1.3])
        with s1:
            task = st.selectbox(
                "Analytical Task",
                options=["regression", "classification", "clustering"],
                format_func=lambda t: {
                    "regression": "Task A — ADR Regression",
                    "classification": "Task B — Cancellation Classification",
                    "clustering": "Task C — Unsupervised Clustering",
                }[t],
                key="ts_task",
            )
        specs = list_specs(task)
        spec_map = {f"{s.display_name} [{s.family}]": s for s in specs}
        with s2:
            chosen_lbl = st.selectbox("Model Architecture", options=list(spec_map.keys()), key="ts_model")
            spec = get_spec(task, spec_map[chosen_lbl].model_id)
        with s3:
            fs_opts = ["original"]
            if spec.supports_pca:
                fs_opts.append("pca")
            if spec.supports_lda:
                fs_opts.append("lda")
            default_fs = "pca" if (task == "clustering" and "pca" in fs_opts) else "original"
            feature_space = st.selectbox(
                "Feature Representation",
                options=fs_opts,
                index=fs_opts.index(default_fs),
                format_func=lambda f: {"original": "Original Engineered Space", "pca": "PCA Reduced Space", "lda": "LDA Supervised Projection"}[f],
                key="ts_fs",
            )

        st.markdown(
            f"{family_pill_html(spec.family)} **{spec.display_name}** — {spec.description} *(Cost profile: `{spec.cost_hint}`)*",
            unsafe_allow_html=True,
        )

        pca_var = 0.90
        if feature_space == "pca":
            pca_var = st.slider("PCA Explained Variance Retention", min_value=0.70, max_value=0.99, value=0.90, step=0.05)

    section_header(
        "Select Training Data Source",
        "Train on an uploaded dataset (partitioned 70/15/15 automatically) or on the official H2 splits.",
        kicker="Step 2 of 4",
    )
    with glass_container("frosted"):
        data_source = st.radio(
            "Dataset Source",
            options=[
                "Upload My Own Dataset (CSV / XLSX) — saved to User Bundles",
                "City Hotel H2 Official Splits — saved to User Bundles (safe sandbox) or Official Project Store",
            ],
            horizontal=True,
            key="ts_datasource",
        )

        user_df = None
        val_rep = None
        ds_label = "H2 City Hotel (official splits)"
        apply_clean = True
        save_target = "user"

        if data_source.startswith("Upload"):
            user_df, val_rep, ds_label = render_data_uploader(
                key_prefix=f"ts_up_{task}",
                task=task,
                required_inputs=spec.required_inputs,
                forbidden_inputs=spec.forbidden_inputs,
                require_target=(task != "clustering"),
                min_rows=60 if task != "clustering" else 30,
            )
            c_cl1, c_cl2 = st.columns(2)
            apply_clean = c_cl1.checkbox(
                "Apply stateless structural cleaning (`clean_data`: deduplicate, filter 0-guest / invalid ADR)",
                value=True,
            )
            c_cl2.caption("Uploaded datasets are automatically partitioned 70% Train / 15% Validation / 15% Holdout Test (`seed=42`).")
        else:
            ns_choice = st.radio(
                "Save Destination",
                options=["User Namespace (`artifacts/user/` — keeps official artifacts untouched)", "Official Project Store (`artifacts/<task>/`)"],
                horizontal=True,
            )
            save_target = "project" if ns_choice.startswith("Official") else "user"

    section_header(
        "Hyperparameters or Automated Optimization",
        "Run the default baseline configuration, adjust parameters manually, or launch a cross-validated search.",
        kicker="Step 3 of 4",
    )
    with glass_container("solid"):
        cfg_mode = st.radio(
            "Training Configuration Mode",
            options=["baseline", "custom", "tune"],
            format_func=lambda m: {
                "baseline": "Baseline (Default Untuned Configuration)",
                "custom": "Custom Hyperparameters (Interactive Controls)",
                "tune": "Hyperparameter Optimization (Cross-Validation Search)",
            }[m],
            horizontal=True,
            key="ts_mode",
        )

        custom_params = None
        edited_space = None
        search_method = "random"
        cv_folds = 5
        n_iter = 10

        if cfg_mode == "baseline":
            param_rows = [{"Parameter": k, "Default Value": str(v)} for k, v in spec.baseline_params.items()]
            st.dataframe(param_rows, width="stretch", hide_index=True)
        elif cfg_mode == "custom":
            custom_params = render_param_editor(spec, key_prefix="ts_pe", preset_values=spec.tuned_params)
        else:
            edited_space = render_search_space_editor(spec, key_prefix="ts_sse")
            if task != "clustering":
                t1, t2, t3 = st.columns(3)
                search_method = t1.selectbox("Search Strategy", options=["random", "grid"], format_func=lambda s: "RandomizedSearchCV" if s == "random" else "Exhaustive GridSearchCV")
                cv_folds = int(t2.slider("CV Folds (Train Split Only)", min_value=3, max_value=10, value=5))
                n_iter = int(t3.slider("Max Candidate Trials (n_iter)", min_value=4, max_value=30, value=10))

    section_header(
        "Execution Controls & Run Metadata",
        "Tag this run, optionally cap training rows for rapid iteration, and submit to the background job runner.",
        kicker="Step 4 of 4",
    )
    with glass_container("reeded"):
        r1, r2, r3 = st.columns([2.2, 1.3, 1.1], vertical_alignment="bottom")
        default_run_name = f"{spec.display_name} ({cfg_mode}, {ds_label})"
        run_name = r1.text_input("Run Label", value=default_run_name)
        default_cap = spec.recommended_max_rows or 0
        cap_rows = r2.number_input(
            "Optional Max Training Rows (0 = full split)",
            min_value=0, max_value=100_000, value=int(default_cap), step=1000,
        )
        eval_holdout = r3.checkbox("Score Holdout Test split", value=True)

        can_launch = (not data_source.startswith("Upload")) or (user_df is not None and val_rep is not None and val_rep.ok)
        if st.button("Launch Training Job", type="primary", icon=":material/play_arrow:", disabled=not can_launch, width="stretch"):
            runner = get_cached_runner()
            eff_cap = int(cap_rows) if cap_rows > 0 else None

            if task in ("regression", "classification"):
                splits_arg = None
                if data_source.startswith("Upload") and user_df is not None:
                    tr, va, te, _ = prepare_user_splits(user_df, task=task, apply_cleaning=apply_clean)
                    splits_arg = (tr, va, te if eval_holdout else None)

                job = runner.submit(
                    title=run_name,
                    kind="tune" if cfg_mode == "tune" else "train",
                    fn=train_supervised_model,
                    kwargs=dict(
                        task=task,
                        model_id=spec.model_id,
                        mode=cfg_mode,
                        custom_params=custom_params,
                        search_space=edited_space,
                        search_method=search_method,
                        cv_folds=cv_folds,
                        n_iter=n_iter,
                        feature_space=feature_space,
                        pca_components=pca_var,
                        tune_threshold=(task == "classification" and cfg_mode != "baseline"),
                        evaluate_test=eval_holdout,
                        splits=splits_arg,
                        dataset_name=ds_label,
                        source=save_target,
                        run_name=run_name,
                        sample_max_rows=eff_cap,
                    ),
                )
            else:
                job = runner.submit(
                    title=run_name,
                    kind="cluster",
                    fn=train_clustering_model,
                    kwargs=dict(
                        model_id=spec.model_id,
                        mode=cfg_mode,
                        custom_params=custom_params,
                        search_space=edited_space,
                        feature_space=feature_space if feature_space in ("pca", "original") else "pca",
                        pca_components=pca_var,
                        df=user_df if data_source.startswith("Upload") else None,
                        dataset_name=ds_label,
                        source=save_target,
                        run_name=run_name,
                        sample_max_rows=eff_cap,
                    ),
                )
            st.session_state["active_job_id"] = job.job_id

    active_job = st.session_state.get("active_job_id")
    if active_job:
        section_header("Live Training Job Status", kicker="Background Worker")
        render_live_job_monitor(active_job)


run_page_safely(_render)
