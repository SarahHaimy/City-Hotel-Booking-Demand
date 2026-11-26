"""Fluted Glass UI design system, reusable glass surface helpers, and layout primitives.

Implements the "Travertine & Smoked Bronze" architectural fluted-glass aesthetic across
every Streamlit screen while preserving 100% of existing application functionality.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import traceback
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import pandas as pd
import streamlit as st

from app.components.tokens import FONT_IMPORT_URL, css_variables
from src.evaluation.figures import family_color

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

GlassVariant = Literal["frosted", "reeded", "solid"]

PRETTY_COLUMN_LABELS: dict[str, str] = {
    "display_name": "Model",
    "model_id": "Model ID",
    "family": "Family",
    "variant": "Variant",
    "feature_space": "Feature Space",
    "source": "Source",
    "is_deployed": "Deployed",
    "train_time_s": "Train Time (s)",
    "n_train_rows": "Train Rows",
    "pca_components": "PCA Components",
    "pca_variance_kept": "Variance Retained",
    "n_features_out": "Output Dims",
    "n_clusters": "Clusters (k)",
    "silhouette": "Silhouette",
    "davies_bouldin": "Davies-Bouldin",
    "calinski_harabasz": "Calinski-Harabasz",
    "noise_pct": "Noise %",
    "min_cluster_pct": "Smallest Cluster %",
    "ROC_AUC": "ROC-AUC",
    "PR_AUC": "PR-AUC",
    "MedAE": "MedAE",
    "MAPE": "MAPE (%)",
    "Brier": "Brier Score",
    "LogLoss": "Log Loss",
    "variable": "Variable",
    "regression": "ADR Regression",
    "classification": "Cancellation Classifier",
    "clustering": "Guest Clustering",
    "rationale": "Timing & Leakage Justification",
}


def _build_global_css() -> str:
    return f"""
<style>
  @import url('{FONT_IMPORT_URL}');

  {css_variables()}

  /* =========================================================================
     1. Base Reset, Scrollbars & Architectural Travertine Canvas
     ========================================================================= */
  ::-webkit-scrollbar {{
      width: 8px;
      height: 8px;
  }}
  ::-webkit-scrollbar-track {{
      background: rgba(221, 213, 199, 0.28);
  }}
  ::-webkit-scrollbar-thumb {{
      background: rgba(138, 128, 112, 0.42);
      border-radius: var(--fg-r-pill);
      border: 2px solid transparent;
      background-clip: padding-box;
  }}
  ::-webkit-scrollbar-thumb:hover {{
      background: rgba(110, 82, 41, 0.62);
      border: 2px solid transparent;
      background-clip: padding-box;
  }}

  html, body, [class*="css"] {{
      font-family: var(--fg-font-sans);
      color: var(--fg-ink-800);
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      text-rendering: optimizeLegibility;
  }}

  /* Sunlit travertine interior canvas with soft upper-right daylight wash */
  .stApp {{
      background-color: var(--fg-canvas);
      background-image:
          radial-gradient(1150px 640px at 90% -8%, rgba(255, 253, 248, 0.92) 0%, rgba(255, 253, 248, 0.0) 68%),
          radial-gradient(920px 580px at 6% 104%, rgba(217, 194, 154, 0.18) 0%, rgba(217, 194, 154, 0.0) 70%),
          linear-gradient(180deg, #F7F3EC 0%, #F2EDE4 52%, #EDE7DC 100%);
      background-attachment: fixed;
  }}

  /* Subtle ambient fluted architectural frieze at the very top of the viewport */
  .stApp::before {{
      content: "";
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      height: 4px;
      background-image:
          linear-gradient(90deg, rgba(165, 132, 79, 0.55), rgba(217, 194, 154, 0.85), rgba(138, 106, 59, 0.55)),
          var(--fg-flute-champagne);
      z-index: 999;
      pointer-events: none;
      opacity: 0.85;
  }}

  header[data-testid="stHeader"] {{
      background: transparent !important;
      backdrop-filter: none !important;
      border-bottom: none !important;
      box-shadow: none !important;
      height: 2.1rem !important;
      pointer-events: none;
  }}
  header[data-testid="stHeader"] button,
  header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] {{
      pointer-events: auto;
      color: var(--fg-ink-700) !important;
      background: rgba(251, 249, 245, 0.78) !important;
      backdrop-filter: var(--fg-blur-soft) !important;
      border: 1px solid rgba(165, 132, 79, 0.24) !important;
      border-radius: var(--fg-r-sm) !important;
      box-shadow: var(--fg-shadow-rest) !important;
      transition: all var(--fg-fast) var(--fg-ease) !important;
  }}
  header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"]:hover {{
      border-color: var(--fg-bronze-500) !important;
      background: var(--fg-stone-50) !important;
  }}
  div[data-testid="stDecoration"],
  div[data-testid="stToolbar"] {{
      display: none !important;
  }}

  /* Main content frame & screen-entry choreography */
  @keyframes fg-pane-enter {{
      from {{
          opacity: 0;
          transform: translateY(5px);
      }}
      to {{
          opacity: 1;
          transform: translateY(0);
      }}
  }}

  .block-container,
  div[data-testid="stMainBlockContainer"] {{
      padding-top: 1.15rem !important;
      padding-bottom: 3.4rem !important;
      padding-left: 2.5rem !important;
      padding-right: 2.5rem !important;
      max-width: 1420px !important;
      animation: fg-pane-enter 340ms var(--fg-ease-out) both;
  }}
  @media (max-width: 768px) {{
      .block-container,
      div[data-testid="stMainBlockContainer"] {{
          padding-left: 1.05rem !important;
          padding-right: 1.05rem !important;
          padding-top: 0.95rem !important;
      }}
  }}

  /* =========================================================================
     2. Editorial & Technical Typography Scale
     ========================================================================= */
  [data-testid="stMarkdownContainer"] h1,
  [data-testid="stMarkdownContainer"] h2 {{
      font-family: var(--fg-font-display) !important;
      font-weight: 600 !important;
      letter-spacing: -0.022em !important;
      color: var(--fg-ink-900) !important;
  }}
  [data-testid="stMarkdownContainer"] h3 {{
      font-family: var(--fg-font-display) !important;
      font-size: 1.20rem !important;
      font-weight: 600 !important;
      letter-spacing: -0.018em !important;
      color: var(--fg-ink-900) !important;
      padding-top: 0.35rem !important;
      margin-bottom: 0.35rem !important;
      line-height: 1.28 !important;
  }}
  [data-testid="stMarkdownContainer"] h4 {{
      font-family: var(--fg-font-display) !important;
      font-size: 1.04rem !important;
      font-weight: 600 !important;
      letter-spacing: -0.014em !important;
      color: var(--fg-ink-900) !important;
      padding-top: 0.25rem !important;
      margin-bottom: 0.25rem !important;
  }}
  [data-testid="stMarkdownContainer"] h5 {{
      font-family: var(--fg-font-sans) !important;
      font-size: 0.84rem !important;
      font-weight: 600 !important;
      letter-spacing: 0.02em !important;
      color: var(--fg-ink-700) !important;
      padding-top: 0.2rem !important;
      margin-bottom: 0.38rem !important;
  }}
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li {{
      line-height: 1.62 !important;
      font-size: var(--fg-text-body);
      color: var(--fg-ink-600);
  }}
  [data-testid="stMarkdownContainer"] strong {{
      color: var(--fg-ink-900);
      font-weight: 600;
  }}
  [data-testid="stMarkdownContainer"] code {{
      font-family: var(--fg-font-mono) !important;
      font-size: 0.80em !important;
      background: rgba(234, 228, 217, 0.72) !important;
      color: var(--fg-bronze-700) !important;
      padding: 0.15em 0.46em !important;
      border-radius: 5px !important;
      border: 1px solid rgba(165, 132, 79, 0.22);
  }}
  [data-testid="stCaptionContainer"] p {{
      color: var(--fg-ink-500) !important;
      font-size: 0.80rem !important;
      line-height: 1.52 !important;
  }}
  [data-testid="stWidgetLabel"] p {{
      font-family: var(--fg-font-sans) !important;
      font-size: 0.79rem !important;
      font-weight: 600 !important;
      letter-spacing: 0.015em !important;
      color: var(--fg-ink-700) !important;
  }}
  hr {{
      border: none !important;
      height: 1px !important;
      background: linear-gradient(
          90deg,
          rgba(165, 132, 79, 0.30) 0%,
          rgba(201, 191, 174, 0.65) 45%,
          rgba(201, 191, 174, 0.12) 100%
      ) !important;
      margin-top: 1.45rem !important;
      margin-bottom: 1.45rem !important;
  }}

  /* =========================================================================
     3. Smoked Graphite Fluted-Glass Sidebar & Architectural Navigation
     ========================================================================= */
  section[data-testid="stSidebar"] {{
      background-color: var(--fg-graphite-900) !important;
      background-image:
          linear-gradient(180deg, rgba(38, 35, 31, 0.90) 0%, rgba(23, 21, 19, 0.95) 100%),
          var(--fg-flute-dark) !important;
      border-right: 1px solid rgba(216, 192, 148, 0.20) !important;
      box-shadow:
          inset -1px 0 0 rgba(255, 248, 232, 0.06),
          10px 0 34px -16px rgba(18, 16, 13, 0.46) !important;
  }}
  section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
      background: transparent !important;
  }}
  section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button {{
      color: var(--fg-on-dark-muted) !important;
      border-radius: var(--fg-r-xs) !important;
      transition: color var(--fg-fast) var(--fg-ease), background-color var(--fg-fast) var(--fg-ease) !important;
  }}
  section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover {{
      color: var(--fg-champagne-on-dark) !important;
      background: rgba(255, 248, 232, 0.07) !important;
  }}

  /* Sidebar navigation section headers */
  [data-testid="stNavSectionHeader"],
  [data-testid="stSidebarNavSeparator"] {{
      font-family: var(--fg-font-mono) !important;
      font-size: 0.64rem !important;
      font-weight: 500 !important;
      text-transform: uppercase !important;
      letter-spacing: 0.13em !important;
      color: var(--fg-on-dark-muted) !important;
      padding-top: 0.82rem !important;
      padding-bottom: 0.25rem !important;
      opacity: 0.86;
  }}
  [data-testid="stNavSectionHeader"] *,
  [data-testid="stSidebarNavSeparator"] * {{
      font-family: var(--fg-font-mono) !important;
      font-size: 0.64rem !important;
      letter-spacing: 0.13em !important;
      text-transform: uppercase !important;
      color: var(--fg-on-dark-muted) !important;
  }}

  /* Sidebar navigation links */
  a[data-testid="stSidebarNavLink"] {{
      position: relative;
      border-radius: var(--fg-r-sm) !important;
      padding: 0.46rem 0.74rem !important;
      margin: 2px 0 !important;
      border: 1px solid transparent !important;
      background: transparent !important;
      transition:
          background-color var(--fg-fast) var(--fg-ease),
          border-color var(--fg-fast) var(--fg-ease),
          transform var(--fg-fast) var(--fg-ease),
          box-shadow var(--fg-fast) var(--fg-ease) !important;
  }}
  a[data-testid="stSidebarNavLink"] span {{
      color: #CFC7B8 !important;
      font-size: 0.855rem !important;
      font-weight: 500 !important;
      letter-spacing: 0.005em !important;
      transition: color var(--fg-fast) var(--fg-ease) !important;
  }}
  a[data-testid="stSidebarNavLink"]:hover {{
      background: rgba(255, 249, 238, 0.06) !important;
      border-color: rgba(216, 192, 148, 0.15) !important;
      transform: translateX(2px);
  }}
  a[data-testid="stSidebarNavLink"]:hover span {{
      color: #F5EFE4 !important;
  }}
  a[data-testid="stSidebarNavLink"][aria-current="page"] {{
      background:
          linear-gradient(90deg, rgba(216, 192, 148, 0.18) 0%, rgba(255, 249, 238, 0.06) 100%) !important;
      border: 1px solid rgba(216, 192, 148, 0.30) !important;
      box-shadow:
          inset 2.5px 0 0 var(--fg-champagne-on-dark),
          inset 0 1px 0 rgba(255, 250, 240, 0.12),
          0 6px 16px -8px rgba(0, 0, 0, 0.55) !important;
  }}
  a[data-testid="stSidebarNavLink"][aria-current="page"] span {{
      color: #FAF5EC !important;
      font-weight: 600 !important;
  }}

  /* Sidebar captions, code, expanders & buttons on dark smoked glass */
  section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
  section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
      color: var(--fg-on-dark-muted) !important;
  }}
  section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {{
      color: var(--fg-on-dark) !important;
  }}
  section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] code {{
      background: rgba(255, 248, 232, 0.08) !important;
      color: var(--fg-champagne-on-dark) !important;
      border-color: rgba(216, 192, 148, 0.20) !important;
  }}
  section[data-testid="stSidebar"] [data-testid="stExpander"],
  section[data-testid="stSidebar"] details[data-testid="stExpander"] {{
      background: rgba(38, 35, 31, 0.76) !important;
      border: 1px solid rgba(216, 192, 148, 0.18) !important;
      border-radius: var(--fg-r-md) !important;
      box-shadow: inset 0 1px 0 rgba(255, 248, 232, 0.06) !important;
  }}
  section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
  section[data-testid="stSidebar"] [data-testid="stExpander"] summary * {{
      color: var(--fg-on-dark) !important;
  }}
  section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"],
  section[data-testid="stSidebar"] button[kind="secondary"] {{
      background: rgba(255, 248, 232, 0.06) !important;
      border: 1px solid rgba(216, 192, 148, 0.22) !important;
      color: var(--fg-on-dark) !important;
      box-shadow: inset 0 1px 0 rgba(255, 248, 232, 0.08) !important;
  }}
  section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] *,
  section[data-testid="stSidebar"] button[kind="secondary"] * {{
      color: var(--fg-on-dark) !important;
  }}
  section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover:not(:disabled),
  section[data-testid="stSidebar"] button[kind="secondary"]:hover:not(:disabled) {{
      background: rgba(216, 192, 148, 0.16) !important;
      border-color: rgba(216, 192, 148, 0.42) !important;
  }}

  /* =========================================================================
     4. Architectural Fluted-Glass Surface Tiers (Cards, Toolbars & Panels)
     ========================================================================= */
  /* Hide zero-height surface tier markers so they add zero gap */
  div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] .fg-surface-hook),
  div[data-testid="stElementContainer"]:has(.fg-surface-hook) {{
      display: none !important;
      margin: 0 !important;
      padding: 0 !important;
      height: 0 !important;
  }}

  /* Base bordered container hook (covers Streamlit 1.67 bordered stVerticalBlock,
     legacy stVerticalBlockBorderWrapper, and explicit glass_container() hooks) */
  div[data-testid="stVerticalBlockBorderWrapper"],
  div[data-testid="stVerticalBlock"][border="true"],
  div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .fg-surface-hook) {{
      position: relative !important;
      overflow: hidden !important;
      border-radius: var(--fg-r-lg) !important;
      padding: 1.15rem 1.28rem 1.12rem 1.28rem !important;
      border: 1px solid rgba(165, 132, 79, 0.22) !important;
      background:
          linear-gradient(165deg, rgba(253, 251, 247, 0.86) 0%, rgba(246, 241, 232, 0.76) 100%) !important;
      backdrop-filter: var(--fg-blur-glass) !important;
      -webkit-backdrop-filter: var(--fg-blur-glass) !important;
      box-shadow:
          var(--fg-shadow-bevel),
          var(--fg-shadow-rest) !important;
      transition:
          border-color var(--fg-base) var(--fg-ease),
          box-shadow var(--fg-base) var(--fg-ease),
          transform var(--fg-base) var(--fg-ease) !important;
  }}

  /* Fine machined fluted-glass crown strip along the top edge of frosted cards */
  div[data-testid="stVerticalBlockBorderWrapper"]::before,
  div[data-testid="stVerticalBlock"][border="true"]::before,
  div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .fg-surface-hook)::before {{
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 5px;
      background-image:
          linear-gradient(90deg, rgba(255, 255, 255, 0.55), rgba(217, 194, 154, 0.30), rgba(255, 255, 255, 0.55)),
          var(--fg-flute-light-fine);
      border-bottom: 1px solid rgba(165, 132, 79, 0.14);
      pointer-events: none;
      z-index: 1;
  }}

  div[data-testid="stVerticalBlockBorderWrapper"]:hover,
  div[data-testid="stVerticalBlock"][border="true"]:hover,
  div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .fg-surface-hook):hover {{
      border-color: rgba(165, 132, 79, 0.40) !important;
      box-shadow:
          var(--fg-shadow-bevel),
          var(--fg-shadow-raised) !important;
  }}

  /* Tier A: "reeded" — authentic full-surface vertical ribbed glass for toolbars & routing bars */
  div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .fg-surface-reeded) {{
      background-image:
          linear-gradient(180deg, rgba(252, 250, 246, 0.82) 0%, rgba(243, 237, 226, 0.74) 100%),
          var(--fg-flute-light-whisper) !important;
      border-color: rgba(165, 132, 79, 0.28) !important;
  }}
  div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .fg-surface-reeded)::before {{
      height: 6px;
      background-image:
          linear-gradient(90deg, rgba(217, 194, 154, 0.35), rgba(255, 255, 255, 0.65), rgba(217, 194, 154, 0.35)),
          var(--fg-flute-light);
      border-bottom: 1px solid rgba(165, 132, 79, 0.20);
  }}

  /* Tier B: "solid" — honed warm limestone/ivory surface for dense forms, tables & long prose */
  div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .fg-surface-solid) {{
      background:
          linear-gradient(180deg, #FDFBF8 0%, #F8F4EC 100%) !important;
      backdrop-filter: none !important;
      -webkit-backdrop-filter: none !important;
      border-color: rgba(180, 166, 142, 0.42) !important;
  }}
  div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .fg-surface-solid)::before {{
      height: 3px;
      background: linear-gradient(90deg, rgba(165, 132, 79, 0.42), rgba(217, 194, 154, 0.22), transparent);
      border-bottom: none;
  }}

  /* =========================================================================
     5. KPI Metric Plaques (Cut Frosted Glass + Left Fluted Mullion)
     ========================================================================= */
  div[data-testid="stMetric"] {{
      position: relative !important;
      overflow: hidden !important;
      background:
          linear-gradient(160deg, rgba(253, 251, 247, 0.90) 0%, rgba(246, 241, 232, 0.78) 100%) !important;
      backdrop-filter: var(--fg-blur-glass) !important;
      -webkit-backdrop-filter: var(--fg-blur-glass) !important;
      border: 1px solid rgba(165, 132, 79, 0.22) !important;
      border-radius: var(--fg-r-md) !important;
      padding: 15px 18px 15px 22px !important;
      box-shadow:
          var(--fg-shadow-bevel),
          var(--fg-shadow-rest) !important;
      transition:
          transform var(--fg-base) var(--fg-ease),
          border-color var(--fg-base) var(--fg-ease),
          box-shadow var(--fg-base) var(--fg-ease) !important;
  }}
  /* Vertical fluted-glass architectural mullion strip on the left edge */
  div[data-testid="stMetric"]::before {{
      content: "";
      position: absolute;
      top: 0;
      bottom: 0;
      left: 0;
      width: 9px;
      background-image:
          linear-gradient(180deg, rgba(217, 194, 154, 0.42), rgba(165, 132, 79, 0.18)),
          var(--fg-flute-light-fine);
      border-right: 1px solid rgba(165, 132, 79, 0.22);
      pointer-events: none;
  }}
  div[data-testid="stMetric"]:hover {{
      transform: translateY(-2px);
      border-color: rgba(165, 132, 79, 0.42) !important;
      box-shadow:
          var(--fg-shadow-bevel),
          var(--fg-shadow-raised) !important;
  }}
  div[data-testid="stMetricLabel"],
  div[data-testid="stMetricLabel"] p {{
      font-family: var(--fg-font-mono) !important;
      text-transform: uppercase !important;
      letter-spacing: 0.085em !important;
      font-size: 0.67rem !important;
      font-weight: 500 !important;
      color: var(--fg-ink-500) !important;
  }}
  div[data-testid="stMetricValue"],
  div[data-testid="stMetricValue"] > div {{
      font-family: var(--fg-font-display) !important;
      font-variant-numeric: tabular-nums !important;
      font-size: 1.56rem !important;
      font-weight: 600 !important;
      letter-spacing: -0.024em !important;
      line-height: 1.18 !important;
      color: var(--fg-ink-900) !important;
  }}
  div[data-testid="stMetricDelta"] {{
      font-family: var(--fg-font-mono) !important;
      font-size: 0.72rem !important;
      font-weight: 500 !important;
  }}

  /* =========================================================================
     6. Expanders, Dialogs, Popovers, Alerts, Uploaders & DataFrames
     ========================================================================= */
  div[data-testid="stExpander"],
  details[data-testid="stExpander"] {{
      background:
          linear-gradient(180deg, rgba(252, 250, 246, 0.86) 0%, rgba(246, 241, 232, 0.78) 100%) !important;
      backdrop-filter: var(--fg-blur-soft) !important;
      border: 1px solid rgba(165, 132, 79, 0.22) !important;
      border-radius: var(--fg-r-md) !important;
      box-shadow: var(--fg-shadow-bevel), var(--fg-shadow-rest) !important;
      overflow: hidden !important;
      transition: border-color var(--fg-fast) var(--fg-ease), box-shadow var(--fg-fast) var(--fg-ease) !important;
  }}
  div[data-testid="stExpander"]:hover,
  details[data-testid="stExpander"]:hover {{
      border-color: rgba(165, 132, 79, 0.38) !important;
  }}
  div[data-testid="stExpander"] summary,
  details[data-testid="stExpander"] summary {{
      padding: 0.72rem 1.05rem !important;
      font-family: var(--fg-font-sans) !important;
      font-weight: 600 !important;
      font-size: 0.87rem !important;
      color: var(--fg-ink-800) !important;
      transition: color var(--fg-fast) var(--fg-ease), background-color var(--fg-fast) var(--fg-ease) !important;
  }}
  div[data-testid="stExpander"] summary:hover,
  details[data-testid="stExpander"] summary:hover {{
      color: var(--fg-bronze-700) !important;
      background: rgba(232, 217, 190, 0.18) !important;
  }}
  div[data-testid="stExpanderDetails"] {{
      border-top: 1px solid rgba(165, 132, 79, 0.14) !important;
      background: rgba(251, 249, 245, 0.65) !important;
  }}

  /* Floating Modals, Popovers & Toasts */
  div[data-testid="stDialog"] > div > div,
  div[role="dialog"],
  div[data-testid="stPopoverBody"] {{
      background:
          linear-gradient(170deg, rgba(253, 251, 247, 0.95) 0%, rgba(244, 239, 229, 0.92) 100%) !important;
      backdrop-filter: var(--fg-blur-frost) !important;
      -webkit-backdrop-filter: var(--fg-blur-frost) !important;
      border: 1px solid rgba(165, 132, 79, 0.32) !important;
      border-radius: var(--fg-r-xl) !important;
      box-shadow:
          var(--fg-shadow-bevel),
          var(--fg-shadow-float) !important;
  }}
  div[data-testid="stToast"] {{
      background:
          linear-gradient(160deg, rgba(31, 29, 26, 0.94) 0%, rgba(22, 20, 18, 0.96) 100%) !important;
      backdrop-filter: var(--fg-blur-glass) !important;
      color: var(--fg-on-dark) !important;
      border: 1px solid rgba(216, 192, 148, 0.30) !important;
      border-radius: var(--fg-r-md) !important;
      box-shadow: var(--fg-shadow-float) !important;
  }}
  div[data-testid="stToast"] * {{
      color: var(--fg-on-dark) !important;
  }}

  /* Dataframes & Markdown Tables */
  div[data-testid="stDataFrame"],
  div[data-testid="stTable"] {{
      background: #FCFBF8 !important;
      border: 1px solid rgba(180, 166, 142, 0.45) !important;
      border-radius: var(--fg-r-md) !important;
      overflow: hidden !important;
      box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.9),
          0 2px 10px -4px rgba(40, 32, 20, 0.08) !important;
  }}

  /* Reeded-glass File Uploader Dropzone */
  section[data-testid="stFileUploaderDropzone"] {{
      background-color: rgba(248, 244, 236, 0.80) !important;
      background-image:
          linear-gradient(180deg, rgba(252, 250, 246, 0.78) 0%, rgba(242, 236, 224, 0.72) 100%),
          var(--fg-flute-light) !important;
      border: 1.5px dashed rgba(138, 106, 59, 0.42) !important;
      border-radius: var(--fg-r-md) !important;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.85) !important;
      transition:
          border-color var(--fg-base) var(--fg-ease),
          background-color var(--fg-base) var(--fg-ease),
          box-shadow var(--fg-base) var(--fg-ease) !important;
  }}
  section[data-testid="stFileUploaderDropzone"]:hover {{
      border-color: var(--fg-bronze-600) !important;
      background-color: rgba(250, 245, 236, 0.95) !important;
      box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.95),
          0 8px 22px -12px rgba(138, 106, 59, 0.28) !important;
  }}

  /* Callout Alerts */
  div[data-testid="stAlertContainer"],
  div[data-testid="stAlert"] {{
      background:
          linear-gradient(90deg, rgba(250, 245, 236, 0.92) 0%, rgba(252, 250, 246, 0.86) 100%) !important;
      backdrop-filter: var(--fg-blur-soft) !important;
      border: 1px solid rgba(165, 132, 79, 0.24) !important;
      border-left: 3.5px solid var(--fg-bronze-500) !important;
      border-radius: var(--fg-r-md) !important;
      box-shadow: var(--fg-shadow-bevel), var(--fg-shadow-rest) !important;
  }}
  div[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{
      border-left-color: var(--fg-success) !important;
      background: linear-gradient(90deg, rgba(231, 238, 228, 0.86) 0%, rgba(252, 250, 246, 0.88) 100%) !important;
  }}
  div[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{
      border-left-color: var(--fg-warning) !important;
      background: linear-gradient(90deg, rgba(245, 236, 219, 0.88) 0%, rgba(252, 250, 246, 0.88) 100%) !important;
  }}
  div[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{
      border-left-color: var(--fg-danger) !important;
      background: linear-gradient(90deg, rgba(244, 227, 223, 0.88) 0%, rgba(252, 250, 246, 0.88) 100%) !important;
  }}

  /* Progress Bar — Fluted Champagne Fill */
  div[data-testid="stProgress"] > div > div > div > div {{
      background-color: var(--fg-bronze-500) !important;
      background-image:
          linear-gradient(90deg, var(--fg-bronze-600), var(--fg-champagne-400)),
          var(--fg-flute-champagne) !important;
      border-radius: var(--fg-r-pill) !important;
  }}

  /* Code Blocks */
  div[data-testid="stCode"] pre {{
      background: #201E1B !important;
      color: #EAE3D5 !important;
      border: 1px solid rgba(216, 192, 148, 0.22) !important;
      border-radius: var(--fg-r-md) !important;
      font-family: var(--fg-font-mono) !important;
      font-size: 0.80rem !important;
  }}

  /* =========================================================================
     7. Tabs, Segmented Controls & Quick-Action Navigation Cards
     ========================================================================= */
  div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
      gap: 0.35rem !important;
      background: rgba(234, 228, 217, 0.62) !important;
      padding: 5px 6px !important;
      border-radius: var(--fg-r-md) !important;
      border: 1px solid rgba(165, 132, 79, 0.18) !important;
      box-shadow: inset 0 1px 2px rgba(40, 32, 20, 0.05) !important;
  }}
  div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
      display: none !important;
  }}
  div[data-testid="stTabs"] [data-baseweb="tab-border"] {{
      display: none !important;
  }}
  div[data-testid="stTabs"] button[data-baseweb="tab"] {{
      padding: 0.48rem 0.98rem !important;
      font-family: var(--fg-font-sans) !important;
      font-size: 0.84rem !important;
      font-weight: 500 !important;
      color: var(--fg-ink-600) !important;
      border-radius: var(--fg-r-sm) !important;
      border: 1px solid transparent !important;
      background: transparent !important;
      transition:
          color var(--fg-fast) var(--fg-ease),
          background-color var(--fg-fast) var(--fg-ease),
          box-shadow var(--fg-fast) var(--fg-ease),
          border-color var(--fg-fast) var(--fg-ease) !important;
  }}
  div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {{
      color: var(--fg-ink-900) !important;
      background: rgba(251, 249, 245, 0.62) !important;
  }}
  div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
      color: var(--fg-ink-900) !important;
      font-weight: 600 !important;
      background:
          linear-gradient(180deg, #FDFBF8 0%, #F5EFE4 100%) !important;
      border-color: rgba(165, 132, 79, 0.30) !important;
      box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.95),
          0 2px 8px -2px rgba(40, 32, 20, 0.12) !important;
  }}
  div[data-testid="stTabPanel"] {{
      padding-top: 1.15rem !important;
  }}

  /* Segmented Control Group */
  div[data-testid="stButtonGroup"] {{
      background: rgba(234, 228, 217, 0.65) !important;
      padding: 4px !important;
      border-radius: var(--fg-r-md) !important;
      border: 1px solid rgba(165, 132, 79, 0.18) !important;
  }}
  button[data-testid="stBaseButton-segmented_control"] {{
      border: 1px solid transparent !important;
      border-radius: var(--fg-r-sm) !important;
      background: transparent !important;
      color: var(--fg-ink-600) !important;
      font-size: 0.81rem !important;
  }}
  button[data-testid="stBaseButton-segmented_controlActive"] {{
      background: linear-gradient(180deg, #FDFBF8 0%, #F4ECE0 100%) !important;
      border: 1px solid rgba(165, 132, 79, 0.34) !important;
      border-radius: var(--fg-r-sm) !important;
      color: var(--fg-ink-900) !important;
      font-weight: 600 !important;
      box-shadow: 0 1px 4px rgba(40, 32, 20, 0.10) !important;
  }}

  /* Quick-Action Page-Link Cards (Fluted Glass Interactive Tiles) */
  a[data-testid="stPageLink-NavLink"] {{
      position: relative !important;
      overflow: hidden !important;
      background-color: rgba(252, 250, 246, 0.86) !important;
      background-image:
          linear-gradient(135deg, rgba(253, 251, 247, 0.92) 0%, rgba(244, 238, 227, 0.78) 100%),
          var(--fg-flute-light-whisper) !important;
      backdrop-filter: var(--fg-blur-soft) !important;
      border: 1px solid rgba(165, 132, 79, 0.25) !important;
      border-radius: var(--fg-r-md) !important;
      padding: 0.82rem 1.05rem !important;
      box-shadow:
          var(--fg-shadow-bevel),
          var(--fg-shadow-rest) !important;
      transition:
          border-color var(--fg-base) var(--fg-ease),
          transform var(--fg-base) var(--fg-ease),
          box-shadow var(--fg-base) var(--fg-ease),
          background-color var(--fg-base) var(--fg-ease) !important;
  }}
  a[data-testid="stPageLink-NavLink"]:hover {{
      border-color: var(--fg-bronze-500) !important;
      transform: translateY(-2px);
      box-shadow:
          var(--fg-shadow-bevel),
          var(--fg-shadow-raised) !important;
  }}
  a[data-testid="stPageLink-NavLink"] p {{
      font-weight: 600 !important;
      font-size: 0.86rem !important;
      color: var(--fg-ink-900) !important;
  }}
  a[data-testid="stPageLink-NavLink"] span[data-testid="stIconMaterial"] {{
      color: var(--fg-bronze-600) !important;
  }}

  /* =========================================================================
     8. Sculpted Inputs, Dropdowns & Fluted Graphite/Bronze Buttons
     ========================================================================= */
  div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
  div[data-testid="stMultiSelect"] [data-baseweb="select"] > div,
  div[data-testid="stTextInput"] div[data-baseweb="input"],
  div[data-testid="stNumberInput"] div[data-baseweb="input"],
  div[data-testid="stDateInput"] div[data-baseweb="input"],
  div[data-testid="stTextArea"] div[data-baseweb="textarea"] {{
      background-color: #FCFBF8 !important;
      border: 1px solid rgba(175, 160, 136, 0.52) !important;
      border-radius: var(--fg-r-sm) !important;
      box-shadow:
          inset 0 1px 2px rgba(40, 32, 20, 0.04),
          0 1px 0 rgba(255, 255, 255, 0.75) !important;
      transition:
          border-color var(--fg-fast) var(--fg-ease),
          box-shadow var(--fg-fast) var(--fg-ease),
          background-color var(--fg-fast) var(--fg-ease) !important;
  }}
  div[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
  div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:hover,
  div[data-testid="stTextInput"] div[data-baseweb="input"]:hover,
  div[data-testid="stNumberInput"] div[data-baseweb="input"]:hover,
  div[data-testid="stDateInput"] div[data-baseweb="input"]:hover,
  div[data-testid="stTextArea"] div[data-baseweb="textarea"]:hover {{
      border-color: var(--fg-bronze-500) !important;
      background-color: #FFFFFF !important;
  }}
  div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
  div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within,
  div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
  div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within,
  div[data-testid="stDateInput"] div[data-baseweb="input"]:focus-within,
  div[data-testid="stTextArea"] div[data-baseweb="textarea"]:focus-within {{
      border-color: var(--fg-bronze-600) !important;
      box-shadow: var(--fg-shadow-focus) !important;
      background-color: #FFFFFF !important;
  }}

  /* Multiselect tags in muted champagne stone */
  div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {{
      background: var(--fg-champagne-100) !important;
      color: var(--fg-bronze-700) !important;
      border: 1px solid rgba(165, 132, 79, 0.32) !important;
      border-radius: var(--fg-r-xs) !important;
      font-weight: 500 !important;
  }}

  /* Dropdown option menus */
  ul[data-testid="stSelectboxVirtualDropdown"],
  div[data-baseweb="popover"] ul {{
      background: #FCFBF8 !important;
      border: 1px solid rgba(165, 132, 79, 0.28) !important;
      border-radius: var(--fg-r-md) !important;
      box-shadow: var(--fg-shadow-float) !important;
  }}
  ul[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
  ul[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] {{
      background: var(--fg-champagne-100) !important;
      color: var(--fg-ink-900) !important;
  }}

  /* Secondary, Download & Popover Trigger Buttons — Frosted Ivory Glass */
  button[data-testid="stBaseButton-secondary"],
  button[kind="secondary"],
  div[data-testid="stDownloadButton"] button,
  div[data-testid="stPopover"] > div > button {{
      background:
          linear-gradient(180deg, rgba(253, 251, 247, 0.94) 0%, rgba(242, 236, 225, 0.88) 100%) !important;
      border: 1px solid rgba(165, 132, 79, 0.28) !important;
      border-radius: var(--fg-r-sm) !important;
      color: var(--fg-ink-800) !important;
      font-weight: 600 !important;
      font-size: 0.84rem !important;
      letter-spacing: 0.005em !important;
      box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.92),
          0 1px 3px rgba(40, 32, 20, 0.06) !important;
      transition:
          border-color var(--fg-fast) var(--fg-ease),
          background var(--fg-fast) var(--fg-ease),
          transform var(--fg-fast) var(--fg-ease),
          box-shadow var(--fg-fast) var(--fg-ease) !important;
  }}
  button[data-testid="stBaseButton-secondary"]:hover:not(:disabled),
  button[kind="secondary"]:hover:not(:disabled),
  div[data-testid="stDownloadButton"] button:hover:not(:disabled),
  div[data-testid="stPopover"] > div > button:hover:not(:disabled) {{
      border-color: var(--fg-bronze-500) !important;
      background:
          linear-gradient(180deg, #FFFFFF 0%, #F5ECDD 100%) !important;
      transform: translateY(-1px);
      box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.98),
          0 6px 16px -8px rgba(40, 32, 20, 0.22) !important;
  }}
  button[data-testid="stBaseButton-secondary"]:active:not(:disabled),
  button[kind="secondary"]:active:not(:disabled) {{
      transform: translateY(0);
  }}

  /* Primary Action Buttons — Smoked Graphite Fluted Glass with Champagne Edge */
  button[data-testid="stBaseButton-primary"],
  button[kind="primary"] {{
      background-color: var(--fg-graphite-800) !important;
      background-image:
          linear-gradient(180deg, rgba(46, 42, 37, 0.94) 0%, rgba(27, 25, 22, 0.98) 100%),
          var(--fg-flute-dark-fine) !important;
      border: 1px solid rgba(216, 192, 148, 0.42) !important;
      border-radius: var(--fg-r-sm) !important;
      color: #FAF5EC !important;
      font-weight: 600 !important;
      font-size: 0.86rem !important;
      letter-spacing: 0.012em !important;
      box-shadow:
          inset 0 1px 0 rgba(243, 234, 219, 0.26),
          0 4px 14px -4px rgba(28, 24, 18, 0.38) !important;
      transition:
          transform var(--fg-fast) var(--fg-ease),
          border-color var(--fg-fast) var(--fg-ease),
          box-shadow var(--fg-fast) var(--fg-ease),
          filter var(--fg-fast) var(--fg-ease) !important;
  }}
  button[data-testid="stBaseButton-primary"] *,
  button[kind="primary"] * {{
      color: #FAF5EC !important;
  }}
  button[data-testid="stBaseButton-primary"]:hover:not(:disabled),
  button[kind="primary"]:hover:not(:disabled) {{
      border-color: var(--fg-champagne-300) !important;
      transform: translateY(-1px);
      filter: brightness(1.07);
      box-shadow:
          inset 0 1px 0 rgba(250, 245, 236, 0.38),
          0 10px 24px -8px rgba(28, 24, 18, 0.48) !important;
  }}
  button[data-testid="stBaseButton-primary"]:active:not(:disabled),
  button[kind="primary"]:active:not(:disabled) {{
      transform: translateY(0);
  }}
  button:focus-visible {{
      outline: none !important;
      box-shadow: var(--fg-shadow-focus) !important;
  }}

  /* =========================================================================
     9. Architectural Page Hero Frieze & Section Headers
     ========================================================================= */
  .studio-header {{
      position: relative;
      overflow: hidden;
      padding: 1.35rem 1.55rem 1.28rem 1.55rem;
      margin-bottom: 1.55rem;
      border-radius: var(--fg-r-xl);
      background:
          linear-gradient(125deg, rgba(253, 251, 247, 0.92) 0%, rgba(248, 243, 234, 0.82) 58%, rgba(240, 232, 218, 0.72) 100%);
      backdrop-filter: var(--fg-blur-frost);
      -webkit-backdrop-filter: var(--fg-blur-frost);
      border: 1px solid rgba(165, 132, 79, 0.26);
      box-shadow:
          var(--fg-shadow-bevel),
          var(--fg-shadow-rest);
  }}
  /* Authentic vertical fluted-glass architectural fascia on the sunlit right wing */
  .studio-header::before {{
      content: "";
      position: absolute;
      top: 0;
      right: 0;
      bottom: 0;
      width: 44%;
      background-image: var(--fg-flute-light);
      -webkit-mask-image: linear-gradient(90deg, transparent 0%, rgba(0, 0, 0, 0.55) 45%, rgba(0, 0, 0, 0.95) 100%);
      mask-image: linear-gradient(90deg, transparent 0%, rgba(0, 0, 0, 0.55) 45%, rgba(0, 0, 0, 0.95) 100%);
      pointer-events: none;
  }}
  /* Brushed champagne horizon accent along the base of the header */
  .studio-header::after {{
      content: "";
      position: absolute;
      left: 1.55rem;
      right: 1.55rem;
      bottom: 0;
      height: 1.5px;
      background: linear-gradient(
          90deg,
          var(--fg-bronze-500) 0%,
          rgba(198, 168, 116, 0.45) 36%,
          transparent 100%
      );
      pointer-events: none;
  }}
  .studio-header-top {{
      position: relative;
      z-index: 1;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 0.36rem;
  }}
  .studio-eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-family: var(--fg-font-mono);
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.13em;
      color: var(--fg-bronze-600);
      font-weight: 500;
  }}
  .studio-eyebrow::before {{
      content: "";
      display: inline-block;
      width: 14px;
      height: 1.5px;
      background: var(--fg-bronze-500);
  }}
  .studio-title {{
      position: relative;
      z-index: 1;
      font-family: var(--fg-font-display);
      font-size: var(--fg-text-hero);
      font-weight: 600;
      color: var(--fg-ink-900);
      letter-spacing: -0.025em;
      margin: 0;
      line-height: 1.18;
  }}
  .studio-sub {{
      position: relative;
      z-index: 1;
      color: var(--fg-ink-600);
      font-size: 0.91rem;
      line-height: 1.56;
      margin-top: 0.42rem;
      max-width: 920px;
  }}

  /* Architectural Section Header */
  .sec-hdr {{
      margin-top: 1.55rem;
      margin-bottom: 0.85rem;
      padding-bottom: 0.52rem;
      border-bottom: 1px solid rgba(180, 166, 142, 0.42);
      position: relative;
  }}
  .sec-hdr::after {{
      content: "";
      position: absolute;
      left: 0;
      bottom: -1px;
      width: 64px;
      height: 1.5px;
      background: var(--fg-bronze-500);
  }}
  .sec-hdr-kicker {{
      font-family: var(--fg-font-mono);
      font-size: 0.66rem;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.125em;
      color: var(--fg-bronze-600);
      margin-bottom: 0.16rem;
  }}
  .sec-hdr-title {{
      font-family: var(--fg-font-display);
      font-size: var(--fg-text-section);
      font-weight: 600;
      color: var(--fg-ink-900);
      letter-spacing: -0.018em;
      line-height: 1.28;
      margin: 0;
  }}
  .sec-hdr-sub {{
      font-size: 0.825rem;
      color: var(--fg-ink-500);
      margin-top: 0.2rem;
      line-height: 1.5;
  }}

  /* =========================================================================
     10. Sidebar Brand Lockup, Status Telemetry & Micro-Badges
     ========================================================================= */
  .brand-lockup {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 4px 2px 15px 2px;
      border-bottom: 1px solid rgba(216, 192, 148, 0.18);
      margin-bottom: 13px;
  }}
  .brand-lockup img {{
      width: 40px;
      height: 40px;
      flex-shrink: 0;
      border-radius: 10px;
      box-shadow: 0 6px 16px -6px rgba(0, 0, 0, 0.55);
  }}
  .brand-name {{
      font-family: var(--fg-font-display);
      font-size: 1.02rem;
      font-weight: 600;
      color: #F6F1E7;
      letter-spacing: -0.016em;
      line-height: 1.18;
  }}
  .brand-tag {{
      font-family: var(--fg-font-mono);
      font-size: 0.63rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--fg-champagne-on-dark);
      font-weight: 500;
      margin-top: 3px;
      opacity: 0.92;
  }}

  /* Status indicator strip in smoked-graphite sidebar */
  .status-strip {{
      display: flex;
      align-items: center;
      gap: 9px;
      padding: 8px 12px;
      border-radius: var(--fg-r-sm);
      font-family: var(--fg-font-mono);
      font-size: 0.72rem;
      font-weight: 500;
      letter-spacing: 0.02em;
      margin-bottom: 12px;
      backdrop-filter: var(--fg-blur-soft);
      box-shadow: inset 0 1px 0 rgba(255, 248, 232, 0.07);
  }}
  .status-ok {{
      background: rgba(255, 249, 238, 0.06);
      border: 1px solid rgba(216, 192, 148, 0.22);
      color: #E7E0D3;
  }}
  .status-warn {{
      background: rgba(162, 116, 39, 0.16);
      border: 1px solid rgba(216, 192, 148, 0.36);
      color: #F3DFBA;
  }}
  .status-err {{
      background: rgba(155, 61, 51, 0.22);
      border: 1px solid rgba(212, 128, 118, 0.36);
      color: #F5D2CE;
  }}
  .dot {{
      width: 7px;
      height: 7px;
      border-radius: 50%;
      display: inline-block;
      flex-shrink: 0;
  }}
  .dot-ok {{
      background: #7FA37A;
      box-shadow: 0 0 0 3px rgba(127, 163, 122, 0.20);
  }}
  .dot-warn {{
      background: #D8A95A;
      box-shadow: 0 0 0 3px rgba(216, 169, 90, 0.20);
  }}
  .dot-err {{
      background: #CF6B5E;
      box-shadow: 0 0 0 3px rgba(207, 107, 94, 0.22);
  }}

  /* Architectural Micro-Label Capsules */
  .pill {{
      display: inline-flex;
      align-items: center;
      font-family: var(--fg-font-mono);
      padding: 2.5px 9px;
      border-radius: var(--fg-r-pill);
      font-size: 0.65rem;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.065em;
      margin-right: 5px;
      vertical-align: middle;
      line-height: 1.42;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65);
  }}
  .pill-deployed {{
      background: var(--fg-success-soft);
      color: #3A5940;
      border: 1px solid rgba(79, 115, 86, 0.34);
  }}
  .pill-user {{
      background: var(--fg-warning-soft);
      color: #7D5618;
      border: 1px solid rgba(162, 116, 39, 0.36);
  }}
  .pill-m1 {{
      background: var(--fg-champagne-100);
      color: var(--fg-bronze-700);
      border: 1px solid rgba(165, 132, 79, 0.38);
  }}
  .pill-space {{
      background: var(--fg-info-soft);
      color: #384F5C;
      border: 1px solid rgba(86, 112, 127, 0.34);
  }}
  .pill-muted {{
      background: rgba(234, 228, 217, 0.78);
      color: var(--fg-ink-600);
      border: 1px solid rgba(180, 166, 142, 0.42);
  }}

  /* Card internal section kicker */
  .section-kicker {{
      display: flex;
      align-items: center;
      gap: 7px;
      font-family: var(--fg-font-mono);
      font-size: 0.66rem;
      text-transform: uppercase;
      letter-spacing: 0.11em;
      color: var(--fg-bronze-600);
      font-weight: 500;
      margin-bottom: 6px;
  }}
  .section-kicker::before {{
      content: "";
      width: 6px;
      height: 6px;
      border-radius: 1.5px;
      background: var(--fg-champagne-400);
      border: 1px solid var(--fg-bronze-600);
      flex-shrink: 0;
  }}

  /* Dropdown controls & option lists: pointer cursor instead of I-beam */
  div[data-testid="stSelectbox"] [role="group"],
  div[data-testid="stSelectbox"] input:not([disabled]):not([data-disabled]),
  div[data-testid="stMultiSelect"] [role="group"],
  div[data-testid="stMultiSelect"] input:not([disabled]):not([data-disabled]),
  div[data-testid="stMultiSelectTagsContainer"],
  div[data-baseweb="select"]:not([aria-disabled="true"]),
  div[data-baseweb="select"]:not([aria-disabled="true"]) *,
  ul[data-testid="stSelectboxVirtualDropdown"] [role="option"]:not([data-disabled]),
  ul[data-testid="stSelectboxVirtualDropdown"] [role="option"]:not([data-disabled]) *,
  div[data-testid="stMultiSelectDropdown"] [role="option"]:not([data-disabled]),
  div[data-testid="stMultiSelectDropdown"] [role="option"]:not([data-disabled]) * {{
      cursor: pointer !important;
  }}

  div[data-testid="stSelectbox"] input,
  div[data-testid="stMultiSelect"] input {{
      caret-color: transparent !important;
      user-select: none !important;
  }}
</style>
"""


GLOBAL_CSS = _build_global_css()


def inject_global_css() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


@contextmanager
def glass_container(variant: GlassVariant = "frosted") -> Iterator[Any]:
    """Render an architectural glass card in one of three deliberate surface tiers:

    - ``"reeded"``  : translucent pane with subtle vertical cylindrical flutes (toolbars & filters).
    - ``"frosted"`` : frosted warm-ivory glass card with fine fluted crown bezel (default cards).
    - ``"solid"``   : opaque honed limestone/ivory panel for dense forms, tables & reading clarity.
    """
    with st.container(border=True) as box:
        st.markdown(
            f'<span class="fg-surface-hook fg-surface-{variant}" aria-hidden="true"></span>',
            unsafe_allow_html=True,
        )
        yield box


def render_sidebar_brand() -> None:
    svg_path = ASSETS_DIR / "hotel_mark.svg"
    img_html = ""
    if svg_path.exists():
        b64 = base64.b64encode(svg_path.read_bytes()).decode("ascii")
        img_html = f'<img src="data:image/svg+xml;base64,{b64}" alt="H2"/>'
    st.markdown(
        f'<div class="brand-lockup">'
        f"{img_html}"
        f'<div><div class="brand-name">City Hotel H2</div>'
        f'<div class="brand-tag">Demand Intelligence</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_sidebar_status(dep_count: int, total_bundles: int) -> None:
    if dep_count == 3:
        cls, dot = "status-ok", "dot-ok"
        txt = f"3/3 Pipelines Active · {total_bundles} Bundles"
    elif total_bundles > 0:
        cls, dot = "status-warn", "dot-warn"
        txt = f"{dep_count}/3 Pipelines Active · {total_bundles} Bundles"
    else:
        cls, dot = "status-err", "dot-err"
        txt = "No Artifacts Built"
    st.markdown(
        f'<div class="status-strip {cls}"><span class="dot {dot}"></span><span>{txt}</span></div>',
        unsafe_allow_html=True,
    )


def pill_html(text: str, kind: str = "muted") -> str:
    return f'<span class="pill pill-{kind}">{text}</span>'


def family_pill_html(family: str) -> str:
    col = family_color(family)
    return (
        f'<span class="pill" style="background:{col}16;color:{col};border:1px solid {col}42;">'
        f"{family.replace('_', ' ')}</span>"
    )


def page_header(title: str, subtitle: str, pills_html: str = "", eyebrow: str = "CITY HOTEL (H2) · CSE382") -> None:
    st.markdown(
        f'<div class="studio-header">'
        f'<div class="studio-header-top">'
        f'<div class="studio-eyebrow">{eyebrow}</div>'
        f'<div>{pills_html}</div>'
        f"</div>"
        f'<div class="studio-title">{title}</div>'
        f'<div class="studio-sub">{subtitle}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "", kicker: str = "") -> None:
    k_html = f'<div class="sec-hdr-kicker">{kicker}</div>' if kicker else ""
    s_html = f'<div class="sec-hdr-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="sec-hdr">{k_html}<div class="sec-hdr-title">{title}</div>{s_html}</div>',
        unsafe_allow_html=True,
    )


def format_table_columns(df: pd.DataFrame) -> dict[str, Any]:
    """Build clean Streamlit column_config labels and numeric formats for experiment/audit tables."""
    cfg: dict[str, Any] = {}
    for col in df.columns:
        label = PRETTY_COLUMN_LABELS.get(str(col))
        if col == "is_deployed":
            cfg[col] = st.column_config.CheckboxColumn(label or "Deployed")
        elif col == "train_time_s":
            cfg[col] = st.column_config.NumberColumn(label or "Train Time (s)", format="%.3f s")
        elif col == "n_train_rows":
            cfg[col] = st.column_config.NumberColumn(label or "Train Rows", format="%d")
        elif col == "pca_variance_kept":
            cfg[col] = st.column_config.NumberColumn(label or "Variance Retained", format="%.4f")
        elif label:
            cfg[col] = st.column_config.Column(label)
    return cfg


def run_page_safely(render_fn: Callable[[], None]) -> None:
    try:
        render_fn()
    except Exception as exc:
        st.error(f"An unexpected error occurred while rendering this view: `{type(exc).__name__}: {exc}`")
        with st.expander("Technical diagnostic traceback"):
            st.code(traceback.format_exc(), language="python")


def render_empty_artifacts_banner(task: str | None = None) -> None:
    label = f" for **{task}**" if task else ""
    st.warning(
        f"No trained artifact bundles were found{label}. "
        "Generate official repository bundles from the **Artifacts & Jobs** page "
        "or by running `python main.py build-artifacts`.",
        icon=":material/warning:",
    )
