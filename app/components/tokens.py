"""Fluted Glass design tokens — the single source of truth for the UI's visual identity.

Every colour, type family, radius, elevation, motion curve and fluted-glass texture
recipe used by the custom CSS layer (`app/components/style.py`) is defined here and
emitted as `--fg-*` CSS custom properties.  The chart series palette lives in
`src/evaluation/figures.py` (so the core package stays UI-agnostic) and is re-exported
here as `--fg-chart-*` variables.

Palette concept — "Travertine & Smoked Bronze":
    * Canvas  : warm travertine / ivory stone, lit from the upper right.
    * Ink     : warm graphite (never pure black) for text and structural surfaces.
    * Accent  : bronze -> champagne metallic family, used sparingly for focus,
                selection and hairline details.
    * Support : smoked slate, sage and clay — muted, architectural, low-chroma.
"""
from __future__ import annotations

from src.evaluation.figures import PALETTE

# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------
COLORS: dict[str, str] = {
    # Travertine canvas & stone surfaces
    "canvas": "#F4F0E9",
    "canvas-deep": "#ECE6DC",
    "stone-50": "#FBF9F5",
    "stone-100": "#F4F0E9",
    "stone-200": "#EAE4D9",
    "stone-300": "#DDD5C7",
    "stone-400": "#C9BFAE",
    # Warm graphite ink
    "ink-900": "#1D1C1A",
    "ink-800": "#23211E",
    "ink-700": "#3A3733",
    "ink-600": "#55514A",
    "ink-500": "#6E695F",
    "ink-400": "#948E82",
    "ink-300": "#B9B2A5",
    # Bronze / champagne metallic accent
    "champagne-50": "#FAF5EC",
    "champagne-100": "#F3EADB",
    "champagne-200": "#E8D9BE",
    "champagne-300": "#D9C29A",
    "champagne-400": "#C6A874",
    "bronze-500": "#A5844F",
    "bronze-600": "#8A6A3B",
    "bronze-700": "#6E5229",
    # Muted architectural support hues
    "slate-500": "#2F4550",
    "sage-500": "#6F8A6A",
    "smoke-500": "#56707F",
    "clay-500": "#A8664F",
    # Semantic
    "success": "#4F7356",
    "success-soft": "#E7EEE4",
    "warning": "#A27427",
    "warning-soft": "#F5ECDB",
    "danger": "#9B3D33",
    "danger-soft": "#F4E3DF",
    "info": "#56707F",
    "info-soft": "#E6ECEE",
    # Graphite glass (sidebar, primary actions, tooltips)
    "graphite-950": "#141311",
    "graphite-900": "#1D1C1A",
    "graphite-800": "#26241F",
    "graphite-700": "#34312C",
    "on-dark": "#ECE6DA",
    "on-dark-muted": "#A39C8F",
    "on-dark-faint": "#6F695F",
    "champagne-on-dark": "#D8C094",
}

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
FONT_IMPORT_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Plus+Jakarta+Sans:wght@500;600;700"
    "&family=Manrope:wght@400;500;600;700"
    "&family=IBM+Plex+Mono:wght@400;500;600"
    "&display=swap"
)

FONTS: dict[str, str] = {
    "display": "'Plus Jakarta Sans', 'Manrope', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif",
    "sans": "'Manrope', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif",
    "mono": "'IBM Plex Mono', 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace",
}

TYPE_SCALE: dict[str, str] = {
    "hero": "clamp(1.62rem, 1.25rem + 1.05vw, 2.02rem)",
    "title": "1.42rem",
    "section": "1.14rem",
    "body": "0.92rem",
    "small": "0.82rem",
    "micro": "0.68rem",
}

# ---------------------------------------------------------------------------
# Shape, elevation & motion
# ---------------------------------------------------------------------------
RADII: dict[str, str] = {
    "xs": "6px",
    "sm": "9px",
    "md": "12px",
    "lg": "16px",
    "xl": "22px",
    "pill": "999px",
}

SHADOWS: dict[str, str] = {
    # Inner bevel: bright top lip + faint inner rim = thickness of a glass pane.
    "bevel": "inset 0 1px 0 rgba(255,255,255,0.85), inset 0 0 0 1px rgba(255,255,255,0.35)",
    "hairline": "0 0 0 1px rgba(70,56,36,0.09)",
    "rest": "0 1px 1px rgba(40,32,20,0.04), 0 8px 24px -14px rgba(40,32,20,0.22)",
    "raised": "0 2px 4px rgba(40,32,20,0.05), 0 18px 40px -18px rgba(40,32,20,0.30)",
    "float": "0 4px 10px rgba(30,24,16,0.08), 0 30px 70px -24px rgba(30,24,16,0.40)",
    "focus": "0 0 0 3px rgba(198,168,116,0.32)",
}

MOTION: dict[str, str] = {
    "ease": "cubic-bezier(0.2, 0.7, 0.2, 1)",
    "ease-out": "cubic-bezier(0.16, 1, 0.3, 1)",
    "fast": "140ms",
    "base": "220ms",
    "slow": "460ms",
}

BLUR: dict[str, str] = {
    "glass": "blur(16px) saturate(1.12)",
    "frost": "blur(22px) saturate(1.08)",
    "soft": "blur(8px)",
}

# ---------------------------------------------------------------------------
# Fluted (reeded) glass texture recipes
#
# Each recipe models one cylindrical rib lit from the upper-left: a dark groove
# line, a bright specular ridge, a soft falling half-tone and a shaded flank
# before the next groove.  Tiled with `repeating-linear-gradient` they read as
# vertical architectural reeding rather than flat transparency.
# ---------------------------------------------------------------------------
FLUTES: dict[str, str] = {
    # Light pane, standard 14px reed (hero panels, dropzones, tiles)
    "light": (
        "repeating-linear-gradient(90deg,"
        " rgba(92,72,44,0.10) 0px,"
        " rgba(255,255,255,0.00) 1px,"
        " rgba(255,255,255,0.70) 3px,"
        " rgba(255,255,255,0.22) 6.5px,"
        " rgba(120,96,60,0.035) 10.5px,"
        " rgba(92,72,44,0.085) 13.4px,"
        " rgba(92,72,44,0.10) 14px)"
    ),
    # Light pane, fine 7px reed (card top edges, inputs accents)
    "light-fine": (
        "repeating-linear-gradient(90deg,"
        " rgba(92,72,44,0.12) 0px,"
        " rgba(255,255,255,0.0) 1px,"
        " rgba(255,255,255,0.75) 2px,"
        " rgba(255,255,255,0.15) 4px,"
        " rgba(92,72,44,0.07) 6.5px,"
        " rgba(92,72,44,0.12) 7px)"
    ),
    # Whisper-light reed laid over full panels that carry text (toolbars)
    "light-whisper": (
        "repeating-linear-gradient(90deg,"
        " rgba(92,72,44,0.045) 0px,"
        " rgba(255,255,255,0.0) 1px,"
        " rgba(255,255,255,0.38) 4px,"
        " rgba(255,255,255,0.08) 9px,"
        " rgba(92,72,44,0.025) 15px,"
        " rgba(92,72,44,0.045) 18px)"
    ),
    # Graphite pane, 16px reed (sidebar)
    "dark": (
        "repeating-linear-gradient(90deg,"
        " rgba(0,0,0,0.34) 0px,"
        " rgba(255,255,255,0.00) 1px,"
        " rgba(255,255,255,0.075) 3px,"
        " rgba(255,255,255,0.025) 7px,"
        " rgba(0,0,0,0.12) 12.5px,"
        " rgba(0,0,0,0.30) 15.4px,"
        " rgba(0,0,0,0.34) 16px)"
    ),
    # Graphite pane, 6px reed (primary buttons)
    "dark-fine": (
        "repeating-linear-gradient(90deg,"
        " rgba(0,0,0,0.30) 0px,"
        " rgba(255,255,255,0.0) 1px,"
        " rgba(255,255,255,0.09) 2px,"
        " rgba(0,0,0,0.10) 4.5px,"
        " rgba(0,0,0,0.30) 6px)"
    ),
    # Champagne pane, 8px reed (progress fills, active markers)
    "champagne": (
        "repeating-linear-gradient(90deg,"
        " rgba(110,82,41,0.28) 0px,"
        " rgba(255,255,255,0.0) 1px,"
        " rgba(255,248,232,0.55) 2.5px,"
        " rgba(255,255,255,0.10) 5px,"
        " rgba(110,82,41,0.22) 7.5px,"
        " rgba(110,82,41,0.28) 8px)"
    ),
}


def css_variables() -> str:
    """Return a `:root { ... }` block exposing every token as an `--fg-*` custom property."""
    lines: list[str] = []
    for name, val in COLORS.items():
        lines.append(f"--fg-{name}: {val};")
    for name, val in FONTS.items():
        lines.append(f"--fg-font-{name}: {val};")
    for name, val in TYPE_SCALE.items():
        lines.append(f"--fg-text-{name}: {val};")
    for name, val in RADII.items():
        lines.append(f"--fg-r-{name}: {val};")
    for name, val in SHADOWS.items():
        lines.append(f"--fg-shadow-{name}: {val};")
    for name, val in MOTION.items():
        lines.append(f"--fg-{name}: {val};")
    for name, val in BLUR.items():
        lines.append(f"--fg-blur-{name}: {val};")
    for name, val in FLUTES.items():
        lines.append(f"--fg-flute-{name}: {val};")
    for idx, val in enumerate(PALETTE, start=1):
        lines.append(f"--fg-chart-{idx}: {val};")
    return ":root {\n  " + "\n  ".join(lines) + "\n}"
