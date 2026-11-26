"""Run-name validation, timestamp defaults, model-specific option checks, and bundle lookup."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from src.registry.artifacts import ArtifactBundle, ArtifactStore, get_store
from src.registry.specs import ModelSpec

RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def validate_run_name(name: str) -> str:
    """Validate an explicit run name to prevent path traversal or invalid folder names."""
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Run name cannot be empty.")
    if ".." in cleaned or "/" in cleaned or "\\" in cleaned:
        raise ValueError(f"Invalid run name '{name}': path separators and '..' are not allowed.")
    if not RUN_NAME_RE.match(cleaned):
        raise ValueError(
            f"Invalid run name '{name}': use 1-80 characters starting with a letter/digit "
            "and containing only letters, digits, '.', '_', or '-'."
        )
    return cleaned


def resolve_run_name(
    explicit_name: str | None,
    prefix: str,
    task: str,
    model_id: str,
    store: ArtifactStore | None = None,
) -> tuple[str, bool]:
    """Return (run_name, is_explicit). Generates a collision-free timestamp name when omitted."""
    if explicit_name is not None and explicit_name.strip():
        return validate_run_name(explicit_name), True

    store = store or get_store()
    now = datetime.now()
    base = f"{prefix}_{now.strftime('%Y%m%d_%H%M%S')}"
    existing = {b.card.variant for b in store.list_bundles(task=task, model_id=model_id)}
    if base not in existing:
        return base, False
    candidate = f"{base}_{now.strftime('%f')[:3]}"
    idx = 1
    while candidate in existing:
        candidate = f"{base}_{idx:02d}"
        idx += 1
    return candidate, False


def validate_model_options(
    spec: ModelSpec,
    provided: dict[str, Any],
) -> dict[str, Any]:
    """Validate model-specific CLI options against `spec.cli_options`.

    Raises ValueError on unsupported flags, invalid choices, or unmet `requires` dependencies.
    """
    allowed = {opt.dest: opt for opt in spec.cli_options}
    resolved: dict[str, Any] = {}

    for dest, val in provided.items():
        if val is None or val is False:
            continue
        if dest not in allowed:
            flag_hint = "--" + dest.replace("_", "-")
            supported = ", ".join(o.flag for o in spec.cli_options) or "none"
            raise ValueError(
                f"Option '{flag_hint}' is not supported for model '{spec.model_id}' "
                f"(supported model-specific options: {supported})."
            )
        opt = allowed[dest]
        if opt.choices and val not in opt.choices:
            raise ValueError(
                f"Invalid value '{val}' for {opt.flag}. Allowed choices: {', '.join(map(str, opt.choices))}."
            )
        resolved[dest] = val

    # Check `requires` dependencies (e.g. --with-optional / --tol require --variants)
    for dest, opt in allowed.items():
        if opt.requires and dest in resolved and not resolved.get(opt.requires):
            req_flag = allowed[opt.requires].flag if opt.requires in allowed else f"--{opt.requires}"
            raise ValueError(f"Option '{opt.flag}' requires '{req_flag}' to be enabled.")

    # Fill defaults for declared options when not explicitly passed
    for dest, opt in allowed.items():
        if dest not in resolved:
            resolved[dest] = opt.default

    return resolved


def find_bundles_by_run_name(
    store: ArtifactStore,
    task: str,
    run_name: str,
    model_id: str | None = None,
) -> list[ArtifactBundle]:
    """Find saved bundles for `task` matching an explicit `--run-name` (or 'all')."""
    bundles = store.list_bundles(task=task, model_id=model_id)
    if run_name.strip().lower() == "all":
        return bundles
    target = validate_run_name(run_name)
    return [b for b in bundles if b.card.variant == target or b.card.run_name == target]
