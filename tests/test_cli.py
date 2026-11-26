"""Tests for CLI Features 1 (interactive menu + Help), 2 (named runs & clean replacement), and 3 (model-specific options)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main as cli_mod
from src.cli.interactive import run_interactive_menu
from src.cli.runs import resolve_run_name, validate_model_options, validate_run_name
from src.data.schema import BOOKING_PRESETS
from src.registry.artifacts import ArtifactStore
from src.registry.registry import get_spec


def _synthetic_csv(tmp_path: Path, n: int = 150) -> Path:
    rng = np.random.default_rng(42)
    presets = list(BOOKING_PRESETS.values())
    rows = []
    for i in range(n):
        base = dict(presets[i % len(presets)])
        base["LeadTime"] = int(rng.integers(0, 320))
        base["StaysInWeekNights"] = int(rng.integers(1, 6))
        base["StaysInWeekendNights"] = int(rng.integers(0, 3))
        base["Adults"] = int(rng.integers(1, 4))
        base["ADR"] = float(round(70.0 + 15.0 * base["Adults"] + 0.1 * base["LeadTime"] + rng.normal(0, 6), 2))
        base["IsCanceled"] = int(rng.random() < 0.4)
        base.setdefault("Company", "null")
        rows.append(base)
    csv_path = tmp_path / "synthetic.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Feature 2: Run names, timestamp defaults, scoped clean replacement, evaluate
# ---------------------------------------------------------------------------

def test_validate_run_name_rules():
    assert validate_run_name("test_run-23.v1") == "test_run-23.v1"
    for bad in ("", "   ", "../evil", "a/b", "a\\b", "-starts_dash", "has space"):
        with pytest.raises(ValueError):
            validate_run_name(bad)


def test_timestamp_defaults_do_not_overwrite_and_explicit_name_replaces_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = ArtifactStore(root_dir=tmp_path / "artifacts")
    monkeypatch.setattr(cli_mod, "get_store", lambda: store)
    csv_path = _synthetic_csv(tmp_path)

    # Two training runs without --run-name -> two distinct timestamped bundles
    rc1 = cli_mod.main(["train", "--task", "classification", "--model", "gaussian_nb", "--data", str(csv_path)])
    rc2 = cli_mod.main(["train", "--task", "classification", "--model", "gaussian_nb", "--data", str(csv_path)])
    assert rc1 == 0 and rc2 == 0
    bundles = store.list_bundles(task="classification", model_id="gaussian_nb")
    assert len(bundles) == 2
    assert bundles[0].card.variant != bundles[1].card.variant

    # Explicit --run-name creates a deterministic run dir
    rc3 = cli_mod.main([
        "train", "--task", "classification", "--model", "gaussian_nb",
        "--data", str(csv_path), "--run-name", "exp_alpha",
    ])
    assert rc3 == 0
    alpha_bundles = [b for b in store.list_bundles(task="classification") if b.card.variant == "exp_alpha"]
    assert len(alpha_bundles) == 1
    stale_file = alpha_bundles[0].bundle_dir / "extras" / "stale_marker.txt"
    stale_file.write_text("old", encoding="utf-8")
    assert stale_file.exists()

    # Re-running with the same --run-name cleanly replaces ONLY that run (no stale files, other runs untouched)
    rc4 = cli_mod.main([
        "train", "--task", "classification", "--model", "gaussian_nb",
        "--data", str(csv_path), "--run-name", "exp_alpha", "--calibrate",
    ])
    assert rc4 == 0
    assert not stale_file.exists()
    assert len(store.list_bundles(task="classification", model_id="gaussian_nb")) == 3

    # Evaluate requires --run-name; explicit name evaluates only that run
    with pytest.raises(SystemExit):
        cli_mod.main(["evaluate", "--task", "classification", "--model", "gaussian_nb"])

    rc_missing = cli_mod.main([
        "evaluate", "--task", "classification", "--model", "gaussian_nb",
        "--run-name", "does_not_exist", "--data", str(csv_path),
    ])
    assert rc_missing == 1

    rc_ok = cli_mod.main([
        "evaluate", "--task", "classification", "--model", "gaussian_nb",
        "--run-name", "exp_alpha", "--data", str(csv_path),
    ])
    assert rc_ok == 0


# ---------------------------------------------------------------------------
# Feature 3: Model-specific options & conflict validation
# ---------------------------------------------------------------------------

def test_model_specific_option_validation_and_conflicts():
    reg_spec = get_spec("regression", "linear_regression")
    clf_spec = get_spec("classification", "gaussian_nb")

    # --with-optional requires --variants
    with pytest.raises(ValueError, match="requires '--variants'"):
        validate_model_options(reg_spec, {"variants": False, "with_optional": True})

    # --variants is not supported on gaussian_nb
    with pytest.raises(ValueError, match="not supported for model 'gaussian_nb'"):
        validate_model_options(clf_spec, {"variants": True})

    # Valid gaussian_nb options pass
    opts = validate_model_options(clf_spec, {"calibrate": True, "var_smoothing": 1e-4, "threshold": 0.55})
    assert opts["calibrate"] is True and opts["var_smoothing"] == 1e-4 and opts["threshold"] == 0.55


def test_cli_linear_regression_variants_option(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = ArtifactStore(root_dir=tmp_path / "artifacts")
    monkeypatch.setattr(cli_mod, "get_store", lambda: store)
    csv_path = _synthetic_csv(tmp_path, n=180)

    # Conflict exits with status 2
    rc_bad = cli_mod.main([
        "train", "--task", "regression", "--model", "linear_regression",
        "--with-optional", "--data", str(csv_path),
    ])
    assert rc_bad == 2

    # Valid --variants run saves variant tables into the run's extras/
    rc = cli_mod.main([
        "train", "--task", "regression", "--model", "linear_regression",
        "--variants", "--run-name", "ablation_test", "--data", str(csv_path),
    ])
    assert rc == 0
    bundles = [b for b in store.list_bundles(task="regression") if b.card.variant == "ablation_test"]
    assert len(bundles) == 1
    summary = bundles[0].read_csv("variant_summary.csv")
    assert summary is not None and len(summary) == 5


# ---------------------------------------------------------------------------
# Feature 1: Interactive menu navigation, Help-and-return, invalid reprompting
# ---------------------------------------------------------------------------

def test_interactive_menu_help_reprompt_and_dispatch():
    dispatched: list[list[str]] = []
    outputs: list[str] = []

    # Sequence:
    # 1. Invalid choice '9' -> reprompt
    # 2. 'h' -> Category Help, stays on Category menu
    # 3. '1' -> Regression
    # 4. 'h' -> Regression Model Help, stays on Model menu
    # 5. '1' -> Linear Regression
    # 6. 'h' -> Action Help, stays on Action menu
    # 7. '1' -> train
    # 8. '1' -> baseline
    # 9. 'original' -> feature space
    # 10. 'y' -> enable --variants
    # 11. 'n' -> disable --with-optional
    # 12. '' -> default --tol
    # 13. 'my_interactive_run' -> --run-name
    # 14. 'b' -> back to Model menu
    # 15. 'b' -> back to Category menu
    # 16. '3' -> Clustering
    # 17. '1' -> KMeans (should only have 1 action: cluster)
    # 18. 'q' -> quit
    inputs = iter([
        "9", "h", "1", "h", "1", "h", "1",
        "1", "original", "y", "n", "", "my_interactive_run",
        "b", "b", "3", "1", "q",
    ])

    rc = run_interactive_menu(
        dispatch_fn=lambda argv: (dispatched.append(argv), 0)[1],
        input_fn=lambda _prompt: next(inputs),
        output_fn=outputs.append,
    )
    assert rc == 0
    full_out = "\n".join(outputs)
    assert "Invalid selection '9'" in full_out
    assert "=== Category Menu Help ===" in full_out
    assert "=== Regression Model Menu Help ===" in full_out
    assert "=== Action Menu Help (Linear Regression (OLS)) ===" in full_out
    assert "1. cluster" in full_out
    assert dispatched == [[
        "train", "--task", "regression", "--model", "linear_regression",
        "--variants", "--run-name", "my_interactive_run",
    ]]


def test_interactive_evaluate_requires_run_name():
    dispatched: list[list[str]] = []
    outputs: list[str] = []
    # Main(1=Reg) -> Model(1=LR) -> Action(3=evaluate) -> blank run name (reprompts!) -> 'run_42' -> '2'(test) -> quit
    inputs = iter(["1", "1", "3", "", "run_42", "2", "q"])

    rc = run_interactive_menu(
        dispatch_fn=lambda argv: (dispatched.append(argv), 0)[1],
        input_fn=lambda _prompt: next(inputs),
        output_fn=outputs.append,
    )
    assert rc == 0
    assert any("Evaluation requires an explicit run name" in line for line in outputs)
    assert dispatched == [[
        "evaluate", "--task", "regression", "--model", "linear_regression",
        "--run-name", "run_42", "--split", "test",
    ]]
