"""Interactive CLI menu with per-level Help-and-return and validation reprompts."""
from __future__ import annotations

import shlex
from typing import Any, Callable

from src.cli.runs import validate_run_name
from src.data.schema import BOOKING_PRESETS
from src.registry.artifacts import get_store
from src.registry.registry import list_specs
from src.registry.specs import ModelSpec

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]
DispatchFn = Callable[[list[str]], int]


CATEGORY_HELP = """
=== Category Menu Help ===
Choose an analytical track or utility workflow:
  1. Regression      — Predict ADR (Average Daily Rate in EUR) on paying bookings.
                       Splits: data/preprocessed/ (or custom CSV/XLSX).
                       Outputs: artifacts/regression/<model>/<run_name>/
  2. Classification  — Predict booking cancellation (IsCanceled: 0/1) at booking time.
                       Splits: data/bayes/ (or custom CSV/XLSX).
                       Outputs: artifacts/classification/<model>/<run_name>/
  3. Clustering      — Unsupervised customer segmentation (no target labels used).
                       Data: cleaned H2 City Hotel bookings.
                       Outputs: artifacts/clustering/<model>/<run_name>/
  4. EDA             — Exploratory data analysis tables and plots (outputs/eda/).
  5. Predict         — Run currently deployed models on a preset booking or batch CSV/XLSX.
  6. Launch GUI      — Start the interactive Streamlit ML Studio.

Assumptions:
  - Leakage-prone post-booking columns are blocked automatically per task.
  - Training runs default to a timestamp run name when omitted; reusing an explicit
    --run-name cleanly replaces that run's directory so no stale files remain.
"""


def _model_menu_help(task: str, specs: list[ModelSpec]) -> str:
    lines = [
        f"\n=== {task.title()} Model Menu Help ===",
        f"Registered models for '{task}':",
    ]
    for s in specs:
        opts = ", ".join(o.flag for o in s.cli_options) or "standard flags only"
        lines.append(f"  - {s.display_name} ({s.model_id}) [{s.family}]")
        lines.append(f"      Actions : {', '.join(s.supported_actions)}")
        lines.append(f"      Options : {opts}")
        lines.append(f"      Inputs  : {len(s.required_inputs)} required, {len(s.forbidden_inputs)} forbidden (anti-leakage)")
    lines.append("Select a number to configure a model, 'b' to go back, or 'q' to quit.\n")
    return "\n".join(lines)


def _action_menu_help(spec: ModelSpec) -> str:
    lines = [
        f"\n=== Action Menu Help ({spec.display_name}) ===",
        f"Supported actions for '{spec.model_id}': {', '.join(spec.supported_actions)}",
    ]
    if "train" in spec.supported_actions:
        lines.append("  - train    : Fit model (baseline or tuned preset) + optional model-specific flags.")
        lines.append("               Omit run name for automatic timestamp; explicit name replaces that run cleanly.")
    if "tune" in spec.supported_actions:
        lines.append("  - tune     : Run cross-validated hyperparameter search strictly on training folds.")
    if "evaluate" in spec.supported_actions:
        lines.append("  - evaluate : Score a previously saved run on val/test or custom CSV.")
        lines.append("               Requires an explicit run name (no implicit default).")
    if "cluster" in spec.supported_actions:
        lines.append("  - cluster  : Fit or optimize unsupervised clustering (Silhouette criterion).")
    if spec.cli_options:
        lines.append("\nModel-specific options:")
        for opt in spec.cli_options:
            req = f" (requires {opt.requires})" if opt.requires else ""
            lines.append(f"  {opt.flag:<16} {opt.help}{req}")
    lines.append("")
    return "\n".join(lines)


def _prompt_choice(
    prompt: str,
    valid: set[str],
    help_text: str,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> str:
    while True:
        raw = input_fn(prompt).strip().lower()
        if raw in ("h", "help", "?"):
            output_fn(help_text)
            continue
        if raw in valid:
            return raw
        output_fn(f"Invalid selection '{raw}'. Enter one of {sorted(valid)} or 'h' for help.")


def _prompt_yes_no(
    prompt: str,
    default: bool,
    help_text: str,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = input_fn(f"{prompt} [{hint}]: ").strip().lower()
        if raw in ("h", "help", "?"):
            output_fn(help_text)
            continue
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        output_fn("Please enter 'y', 'n', or 'h' for help.")


def _prompt_run_name(
    required: bool,
    help_text: str,
    input_fn: InputFn,
    output_fn: OutputFn,
    available: list[str] | None = None,
) -> str | None:
    if available:
        output_fn(f"  Available saved runs: {', '.join(available)}")
    suffix = " (required)" if required else " (leave blank for timestamp)"
    while True:
        raw = input_fn(f"Run name{suffix}: ").strip()
        if raw.lower() in ("h", "help", "?"):
            output_fn(help_text)
            continue
        if not raw:
            if required:
                output_fn("Evaluation requires an explicit run name (or 'all'). Try again.")
                continue
            return None
        if required and raw.lower() == "all":
            return "all"
        try:
            return validate_run_name(raw)
        except ValueError as exc:
            output_fn(str(exc))


def _prompt_model_options(
    spec: ModelSpec,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> list[str]:
    """Prompt for declared `spec.cli_options`, enforcing `requires` dependencies."""
    if not spec.cli_options:
        return []
    flags: list[str] = []
    chosen: dict[str, Any] = {}
    for opt in spec.cli_options:
        if opt.requires and not chosen.get(opt.requires):
            continue
        opt_help = f"\nOption {opt.flag}: {opt.help or 'Model-specific option.'}\n"
        if opt.kind == "bool":
            val = _prompt_yes_no(f"Enable {opt.flag} ({opt.help})", bool(opt.default), opt_help, input_fn, output_fn)
            chosen[opt.dest] = val
            if val:
                flags.append(opt.flag)
        elif opt.choices:
            while True:
                raw = input_fn(
                    f"{opt.flag} ({'/'.join(map(str, opt.choices))}) [default={opt.default}]: "
                ).strip()
                if raw.lower() in ("h", "help", "?"):
                    output_fn(opt_help)
                    continue
                if not raw:
                    chosen[opt.dest] = opt.default
                    break
                if raw in opt.choices:
                    chosen[opt.dest] = raw
                    flags.extend([opt.flag, raw])
                    break
                output_fn(f"Invalid value '{raw}'. Choices: {', '.join(map(str, opt.choices))}.")
        else:
            while True:
                default_lbl = "none" if opt.default is None else str(opt.default)
                raw = input_fn(f"{opt.flag} [default={default_lbl}]: ").strip()
                if raw.lower() in ("h", "help", "?"):
                    output_fn(opt_help)
                    continue
                if not raw:
                    chosen[opt.dest] = opt.default
                    break
                try:
                    cast_val = int(raw) if opt.kind == "int" else (float(raw) if opt.kind == "float" else raw)
                except ValueError:
                    output_fn(f"Expected a valid {opt.kind} value for {opt.flag}.")
                    continue
                chosen[opt.dest] = cast_val
                flags.extend([opt.flag, str(cast_val)])
                break
    return flags


def _configure_and_run_action(
    spec: ModelSpec,
    action: str,
    dispatch_fn: DispatchFn,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> int:
    argv: list[str] = []

    if action == "train":
        argv = ["train", "--task", spec.task, "--model", spec.model_id]
        mode_help = "\nMode: 'baseline' uses default parameters; 'tuned_preset' uses validated tuned parameters.\n"
        mode = _prompt_choice(
            "Mode (1=baseline, 2=tuned_preset, h=help) [1]: ",
            {"", "1", "2"},
            mode_help,
            input_fn,
            output_fn,
        )
        if mode == "2":
            argv.extend(["--mode", "tuned_preset"])

        fs_opts = ["original"] + (["pca"] if spec.supports_pca else []) + (["lda"] if spec.supports_lda else [])
        fs_help = f"\nFeature space options for {spec.model_id}: {', '.join(fs_opts)}.\n"
        while True:
            fs = input_fn(f"Feature space ({'/'.join(fs_opts)}) [original]: ").strip().lower()
            if fs in ("h", "help", "?"):
                output_fn(fs_help)
                continue
            if not fs:
                fs = "original"
            if fs in fs_opts:
                break
            output_fn(f"Invalid feature space '{fs}'. Choose from: {', '.join(fs_opts)}.")
        if fs != "original":
            argv.extend(["--feature-space", fs])

        argv.extend(_prompt_model_options(spec, input_fn, output_fn))

        run_name = _prompt_run_name(
            required=False,
            help_text="\nOptional run name. Leave blank to auto-generate a timestamped run name.\n",
            input_fn=input_fn,
            output_fn=output_fn,
        )
        if run_name:
            argv.extend(["--run-name", run_name])

    elif action == "tune":
        argv = ["tune", "--task", spec.task, "--model", spec.model_id]
        method = _prompt_choice(
            "Search method (1=random, 2=grid, h=help) [1]: ",
            {"", "1", "2"},
            "\nrandom = RandomizedSearchCV; grid = exhaustive GridSearchCV.\n",
            input_fn,
            output_fn,
        )
        if method == "2":
            argv.extend(["--method", "grid"])

        run_name = _prompt_run_name(
            required=False,
            help_text="\nOptional run name. Leave blank to auto-generate a timestamped run name.\n",
            input_fn=input_fn,
            output_fn=output_fn,
        )
        if run_name:
            argv.extend(["--run-name", run_name])

    elif action == "evaluate":
        store = get_store()
        saved = [b.card.variant for b in store.list_bundles(task=spec.task, model_id=spec.model_id)]
        run_name = _prompt_run_name(
            required=True,
            help_text="\nEvaluation requires an explicit saved run name (or 'all' to evaluate all runs).\n",
            input_fn=input_fn,
            output_fn=output_fn,
            available=saved,
        )
        split = _prompt_choice(
            "Split to evaluate (1=val, 2=test, h=help) [1]: ",
            {"", "1", "2"},
            "\nval = validation split; test = held-out test split.\n",
            input_fn,
            output_fn,
        )
        argv = [
            "evaluate",
            "--task",
            spec.task,
            "--model",
            spec.model_id,
            "--run-name",
            str(run_name),
            "--split",
            "test" if split == "2" else "val",
        ]

    elif action == "cluster":
        argv = ["cluster", "--algorithm", spec.model_id]
        mode = _prompt_choice(
            "Mode (1=baseline, 2=tune, h=help) [1]: ",
            {"", "1", "2"},
            "\nbaseline = fit single clustering; tune = optimize k via internal Silhouette score.\n",
            input_fn,
            output_fn,
        )
        if mode == "2":
            argv.extend(["--mode", "tune"])
        else:
            while True:
                k_raw = input_fn("Number of clusters k [default=3]: ").strip()
                if k_raw.lower() in ("h", "help", "?"):
                    output_fn("\nInteger number of clusters (>= 2), or leave blank for model default.\n")
                    continue
                if not k_raw:
                    break
                if k_raw.isdigit() and int(k_raw) >= 2:
                    argv.extend(["--k", k_raw])
                    break
                output_fn("Please enter an integer >= 2.")

        while True:
            pca_raw = input_fn("PCA variance ratio or 'none' [0.90]: ").strip().lower()
            if pca_raw in ("h", "help", "?"):
                output_fn("\nEnter 'none' to disable PCA, or a float in (0, 1] such as 0.90.\n")
                continue
            if not pca_raw:
                break
            if pca_raw == "none":
                argv.extend(["--pca", "none"])
                break
            try:
                val = float(pca_raw)
                if 0.0 < val <= 1.0:
                    argv.extend(["--pca", pca_raw])
                    break
            except ValueError:
                pass
            output_fn("Enter 'none' or a float between 0 and 1 (e.g. 0.90).")

        argv.extend(_prompt_model_options(spec, input_fn, output_fn))

        run_name = _prompt_run_name(
            required=False,
            help_text="\nOptional run name. Leave blank to auto-generate a timestamped run name.\n",
            input_fn=input_fn,
            output_fn=output_fn,
        )
        if run_name:
            argv.extend(["--run-name", run_name])

    cmd_preview = "python main.py " + " ".join(shlex.quote(x) for x in argv)
    output_fn(f"\nRunning: {cmd_preview}\n")
    return dispatch_fn(argv)


def _model_menu(
    task: str,
    dispatch_fn: DispatchFn,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> bool:
    """Returns False if user chose Quit, True to return to Category Menu."""
    specs = list_specs(task)
    while True:
        output_fn(f"\n--- {task.title()} Models ---")
        for idx, s in enumerate(specs, start=1):
            output_fn(f"  {idx}. {s.display_name} ({s.model_id}) [{s.family}]")
        output_fn("  h. Help")
        output_fn("  b. Back")
        output_fn("  q. Quit")

        valid = {str(i) for i in range(1, len(specs) + 1)} | {"b", "q"}
        choice = _prompt_choice(
            "Select model: ",
            valid,
            _model_menu_help(task, specs),
            input_fn,
            output_fn,
        )
        if choice == "b":
            return True
        if choice == "q":
            return False

        spec = specs[int(choice) - 1]
        if not _action_menu(spec, dispatch_fn, input_fn, output_fn):
            return False


def _action_menu(
    spec: ModelSpec,
    dispatch_fn: DispatchFn,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> bool:
    """Returns False if user chose Quit, True to return to Model Menu."""
    actions = list(spec.supported_actions)
    while True:
        output_fn(f"\n--- {spec.display_name} ({spec.model_id}) Actions ---")
        for idx, act in enumerate(actions, start=1):
            output_fn(f"  {idx}. {act}")
        output_fn("  h. Help")
        output_fn("  b. Back")
        output_fn("  q. Quit")

        valid = {str(i) for i in range(1, len(actions) + 1)} | {"b", "q"}
        choice = _prompt_choice(
            "Select action: ",
            valid,
            _action_menu_help(spec),
            input_fn,
            output_fn,
        )
        if choice == "b":
            return True
        if choice == "q":
            return False

        action = actions[int(choice) - 1]
        rc = _configure_and_run_action(spec, action, dispatch_fn, input_fn, output_fn)
        if rc != 0:
            output_fn(f"Command exited with status {rc}.")


def run_interactive_menu(
    dispatch_fn: DispatchFn,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> int:
    """Top-level interactive CLI menu: category -> model -> action -> options."""
    output_fn("\nCity Hotel Booking Demand — Interactive CLI")
    output_fn("Type 'h' at any menu for contextual help, or 'q' to quit.")

    while True:
        output_fn("\n=== Main Menu ===")
        output_fn("  1. Regression (ADR Prediction)")
        output_fn("  2. Classification (Cancellation Prediction)")
        output_fn("  3. Clustering (Customer Segmentation)")
        output_fn("  4. Exploratory Data Analysis (EDA)")
        output_fn("  5. Predict (Preset Booking)")
        output_fn("  6. Launch Streamlit GUI")
        output_fn("  h. Help")
        output_fn("  q. Quit")

        choice = _prompt_choice(
            "Select category: ",
            {"1", "2", "3", "4", "5", "6", "q"},
            CATEGORY_HELP,
            input_fn,
            output_fn,
        )
        if choice == "q":
            output_fn("Goodbye.")
            return 0
        if choice == "1":
            if not _model_menu("regression", dispatch_fn, input_fn, output_fn):
                return 0
        elif choice == "2":
            if not _model_menu("classification", dispatch_fn, input_fn, output_fn):
                return 0
        elif choice == "3":
            if not _model_menu("clustering", dispatch_fn, input_fn, output_fn):
                return 0
        elif choice == "4":
            output_fn("\nRunning: python main.py analyze\n")
            dispatch_fn(["analyze"])
        elif choice == "5":
            presets = list(BOOKING_PRESETS.keys())
            output_fn("\n--- Booking Presets ---")
            for idx, name in enumerate(presets, start=1):
                output_fn(f"  {idx}. {name}")
            p_choice = _prompt_choice(
                "Select preset [1]: ",
                {""} | {str(i) for i in range(1, len(presets) + 1)},
                "\nChoose a sample booking profile to score across all deployed models.\n",
                input_fn,
                output_fn,
            )
            preset_name = presets[int(p_choice) - 1] if p_choice else presets[0]
            output_fn(f"\nRunning: python main.py predict --task all --preset {shlex.quote(preset_name)}\n")
            dispatch_fn(["predict", "--task", "all", "--preset", preset_name])
        elif choice == "6":
            output_fn("\nRunning: python main.py gui\n")
            return dispatch_fn(["gui"])
