"""Model specification dataclasses driving the GUI catalog, forms, and training service."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

TaskType = Literal["regression", "classification", "clustering"]
ParamKind = Literal["int", "float", "log_float", "choice", "bool"]
OptionKind = Literal["bool", "int", "float", "str"]
FeatureSpace = Literal["original", "pca", "lda"]


@dataclass(frozen=True)
class ModelOptionSpec:
    """Model-specific CLI option exposed only for models that declare it."""

    flag: str
    dest: str
    kind: OptionKind = "bool"
    default: Any = False
    choices: tuple[Any, ...] = field(default_factory=tuple)
    help: str = ""
    requires: str | None = None


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    kind: ParamKind
    default: Any
    low: float | None = None
    high: float | None = None
    step: float | None = None
    choices: tuple[Any, ...] = field(default_factory=tuple)
    help: str = ""


@dataclass
class ModelSpec:
    model_id: str
    task: TaskType
    display_name: str
    family: str
    description: str
    assumptions_and_pros: str
    required_inputs: list[str]
    forbidden_inputs: list[str]
    params: list[ParamSpec]
    baseline_params: dict[str, Any]
    tuned_params: dict[str, Any]
    search_space: dict[str, list[Any]]
    default_scoring: str
    build_pipeline: Callable[[dict[str, Any], FeatureSpace, float | int | None], Any]
    supports_proba: bool = False
    supports_predict_new: bool = True
    supports_pca: bool = True
    supports_lda: bool = False
    cost_hint: Literal["fast", "medium", "slow"] = "fast"
    recommended_max_rows: int | None = None
    track_origin: Literal["milestone_1", "milestone_2_suite"] = "milestone_2_suite"
    supported_actions: tuple[str, ...] = ("train", "tune", "evaluate")
    cli_options: list[ModelOptionSpec] = field(default_factory=list)

    def default_param_dict(self) -> dict[str, Any]:
        out = {p.name: p.default for p in self.params}
        out.update(self.baseline_params)
        return out
