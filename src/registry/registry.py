"""Central model registry populated by regression, classification, and clustering tracks."""
from __future__ import annotations

from src.registry.specs import ModelSpec, TaskType

REGISTRY: dict[tuple[str, str], ModelSpec] = {}
_INITIALIZED = False


def register_model(spec: ModelSpec) -> ModelSpec:
    REGISTRY[(spec.task, spec.model_id)] = spec
    return spec


def ensure_all_registered() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    # Import each track's models module so its ModelSpecs self-register.
    from src.classification import models as _clf  # noqa: F401
    from src.clustering import models as _clu  # noqa: F401
    from src.regression import models as _reg  # noqa: F401

    _INITIALIZED = True


def get_spec(task: TaskType | str, model_id: str) -> ModelSpec:
    ensure_all_registered()
    key = (str(task), str(model_id))
    if key not in REGISTRY:
        available = [m for (t, m) in REGISTRY if t == task]
        raise KeyError(f"Unknown model '{model_id}' for task '{task}'. Available: {available}")
    return REGISTRY[key]


def list_specs(task: TaskType | str | None = None) -> list[ModelSpec]:
    ensure_all_registered()
    specs = list(REGISTRY.values())
    if task is not None:
        specs = [s for s in specs if s.task == task]
    return specs
