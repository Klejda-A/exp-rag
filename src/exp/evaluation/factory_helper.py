"""Helper functions for the evaluation factory."""

from encourage.llm import BatchInferenceRunner
from encourage.metrics import METRIC_REGISTRY, get_metric_from_registry
from encourage.rag import RAGMethod, RAGMethodInterface
from omegaconf import DictConfig, OmegaConf


def get_rag_method_class(method_name: str) -> type[RAGMethodInterface]:
    """Get the RAG method class from a string."""
    try:
        return RAGMethod(method_name).get_class()  # type: ignore
    except KeyError:
        raise ValueError(f"Unsupported RAG method: {method_name}") from None


def load_metrics(
    config: list[str | dict[str, dict[str, str]]], runner: BatchInferenceRunner = None
) -> list:
    """Load metrics from the config."""
    metrics = []
    for m in config:
        if isinstance(m, str):
            name, args = m, {}  # type: ignore
        elif isinstance(m, (dict, DictConfig)):
            # Convert DictConfig to plain dict
            m = OmegaConf.to_container(m, resolve=True)  # type: ignore
            name, args = next(iter(m.items()))
        else:
            raise ValueError(f"Invalid metric config: {m}")

        cls = METRIC_REGISTRY[name.lower()]

        if cls.requires_runner():
            metric = get_metric_from_registry(name, runner=runner, **args)
        else:
            metric = get_metric_from_registry(name, **args)

        metrics.append(metric)
    return metrics
