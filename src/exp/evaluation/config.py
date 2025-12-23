"""Configuration dataclasses for the hydra modules."""

from dataclasses import dataclass
from typing import Union


@dataclass
class Model:
    """Model configuration."""

    model_name: str
    model_name_short: str
    gpu_memory_utilization: float
    temperature: float
    max_tokens: int
    max_model_len: int
    top_p: float = 0.95
    tensor_parallel_size: int = 1
    quantization: str | None = None
    tool_call_parser: str | None = None


@dataclass
class Dataset:
    """Dataset configuration."""

    name: str
    split: str
    sys_prompt_path: str
    input_dir: str
    subset: str


@dataclass
class MLFlowConfig:
    """MLFlow configuration."""

    experiment_id: str
    uri: str


@dataclass
class VectorDB:
    """VectorDB configuration."""

    collection_name: str
    top_k: int
    embedding_function: str = "default"


@dataclass
class RAGConfig:
    """RAG configuration."""

    rag_method: str


@dataclass
class Config:
    """Configuration dataclass for the hydra modules."""

    model: Model
    dataset: Dataset
    mlflow: MLFlowConfig
    vector_db: VectorDB
    rag: RAGConfig
    metrics: list[Union[str, dict[str, dict[str, str]]]]
    output_folder: str
    template_name: str
    vllm_port: int
    base_url: str
