# config/settings.py
from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path


# @dataclass
# class ModelConfig:
#     embed_model_path: str
#     llm_model_path: str
#     rerank_model_path: str
#     device: str = "auto"
#     max_length: int = 512
#     batch_size: int = 32
@dataclass
class ModelConfig:
    embed_model_path: str = "/path/to/embed/model" # 默认值，实际应从环境变量或启动时传入
    llm_model_path: str = "/path/to/llm/model"     # 默认值，实际应从环境变量或启动时传入
    rerank_model_path: str = "/path/to/rerank/model" # 默认值，实际应从环境变量或启动时传入
    device: str = "auto"
    max_length: int = 512
    batch_size: int = 32 # 用于Embedding和Rerank的默认批量大小

    # vLLM specific settings (removed)
    use_vllm: bool = False # 是否使用vLLM作为LLM  # ✅ 确保有这个属性
    # vllm_tensor_parallel_size: int = 1
    # vllm_gpu_memory_utilization: float = 0.9
    # vllm_max_model_len: Optional[int] = None # 如果不设置，vLLM会尝试自动检测
    # vllm_temperature: float = 0.7
    # vllm_top_p: float = 0.95
    # vllm_max_new_tokens: int = 512
    # vllm_stop_sequences: Optional[List[str]] = field(default_factory=list) # 例如 ["\nUser:", "\nAssistant:"]


@dataclass
class VectorStoreConfig:
    uri: str = "http://localhost:19530"
    collection_name: str = "enterprise_rag"
    dim: int = 1024
    index_type: str = "HNSW"
    metric_type: str = "COSINE"


@dataclass
class RetrievalConfig:
    similarity_top_k: int = 20
    rerank_top_k: int = 8
    score_threshold: float = 0.4
    min_good_results: int = 2
    max_context_chars: int = 1800
    chunk_size: int = 512
    chunk_overlap: int = 50
    batch_size: int = 32


@dataclass
class AppConfig:
    data_dir: str = "data"
    log_level: str = "INFO"
    cache_dir: str = ".cache"
    max_query_length: int = 200
    rate_limit: int = 100  # requests per minute
    redis_url: str = "redis://localhost:6379/0" # Redis连接URL

    # FastAPI settings
    api_host: str = "127.0.0.1"  # ✅ 确保有这个属性
    api_port: int = 8000        # ✅ 确保有这个属性

class Settings:
    def __init__(self, config_path: Optional[str] = None):
        self.model = ModelConfig(
            embed_model_path=os.getenv("EMBED_MODEL_PATH", "/path/to/embed/model"),
            llm_model_path=os.getenv("LLM_MODEL_PATH", "/path/to/llm/model"),
            rerank_model_path=os.getenv("RERANK_MODEL_PATH", "/path/to/rerank/model")
        )
        self.vector_store = VectorStoreConfig()
        self.retrieval = RetrievalConfig()
        self.app = AppConfig()
