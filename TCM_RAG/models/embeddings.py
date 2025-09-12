# models/embeddings.py
import torch
import torch.nn.functional as F
from typing import List, Any
from transformers import AutoTokenizer, AutoModel
from llama_index.core.embeddings import BaseEmbedding
from utils.cache import cache_manager
from utils.logger import logger
import time


class EnterpriseEmbedding(BaseEmbedding):
    """企业级自定义Embedding类，支持本地HF模型加载和Redis缓存"""

    # ===== 声明Pydantic字段（必须显式声明，否则直接 self.xxx 会报错） =====
    tokenizer: Any = None
    model: Any = None
    device: str = None
    embed_dim: int = 0
    max_length: int = 512
    batch_size: int = 32
    use_cache: bool = True

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 32,
        use_cache: bool = True,
        **kwargs: Any
    ):
        """
        初始化本地Embedding模型
        """
        # ✅ 先调用父类构造，初始化Pydantic内部
        super().__init__(embed_dim=0, **kwargs)

        # ✅ 用 object.__setattr__ 绕过Pydantic限制设置属性
        object.__setattr__(self, "max_length", max_length)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "use_cache", use_cache)

        # 设备选择
        if device == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = device

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device_str)
            model.eval()
            embed_dim = model.config.hidden_size

            # ✅ 保存模型与设备信息
            object.__setattr__(self, "tokenizer", tokenizer)
            object.__setattr__(self, "model", model)
            object.__setattr__(self, "device", device_str)
            object.__setattr__(self, "embed_dim", embed_dim)

            logger.logger.info(f"Embedding model loaded on {device_str}")

        except Exception as e:
            logger.log_error(e, {"model_path": model_path})
            raise

    @classmethod
    def class_name(cls) -> str:
        return "EnterpriseEmbedding"

    def _embed_batch(self, sentences: List[str]) -> List[List[float]]:
        """批量嵌入，支持缓存"""
        if self.use_cache:
            cached_results = []
            uncached_sentences = []
            uncached_indices = []

            for i, sentence in enumerate(sentences):
                cached = cache_manager.get_embeddings(sentence)
                if cached:
                    cached_results.append((i, cached))
                else:
                    uncached_sentences.append(sentence)
                    uncached_indices.append(i)

            if not uncached_sentences:
                return [result for _, result in sorted(cached_results)]
        else:
            uncached_sentences = sentences
            uncached_indices = list(range(len(sentences)))
            cached_results = []

        # 处理未缓存的句子
        start_time = time.time()
        try:
            encoded_input = self.tokenizer(
                uncached_sentences,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length
            )
            encoded_input = {k: v.to(self.model.device) for k, v in encoded_input.items()}

            with torch.no_grad():
                model_output = self.model(**encoded_input)
                sentence_embeddings = model_output[0][:, 0]  # 取 CLS token
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            embeddings = sentence_embeddings.cpu().tolist()

            # 缓存结果
            if self.use_cache:
                for sentence, embedding in zip(uncached_sentences, embeddings):
                    cache_manager.cache_embeddings(sentence, embedding)

            logger.logger.info(
                f"Embedded {len(uncached_sentences)} sentences in {time.time() - start_time:.2f}s"
            )

        except Exception as e:
            logger.log_error(e, {"num_sentences": len(uncached_sentences)})
            raise

        # 合并缓存和新计算结果
        if self.use_cache:
            all_results = cached_results + [(idx, emb) for idx, emb in zip(uncached_indices, embeddings)]
            return [result for _, result in sorted(all_results)]
        else:
            return embeddings

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        # 分批处理
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
