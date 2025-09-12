# utils/cache.py
import redis
import json
import hashlib
from typing import Any, Optional, List
from datetime import timedelta
import pickle


class CacheManager:
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1小时

    def _make_key(self, prefix: str, *args) -> str:
        key_data = "|".join(str(arg) for arg in args)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"

    def get_query_results(self, query: str) -> Optional[List[Any]]:
        key = self._make_key("query", query)
        try:
            cached = self.redis_client.get(key)
            if cached:
                return pickle.loads(cached)
        except Exception as e:
            logger.log_error(e, {"operation": "cache_get", "key": key})
        return None

    def cache_query_results(self, query: str, results: List[Any], ttl: int = None):
        key = self._make_key("query", query)
        try:
            self.redis_client.setex(
                key,
                ttl or self.default_ttl,
                pickle.dumps(results)
            )
        except Exception as e:
            logger.log_error(e, {"operation": "cache_set", "key": key})

    def get_embeddings(self, text: str) -> Optional[List[float]]:
        key = self._make_key("embedding", text)
        try:
            cached = self.redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.log_error(e, {"operation": "embedding_cache_get"})
        return None

    def cache_embeddings(self, text: str, embedding: List[float]):
        key = self._make_key("embedding", text)
        try:
            self.redis_client.setex(
                key,
                86400,  # 24小时
                json.dumps(embedding)
            )
        except Exception as e:
            logger.log_error(e, {"operation": "embedding_cache_set"})


cache_manager = CacheManager()
