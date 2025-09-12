# services/retriever.py
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from llama_index.core.schema import NodeWithScore
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from utils.cache import cache_manager
from utils.logger import logger
from config.settings import RetrievalConfig
import torch
import json

@dataclass
class RetrievalResult:
    nodes: List[NodeWithScore]
    retrieval_time: float
    cache_hit: bool
    method_used: str
    total_candidates: int


class EnterpriseRetriever:
    def __init__(
            self,
            vector_retriever,
            documents: List,
            rerank_model_path: str,
            config: RetrievalConfig
    ):
        self.vector_retriever = vector_retriever
        self.documents = documents
        self.config = config

        # 初始化BM25
        self.bm25 = BM25Okapi([doc.text.split() for doc in documents])

        # 初始化重排序模型
        try:
            self.reranker = CrossEncoder(
                rerank_model_path,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.log_error(e, {"model_path": rerank_model_path})
            raise

    def _expand_query(self, query: str, llm, max_variants: int = 2) -> List[str]:
        """查询扩展"""
        cache_key = f"query_expansion:{query}"

        # 检查缓存
        cached_expansions = cache_manager.redis_client.get(cache_key)
        if cached_expansions:
            return json.loads(cached_expansions)

        prompt = f"""请将以下问题改写为{max_variants}个意思相近但表达不同的中文问法。
        要求：
        1. 保持原意不变
        2. 使用不同的词汇和句式
        3. 每行一个问法
        4. 不要添加解释

        原问题：{query}

        改写后的问法："""

        try:
            response = llm.complete(prompt)
            variants = [v.strip() for v in response.text.split("\n") if v.strip()]
            expansions = [query] + variants[:max_variants]

            # 缓存结果
            cache_manager.redis_client.setex(cache_key, 3600, json.dumps(expansions))

            return expansions

        except Exception as e:
            logger.log_error(e, {"query": query})
            return [query]

    def _dense_retrieve(self, query: str) -> List[NodeWithScore]:
        """密集检索"""
        try:
            return self.vector_retriever.retrieve(query)
        except Exception as e:
            logger.log_error(e, {"query": query, "method": "dense_retrieve"})
            return []

    def _sparse_retrieve(self, query: str, top_k: int) -> List[NodeWithScore]:
        """稀疏检索（BM25）"""
        try:
            bm25_scores = self.bm25.get_scores(query.split())
            top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True
            )[:top_k * 2]

            results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    # 创建NodeWithScore对象
                    node_with_score = NodeWithScore(
                        node=doc,
                        score=float(bm25_scores[idx])
                    )
                    results.append(node_with_score)

            return results

        except Exception as e:
            logger.log_error(e, {"query": query, "method": "sparse_retrieve"})
            return []

    def _rerank_results(self, query: str, results: List[NodeWithScore]) -> List[NodeWithScore]:
        """重排序结果"""
        if not results:
            return results

        try:
            pairs = [(query, result.node.text) for result in results]
            scores = self.reranker.predict(pairs, batch_size=self.config.batch_size)

            # 更新分数并排序
            for result, score in zip(results, scores):
                result.score = float(score)

            return sorted(results, key=lambda x: x.score, reverse=True)

        except Exception as e:
            logger.log_error(e, {"query": query, "num_results": len(results)})
            return results

    def hybrid_retrieve(
            self,
            query: str,
            llm=None,
            use_cache: bool = True,
            debug: bool = False
    ) -> RetrievalResult:
        """混合检索"""
        start_time = time.time()
        cache_hit = False

        # 检查缓存
        if use_cache:
            cached_result = cache_manager.get_query_results(query)
            if cached_result:
                cache_hit = True
                retrieval_time = time.time() - start_time
                logger.log_retrieval(query, len(cached_result), retrieval_time)

                return RetrievalResult(
                    nodes=cached_result,
                    retrieval_time=retrieval_time,
                    cache_hit=True,
                    method_used="cache",
                    total_candidates=len(cached_result)
                )

        # 合并候选结果
        merged_results = {}

        # 1. 密集检索
        dense_results = self._dense_retrieve(query)
        if debug:
            logger.logger.info(f"Dense retrieval: {len(dense_results)} results")

        for result in dense_results:
            merged_results[result.node.node_id] = result

        # 2. 稀疏检索
        sparse_results = self._sparse_retrieve(query, self.config.similarity_top_k)
        if debug:
            logger.logger.info(f"Sparse retrieval: {len(sparse_results)} results")

        for result in sparse_results:
            if result.node.node_id not in merged_results:
                merged_results[result.node.node_id] = result

        # 3. 初步重排序检查
        initial_results = list(merged_results.values())
        reranked_results = self._rerank_results(query, initial_results)

        top_scores = [r.score for r in reranked_results[:self.config.rerank_top_k]]
        good_results_count = sum(1 for score in top_scores if score >= self.config.score_threshold)

        if debug:
            logger.logger.info(
                f"Initial rerank: top score = {max(top_scores) if top_scores else 0:.4f}, good results = {good_results_count}")

        # 4. 查询扩展（如果需要）
        method_used = "hybrid"
        if good_results_count < self.config.min_good_results and llm:
            if debug:
                logger.logger.info("Triggering query expansion...")

            method_used = "hybrid_expanded"
            expanded_queries = self._expand_query(query, llm)

            for expanded_query in expanded_queries[1:]:  # 跳过原查询
                expanded_dense = self._dense_retrieve(expanded_query)
                for result in expanded_dense:
                    if result.node.node_id not in merged_results:
                        merged_results[result.node.node_id] = result

        # 5. 最终重排序
        final_results = list(merged_results.values())
        final_reranked = self._rerank_results(query, final_results)

        # 过滤低分结果
        filtered_results = [
                               r for r in final_reranked
                               if r.score >= self.config.score_threshold
                           ][:self.config.rerank_top_k]

        retrieval_time = time.time() - start_time

        # 缓存结果
        if use_cache and filtered_results:
            cache_manager.cache_query_results(query, filtered_results)

        logger.log_retrieval(query, len(filtered_results), retrieval_time)

        return RetrievalResult(
            nodes=filtered_results,
            retrieval_time=retrieval_time,
            cache_hit=cache_hit,
            method_used=method_used,
            total_candidates=len(final_results)
        )
