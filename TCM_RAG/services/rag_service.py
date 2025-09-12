# services/rag_service.py
import time
from typing import Dict, Any, Optional
from datetime import datetime

from config.settings import Settings
from models.embeddings import EnterpriseEmbedding
from services.document_processor import DocumentProcessor
from services.retriever import EnterpriseRetriever
from services.answer_generator import AnswerGenerator
from utils.logger import logger
from utils.metrics import metrics_collector, QueryMetrics
from utils.cache import cache_manager

from llama_index.core import Settings as LlamaSettings, VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext
from llama_index.llms.huggingface import HuggingFaceLLM


class EnterpriseRAGService:
    def __init__(self, config_path: Optional[str] = None):
        self.settings = Settings(config_path)
        self.is_initialized = False

        # 组件初始化
        self.embed_model = None
        self.llm = None
        self.retriever = None
        self.answer_generator = None
        self.vector_store = None
        self.index = None

        logger.logger.info("EnterpriseRAGService initialized")

    def initialize(self):
        """初始化所有组件"""
        try:
            logger.logger.info("Starting RAG service initialization...")

            # 1. 初始化嵌入模型
            self.embed_model = EnterpriseEmbedding(
                model_path=self.settings.model.embed_model_path,
                device=self.settings.model.device,
                max_length=self.settings.model.max_length,
                batch_size=self.settings.model.batch_size
            )
            LlamaSettings.embed_model = self.embed_model

            # 2. 初始化LLM
            self.llm = HuggingFaceLLM(
                model_name=self.settings.model.llm_model_path,
                tokenizer_name=self.settings.model.llm_model_path,
                device_map="auto",
                model_kwargs={"trust_remote_code": True, "torch_dtype": "auto"},
                tokenizer_kwargs={"trust_remote_code": True}
            )
            LlamaSettings.llm = self.llm

            # 3. 处理文档
            doc_processor = DocumentProcessor(
                chunk_size=self.settings.retrieval.chunk_size,
                chunk_overlap=self.settings.retrieval.chunk_overlap
            )
            documents = doc_processor.process_directory(self.settings.app.data_dir)
            nodes = doc_processor.create_nodes(documents)

            # 4. 初始化向量存储
            self.vector_store = MilvusVectorStore(
                uri=self.settings.vector_store.uri,
                collection_name=self.settings.vector_store.collection_name,
                dim=self.embed_model.embed_dim,
                overwrite=True
            )

            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.index = VectorStoreIndex(nodes, storage_context=storage_context)
            vector_retriever = self.index.as_retriever(
                similarity_top_k=self.settings.retrieval.similarity_top_k
            )

            # 5. 初始化检索器
            self.retriever = EnterpriseRetriever(
                vector_retriever=vector_retriever,
                documents=documents,
                rerank_model_path=self.settings.model.rerank_model_path,
                config=self.settings.retrieval
            )

            # 6. 初始化答案生成器
            self.answer_generator = AnswerGenerator(
                llm=self.llm,
                config=self.settings.retrieval
            )

            self.is_initialized = True
            logger.logger.info("RAG service initialization completed successfully")

        except Exception as e:
            logger.log_error(e, {"stage": "initialization"})
            raise

    def _validate_query(self, query: str) -> bool:
        """验证查询"""
        if not query or not query.strip():
            return False

        if len(query) > self.settings.app.max_query_length:
            return False

        return True

    def query(
            self,
            question: str,
            user_id: Optional[str] = None,
            use_cache: bool = True,
            include_debug: bool = False
    ) -> Dict[str, Any]:
        """处理查询"""
        start_time = time.time()

        if not self.is_initialized:
            return {
                "error": "Service not initialized",
                "answer": "服务未初始化，请稍后重试。"
            }

        # 验证查询
        if not self._validate_query(question):
            return {
                "error": "Invalid query",
                "answer": "查询无效，请检查输入。"
            }

        logger.log_query(question, user_id)

        try:
            # 检索
            retrieval_result = self.retriever.hybrid_retrieve(
                query=question,
                llm=self.llm,
                use_cache=use_cache,
                debug=include_debug
            )

            # 生成答案
            answer_result = self.answer_generator.generate_answer(
                query=question,
                retrieval_results=retrieval_result.nodes
            )

            total_time = time.time() - start_time

            # 记录指标
            metrics = QueryMetrics(
                timestamp=datetime.now(),
                query=question,
                retrieval_time=retrieval_result.retrieval_time,
                generation_time=answer_result["generation_time"],
                total_time=total_time,
                num_results=len(retrieval_result.nodes),
                confidence=answer_result["confidence"],
                cache_hit=retrieval_result.cache_hit,
                method_used=retrieval_result.method_used,
                user_id=user_id
            )
            metrics_collector.record_query(metrics)

            # 构建响应
            response = {
                "answer": answer_result["answer"],
                "confidence": answer_result["confidence"],
                "sources": answer_result["sources"],
                "total_time": total_time,
                "retrieval_time": retrieval_result.retrieval_time,
                "generation_time": answer_result["generation_time"],
                "cache_hit": retrieval_result.cache_hit,
                "method_used": retrieval_result.method_used,
                "num_sources": len(answer_result["sources"])
            }

            if include_debug:
                response["debug"] = {
                    "total_candidates": retrieval_result.total_candidates,
                    "context_used": answer_result["context_used"],
                    "retrieval_scores": [r.score for r in retrieval_result.nodes]
                }

            return response

        except Exception as e:
            logger.log_error(e, {"query": question, "user_id": user_id})
            metrics_collector.record_error(type(e).__name__)

            return {
                "error": str(e),
                "answer": "处理您的问题时发生错误，请稍后重试。",
                "total_time": time.time() - start_time
            }

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            metrics = metrics_collector.get_system_metrics()

            return {
                "status": "healthy" if self.is_initialized else "not_initialized",
                "metrics": {
                    "total_queries": metrics.total_queries,
                    "avg_response_time": metrics.avg_total_time,
                    "cache_hit_rate": metrics.cache_hit_rate,
                    "avg_confidence": metrics.avg_confidence,
                    "queries_per_minute": metrics.queries_per_minute,
                    "error_rate": metrics.error_rate
                },
                "components": {
                    "embed_model": self.embed_model is not None,
                    "llm": self.llm is not None,
                    "retriever": self.retriever is not None,
                    "vector_store": self.vector_store is not None
                }
            }
        except Exception as e:
            logger.log_error(e, {"operation": "get_system_status"})
            return {"status": "error", "error": str(e)}

    def clear_cache(self):
        """清空缓存"""
        try:
            cache_manager.redis_client.flushdb()
            logger.logger.info("Cache cleared successfully")
            return {"success": True}
        except Exception as e:
            logger.log_error(e, {"operation": "clear_cache"})
# services/rag_service.py (续)
            return {"success": False, "error": str(e)}

    def export_metrics_to_file(self, filepath: str = "rag_metrics.json"):
        """导出当前指标到文件"""
        metrics_collector.export_metrics(filepath)
        logger.logger.info(f"Metrics exported to {filepath}")

