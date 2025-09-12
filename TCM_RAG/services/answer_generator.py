# services/answer_generator.py
import time
from typing import List, Dict, Any, Tuple
from llama_index.core.schema import NodeWithScore
from llama_index.core.prompts import PromptTemplate
from utils.logger import logger
from config.settings import RetrievalConfig


class AnswerGenerator:
    def __init__(self, llm, config: RetrievalConfig):
        self.llm = llm
        self.config = config

        self.prompt_template = PromptTemplate(
            """你是一个专业的知识助手，请基于以下检索到的资料回答用户问题。

=== 检索到的资料 ===
{context_str}

=== 用户问题 ===
{query_str}

=== 回答要求 ===
1. 答案必须基于提供的资料，不得编造信息
2. 如果资料不足以回答问题，请明确说明
3. 答案要准确、完整、条理清晰
4. 使用专业但易懂的语言
5. 如果涉及多个方面，请分点说明

请提供你的答案："""
        )

    def _compress_context_with_sources(
            self,
            query: str,
            results: List[NodeWithScore]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """压缩上下文并提取来源信息"""
        compressed_context = ""
        sources = []
        current_length = 0

        for i, result in enumerate(results):
            text = result.node.text
            metadata = result.node.metadata

            # 检查是否超过长度限制
            if current_length + len(text) > self.config.max_context_chars:
                # 尝试截断
                remaining_chars = self.config.max_context_chars - current_length
                if remaining_chars > 100:  # 至少保留100字符
                    text = text[:remaining_chars] + "..."
                else:
                    break

            compressed_context += f"\n[资料{i + 1}] {text}\n"
            current_length += len(text)

            # 提取来源信息
            source_info = {
                "index": i + 1,
                "score": result.score,
                "file_name": metadata.get("file_name", "未知来源"),
                "preview": text.replace("\n", " ")[:80] + "..." if len(text) > 80 else text,
                "metadata": metadata
            }
            sources.append(source_info)

        return compressed_context.strip(), sources

    def generate_answer(
            self,
            query: str,
            retrieval_results: List[NodeWithScore],
            include_sources: bool = True
    ) -> Dict[str, Any]:
        start_time = time.time()

        try:
            # 1. 压缩上下文
            context, sources = self._compress_context_with_sources(query, retrieval_results)

            if not context:
                return {
                    "answer": "抱歉，我无法从检索到的资料中找到相关信息来回答您的问题。",
                    "sources": [],
                    "generation_time": time.time() - start_time,
                    "context_used": "",
                    "confidence": 0.0
                }

            # 2. 生成 Prompt
            prompt = self.prompt_template.format(
                context_str=context,
                query_str=query
            )

            # 3. 调用 LLM 生成答案
            response = self.llm.complete(prompt)
            answer = response.text.strip()

            # 4. 计算置信度
            avg_score = sum(r.score for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0
            confidence = min(avg_score, 1.0)

            return {
                "answer": answer,
                "sources": sources if include_sources else [],
                "generation_time": time.time() - start_time,
                "context_used": context,
                "confidence": confidence
            }

        except Exception as e:
            # 捕获异常返回默认信息
            from utils.logger import logger
            logger.log_error(e, {"query": query})

            return {
                "answer": "抱歉，生成答案时发生错误，请稍后再试。",
                "sources": [],
                "generation_time": time.time() - start_time,
                "context_used": "",
                "confidence": 0.0,
                "error": str(e)
            }
