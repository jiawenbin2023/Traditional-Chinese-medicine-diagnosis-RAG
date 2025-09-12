# main.py
import os
import sys
import torch
from services.rag_service import EnterpriseRAGService
from utils.logger import logger
from config.settings import Settings


# 设置LlamaIndex全局日志级别 (可选，与自定义logger配合使用)
# llama_index.core.set_global_handler("simple")

def main():
    # 确保日志目录存在
    os.makedirs("logs", exist_ok=True)

    # 1. 加载配置（可以从环境变量或配置文件加载）
    os.environ["EMBED_MODEL_PATH"] = "/home/ubuntu/PycharmProjects/pythonRAG/models/BAAI/bge-large-zh"
    os.environ["LLM_MODEL_PATH"] = "/home/ubuntu/PycharmProjects/pythonRAG/models/qwen/Qwen2-7B-Instruct"
    os.environ["RERANK_MODEL_PATH"] = "/home/ubuntu/PycharmProjects/pythonRAG/models/BAAI/bge-reranker-base"

    # 初始化RAG服务
    rag_service = EnterpriseRAGService()

    # 2. 初始化所有组件
    try:
        rag_service.initialize()
    except Exception as e:
        logger.log_error(e, {"stage": "main_initialization"})
        print("❌ RAG服务初始化失败，请检查配置和模型路径。", file=sys.stderr)
        sys.exit(1)

    print("\n==")
    print("🤖 企业级智能多路 RAG 已启动 —— 输入 'exit' 或 'quit' 退出")
    print("==")

    # 3. 主循环
    while True:
        try:
            q = input("\n❓ 请输入你的问题: ").strip()
            if not q:
                continue
            if q.lower() in ["exit", "quit", "退出"]:
                break

            response = rag_service.query(q, user_id="user_console_001", include_debug=True)

            if "error" in response:
                print(f"\n❌ 错误: {response['error']}")
                print(f"💡 答案: {response.get('answer', '无法提供答案。')}")
            else:
                print(f"\n💡 答案:\n{response['answer']}")
                print("\n📚 参考资料来源:")
                if response['sources']:
                    for i, src in enumerate(response['sources'], 1):
                        print(f" {i}. 文件: {src['file_name']} | 分数: {src['score']:.4f} | 预览: {src['preview']}")
                else:
                    print(" 无")

                print(
                    f"\n⏱ 总耗时: {response['total_time']:.2f} 秒 (检索: {response['retrieval_time']:.2f}s, 生成: {response['generation_time']:.2f}s)"
                )
                print(f"📊 置信度: {response['confidence']:.2%}")
                print(f"⚡ 缓存命中: {'是' if response['cache_hit'] else '否'} (召回方式: {response['method_used']})")

                if 'debug' in response:
                    print("\n--- 调试信息 ---")
                    print(f"总候选文档数: {response['debug']['total_candidates']}")
                    # print(f"使用的上下文:\n{response['debug']['context_used'][:500]}...")
                    # print(f"召回分数: {response['debug']['retrieval_scores']}")
                    print("----------------")

        except KeyboardInterrupt:
            print("\n检测到Ctrl+C，正在退出...")
            break
        except Exception as e:
            logger.log_error(e, {"stage": "main_loop_query"})
            print(f"发生未预期的错误: {e}", file=sys.stderr)

    # 4. 打印最终系统指标
    status = rag_service.get_system_status()
    print("\n--- 系统概览 ---")
    print(f"状态: {status['status']}")
    print(f"总查询数: {status['metrics']['total_queries']}")
    print(f"平均响应时间: {status['metrics']['avg_response_time']:.2f}s")
    print(f"缓存命中率: {status['metrics']['cache_hit_rate']:.2%}")
    print(f"平均置信度: {status['metrics']['avg_confidence']:.2%}")
    print(f"每分钟查询数: {status['metrics']['queries_per_minute']:.2f}")
    print(f"错误率: {status['metrics']['error_rate']:.2%}")
    print("----------------")

    # 5. 导出指标（可选）
    rag_service.export_metrics_to_file("rag_system_metrics_final.json")
    print("RAG服务已关闭。")


if __name__ == "__main__":
    main()
