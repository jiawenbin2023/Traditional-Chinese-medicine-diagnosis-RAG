# run_api.py
import uvicorn
import os
from config.settings import Settings

# 从config获取FastAPI的地址和端口
app_settings = Settings()

if __name__ == "__main__":
    # 设置环境变量，模拟生产环境配置
    os.environ["EMBED_MODEL_PATH"] = "/home/ubuntu/PycharmProjects/pythonRAG/models/BAAI/bge-large-zh"
    os.environ["LLM_MODEL_PATH"] = "/home/ubuntu/PycharmProjects/pythonRAG/models/qwen/Qwen2-7B-Instruct"  # 或你的vLLM模型
    os.environ["RERANK_MODEL_PATH"] = "/home/ubuntu/PycharmProjects/pythonRAG/models/BAAI/bge-reranker-base"

    # 确保 data 目录存在
    os.makedirs(app_settings.app.data_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)  # 确保日志目录存在

    print(f"Starting FastAPI server on {app_settings.app.api_host}:{app_settings.app.api_port}")
    print("Ensure Redis and Milvus are running.")
    print(f"LLM Model Path: {os.getenv('LLM_MODEL_PATH')}")
    print(f"Using vLLM: {app_settings.model.use_vllm}")  # ✅ 确认不使用 vLLM

    # 使用 api/main.py 作为 FastAPI 应用入口
    uvicorn.run(
        "api.main:app",
        host=app_settings.app.api_host,
        port=app_settings.app.api_port,
        reload=False,  # 开发模式下可以开启热重载
        workers=1,  # ✅ 确保只用一个worker
        log_level=app_settings.app.log_level.lower()
    )
