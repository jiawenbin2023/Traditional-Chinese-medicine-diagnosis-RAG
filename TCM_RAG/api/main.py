# api/main.py
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn
import os

from services.rag_service import EnterpriseRAGService
from utils.logger import logger
from utils.metrics import metrics_collector
from config.settings import Settings

# 初始化 RAG 服务 (在应用启动时初始化一次)
rag_service: Optional[EnterpriseRAGService] = None
app_settings = Settings()  # 获取配置

# FastAPI 应用
app = FastAPI(
    title="Enterprise RAG API",
    description="智能中医知识问答系统",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "api_user"
    use_cache: bool = True
    include_debug: bool = False


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    total_time: float
    retrieval_time: float
    generation_time: float
    cache_hit: bool
    method_used: str
    num_sources: int
    debug: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """在FastAPI应用启动时初始化RAG服务"""
    global rag_service
    logger.logger.info("FastAPI application startup event triggered.")
    try:
        rag_service = EnterpriseRAGService()
        rag_service.initialize()
        logger.logger.info("RAG service initialized successfully for FastAPI.")
    except Exception as e:
        logger.log_error(e, {"stage": "fastapi_startup"})
        # 如果初始化失败，应用应该优雅地启动，但查询会报错
        logger.logger.error("Failed to initialize RAG service. Queries will not work.")
        # 或者直接抛出异常阻止应用启动，根据需求选择
        # raise RuntimeError("RAG service failed to initialize.") from e


@app.on_event("shutdown")
async def shutdown_event():
    """在FastAPI应用关闭时执行清理操作"""
    logger.logger.info("FastAPI application shutdown event triggered.")
    if rag_service:
        # rag_service.export_metrics_to_file("rag_api_metrics_final.json") # 移除metrics输出
        logger.logger.info("RAG service metrics exported.")
    logger.logger.info("FastAPI application shutdown completed.")


# @app.get("/health", summary="健康检查", response_model=Dict[str, Any])
# async def health_check():
#     """检查API服务是否正常运行及RAG服务状态。"""
#     if rag_service and rag_service.is_initialized:
#         status_info = rag_service.get_system_status()
#         return JSONResponse(content=status_info, status_code=status.HTTP_200_OK)
#     else:
#         return JSONResponse(
#             content={"status": "unhealthy", "message": "RAG service not initialized or failed to start."},
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE
#         )

@app.get("/health", summary="健康检查", response_model=Dict[str, Any])
async def health_check():
    """检查API服务是否正常运行及RAG服务状态。"""
    if rag_service and rag_service.is_initialized:
        status_info = rag_service.get_system_status() # <-- 恢复这行
        return JSONResponse(content=status_info, status_code=status.HTTP_200_OK)
    else:
        return JSONResponse(
            content={"status": "unhealthy", "message": "RAG service not initialized or failed to start."},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )




@app.post("/query", summary="RAG查询接口", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    接收用户查询，执行RAG流程，并返回答案及相关信息。
    """
    if not rag_service or not rag_service.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not initialized or failed to start."
        )

    try:
        response = rag_service.query(
            request.query,
            user_id=request.user_id,
            use_cache=request.use_cache,
            include_debug=request.include_debug
        )
        if response.get("error"):
            # 如果RAG服务内部返回错误，也作为HTTP 500处理
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response["error"]
            )

        return QueryResponse(**response)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.log_error(e, {"api_endpoint": "/query", "request_query": request.query})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during query processing: {e}"
        )


@app.get("/metrics", summary="获取系统指标", response_model=Dict[str, Any])
async def get_metrics():
    """获取当前RAG系统的运行指标。"""
    if not rag_service or not rag_service.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not initialized or failed to start."
        )
    return rag_service.get_system_status()


@app.post("/clear_cache", summary="清空Redis缓存", response_model=Dict[str, Any])
async def clear_redis_cache():
    """清空RAG系统使用的Redis缓存。"""
    if not rag_service or not rag_service.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not initialized or failed to start."
        )
    result = rag_service.clear_cache()
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "Failed to clear cache.")
        )
    return {"message": "Redis cache cleared successfully.", "success": True}


# 运行 FastAPI 的主函数（用于调试或直接启动）
if __name__ == "__main__":
    # 设置环境变量，模拟生产环境配置
    os.environ["EMBED_MODEL_PATH"] = "/home/ubuntu/PycharmProjects/pythonRAG/models/BAAI/bge-large-zh"
    os.environ["LLM_MODEL_PATH"] = "/home/ubuntu/PycharmProjects/pythonRAG/models/qwen/Qwen2-7B-Instruct"  # 或你的vLLM模型
    os.environ["RERANK_MODEL_PATH"] = "/home/ubuntu/PycharmProjects/pythonRAG/models/BAAI/bge-reranker-base"

    # 如果要使用vLLM，确保设置 use_vllm 为 True
    # app_settings.model.use_vllm = True # 可以在这里强制设置，或者在config/settings.py中默认设置

    uvicorn.run(app, host=app_settings.app.api_host, port=app_settings.app.api_port)
