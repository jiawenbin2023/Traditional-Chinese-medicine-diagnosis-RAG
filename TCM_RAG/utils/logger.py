# utils/logger.py
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any
from pythonjsonlogger import jsonlogger
import traceback

class EnterpriseLogger:
    def __init__(self, name: str, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        # 文件处理器 (JSON格式)
        file_handler = logging.FileHandler(
            self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        json_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
        file_handler.setFormatter(json_formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def log_query(self, query: str, user_id: str = None, metadata: Dict[str, Any] = None):
        self.logger.info("query_received", extra={
            "query": query,
            "user_id": user_id,
            "metadata": metadata or {}
        })

    def log_retrieval(self, query: str, num_results: int, retrieval_time: float):
        self.logger.info("retrieval_completed", extra={
            "query": query,
            "num_results": num_results,
            "retrieval_time": retrieval_time
        })

    # def log_error(self, error: Exception, context: Dict[str, Any] = None):
    #     self.logger.error("error_occurred", extra={
    #         "error_type": type(error).__name__,
    #         "error_message": str(error),
    #         "context": context or {}
    #     })

    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()  # 获取完整堆栈

        self.logger.error(
            f"Error occurred: {error_type} - {error_message}\n{stack_trace}",
            extra={
                "error_type": error_type,
                "error_message": error_message,
                "stack_trace": stack_trace,
                "context": context or {}
            }
        )


logger = EnterpriseLogger("enterprise_rag")
