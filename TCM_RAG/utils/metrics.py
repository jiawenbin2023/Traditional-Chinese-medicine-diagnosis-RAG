# utils/metrics.py
import time
import threading
from collections import defaultdict, deque
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json


@dataclass
class QueryMetrics:
    timestamp: datetime
    query: str
    retrieval_time: float
    generation_time: float
    total_time: float
    num_results: int
    confidence: float
    cache_hit: bool
    method_used: str
    user_id: str = None


@dataclass
class SystemMetrics:
    total_queries: int
    avg_retrieval_time: float
    avg_generation_time: float
    avg_total_time: float
    cache_hit_rate: float
    avg_confidence: float
    queries_per_minute: float
    error_rate: float


class MetricsCollector:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.query_history = deque(maxlen=window_size)
        self.error_count = defaultdict(int)
        self.lock = threading.Lock()

    def record_query(self, metrics: QueryMetrics):
        """记录查询指标"""
        with self.lock:
            self.query_history.append(metrics)

    def record_error(self, error_type: str):
        """记录错误"""
        with self.lock:
            self.error_count[error_type] += 1

    def get_system_metrics(self, time_window_minutes: int = 60) -> SystemMetrics:
        """获取系统指标"""
        with self.lock:
            if not self.query_history:
                return SystemMetrics(0, 0, 0, 0, 0, 0, 0, 0)

            # 过滤时间窗口内的数据
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            recent_queries = [q for q in self.query_history if q.timestamp >= cutoff_time]

            if not recent_queries:
                return SystemMetrics(0, 0, 0, 0, 0, 0, 0, 0)

            total_queries = len(recent_queries)
            avg_retrieval_time = sum(q.retrieval_time for q in recent_queries) / total_queries
            avg_generation_time = sum(q.generation_time for q in recent_queries) / total_queries
            avg_total_time = sum(q.total_time for q in recent_queries) / total_queries
            cache_hit_rate = sum(1 for q in recent_queries if q.cache_hit) / total_queries
            avg_confidence = sum(q.confidence for q in recent_queries) / total_queries
            queries_per_minute = total_queries / time_window_minutes

            # 计算错误率
            total_errors = sum(self.error_count.values())
            error_rate = total_errors / (total_queries + total_errors) if (total_queries + total_errors) > 0 else 0

            return SystemMetrics(
                total_queries=total_queries,
                avg_retrieval_time=avg_retrieval_time,
                avg_generation_time=avg_generation_time,
                avg_total_time=avg_total_time,
                cache_hit_rate=cache_hit_rate,
                avg_confidence=avg_confidence,
                queries_per_minute=queries_per_minute,
                error_rate=error_rate
            )

    def export_metrics(self, filepath: str):
        """导出指标到文件"""
        with self.lock:
            metrics_data = {
                "system_metrics": asdict(self.get_system_metrics()),
                "recent_queries": [asdict(q) for q in list(self.query_history)[-100:]],
                "error_counts": dict(self.error_count),
                "export_time": datetime.now().isoformat()
            }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2, default=str)


metrics_collector = MetricsCollector()
