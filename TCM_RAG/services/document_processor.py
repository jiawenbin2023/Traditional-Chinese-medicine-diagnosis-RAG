# services/document_processor.py
import os
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from utils.logger import logger
import hashlib
from datetime import datetime


class DocumentProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_tokenizer_fn=self._sentence_tokenizer
        )
        self.supported_formats = {'.txt', '.json', '.md', '.pdf'}

    def _sentence_tokenizer(self, text: str) -> List[str]:
        """改进的句子分词器"""
        sentences = re.split(r'(?<=[。！？?])', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    def _clean_text(self, text: str) -> str:
        """文本清洗"""
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 移除重复行
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        seen = set()
        cleaned = []
        for line in lines:
            if line not in seen and len(line) > 10:  # 过滤太短的行
                cleaned.append(line)
                seen.add(line)
        return "\n".join(cleaned)

    def _extract_metadata(self, filepath: Path) -> Dict[str, Any]:
        """提取文件元数据"""
        stat = filepath.stat()
        return {
            "file_name": filepath.name,
            "file_path": str(filepath),
            "file_size": stat.st_size,
            "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "file_hash": self._calculate_file_hash(filepath)
        }

    def _calculate_file_hash(self, filepath: Path) -> str:
        """计算文件哈希"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def process_txt_file(self, filepath: Path) -> List[Document]:
        """处理TXT文件"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            cleaned_content = self._clean_text(content)
            metadata = self._extract_metadata(filepath)

            return [Document(text=cleaned_content, metadata=metadata)]

        except Exception as e:
            logger.log_error(e, {"file_path": str(filepath)})
            return []


    # services/document_processor.py (续)
    def process_json_file(self, filepath: Path) -> List[Document]:
        """处理JSON文件"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            documents = []
            base_metadata = self._extract_metadata(filepath)

            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        text = self._dict_to_text(item)
                        metadata = {**base_metadata, "item_index": i}
                        documents.append(Document(text=text, metadata=metadata))
            elif isinstance(data, dict):
                text = self._dict_to_text(data)
                documents.append(Document(text=text, metadata=base_metadata))

            return documents

        except Exception as e:
            logger.log_error(e, {"file_path": str(filepath)})
            return []

    def _dict_to_text(self, data: Dict[str, Any]) -> str:
        """将字典转换为文本"""
        text_parts = []
        for key, value in data.items():
            if isinstance(value, (str, int, float)):
                text_parts.append(f"{key}：{value}")
            elif isinstance(value, list):
                text_parts.append(f"{key}：{', '.join(map(str, value))}")
        return "\n".join(text_parts)

    def process_directory(self, data_dir: str) -> List[Document]:
        """处理整个目录"""
        documents = []
        data_path = Path(data_dir)

        if not data_path.exists():
            logger.logger.error(f"Data directory not found: {data_dir}")
            return documents

        for filepath in data_path.rglob("*"):
            if filepath.is_file() and filepath.suffix in self.supported_formats:
                logger.logger.info(f"Processing file: {filepath}")

                if filepath.suffix == '.txt':
                    docs = self.process_txt_file(filepath)
                elif filepath.suffix == '.json':
                    docs = self.process_json_file(filepath)
                # TODO: 添加PDF, MD等格式支持

                documents.extend(docs)

        logger.logger.info(f"Processed {len(documents)} documents from {data_dir}")
        return documents

    def create_nodes(self, documents: List[Document]):
        """创建节点"""
        try:
            nodes = self.splitter.get_nodes_from_documents(documents)
            logger.logger.info(f"Created {len(nodes)} nodes from {len(documents)} documents")
            return nodes
        except Exception as e:
            logger.log_error(e, {"num_documents": len(documents)})
            raise

