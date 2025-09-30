# Enterprise-level Chinese knowledge Q&A system (企业级中文知识问答系统)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-orange.svg)
![Milvus](https://img.shields.io/badge/Milvus-v2.4.6-lightgrey.svg)
![Redis](https://img.shields.io/badge/Redis-latest-red)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## 📖 项目简介

本项目旨在构建一个**高效、精准的企业级知识问答系统**，利用先进的**检索增强生成 (Retrieval-Augmented Generation, RAG)** 技术，结合大规模语言模型 (LLM) 和专业知识库，为用户提供智能、权威的问答服务。系统解决了传统问答检索效率低下、信息孤岛严重的问题，大幅提升了知识获取效率和辅助决策能力。

### ✨ 核心功能与特点

*   **智能问答：** 针对用户提出的产品等问题，提供准确、连贯且有据可查的答案。
*   **混合检索策略：** 结合语义检索 (向量相似度) 和关键词检索 (BM25)，确保从海量知识库中召回最相关、最全面的上下文信息。
*   **高性能 Reranking：** 通过独立的重排模型对初次检索结果进行精细排序，进一步提升上下文与查询的相关性。
*   **实时交互界面：** 提供基于 Streamlit 的用户友好型 Web 界面，支持实时问答与结果展示。
*   **模块化与可扩展：** 清晰的代码结构和分层设计，便于功能扩展、模型替换和维护。
*   **高效缓存机制：** 集成 Redis 缓存，显著提升重复查询的响应速度。
*   **API 服务化：** 基于 FastAPI 提供稳定、高性能的 RESTful API 接口，方便其他应用集成。
*   **GPU 加速：** 利用 PyTorch 和 NVIDIA GPU 硬件加速 LLM 推理和 Embedding 计算。

## 🚀 效果概览

本项目采用前后端分离的微服务架构，核心 RAG 逻辑封装在 FastAPI 后端中，并通过 Streamlit 提供交互界面。
![image-20250912155108981](/home/ubuntu/.config/Typora/typora-user-images/image-20250912155108981.png)

## 💻 技术栈


    检索增强生成 (RAG) 架构： 采用混合检索策略（Dense + Sparse），结合语义检索和关键词检索优势，确保从海量知识库中召回最相关的上下文信息。
    
    LLM 集成 (HuggingFaceLLM)： 深度集成 HuggingFaceLLM，加载并利用 Qwen2-7B-Instruct 等大型语言模型进行答案生成，确保回答的准确性、连贯性和专业性。
    
    向量数据库 (Milvus)： 利用高性能向量数据库 Milvus 存储和管理知识文档的嵌入向量，实现高效的语义相似度检索。
    
    高性能嵌入与重排 (Embedding & Reranking)： 采用 BAAI/bge-large-zh 作为嵌入模型，提升中文文本的语义表示能力；引入 BAAI/bge-reranker-base 进行文档重排，优化检索结果的相关性，提高RAG召回质量。
    
    缓存机制 (Redis)： 集成 Redis 作为高性能缓存层，存储常见查询结果，显著提升系统响应速度和资源利用率。
    
    模块化与可扩展架构： 采用清晰的层次化设计（config, models, services, api, utils），确保系统的高内聚、低耦合，易于维护和功能扩展。
    
    Web 服务 (FastAPI)： 利用 FastAPI 框架构建高性能、异步的 RESTful API 服务，提供稳定、可扩展的后端支撑。
    
    交互式前端 (Streamlit)： 使用 Streamlit 快速开发直观、用户友好的交互式问答界面，方便用户进行实时提问和结果展示。

## ⚙️ 快速开始 (使用 Docker Compose)

步骤1：克隆我的仓库，安装requirement.txt

下载代码至本地；

```bash
pip install -r requirements.txt
```

步骤 2: 下载并存放大模型文件

由于模型文件较大，没有上传到仓库中，所以需要我们将其下载到本地目录的models中。


```python
#下载以下模型到对应的目录：
    BAAI/bge-large-zh (嵌入模型): 下载到 ./models/BAAI/bge-large-zh/
       # 可通过 Hugging Face Hub 下载：huggingface-cli download BAAI/bge-large-zh --local-dir /home/ubuntu/models/BAAI/bge-large-zh
    BAAI/bge-reranker-base (重排模型): 下载到 ./models/BAAI/bge-reranker-base/
     #   可通过 Hugging Face Hub 下载：huggingface-cli download BAAI/bge-reranker-base --local-dir /home/ubuntu/models/BAAI/bge-reranker-base
    Qwen2-7B-Instruct (LLM): 下载到 ./models/qwen/Qwen2-7B-Instruct/
      #  可通过 Hugging Face Hub 下载：huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir /home/ubuntu/models/qwen/Qwen2-7B-Instruct
```


步骤 3: 准备知识库数据

确保你的知识库文件（例如 TCM.json）放置在项目根目录下的 data/ 目录中。

步骤4：启动milvus-standalone和redis服务

```bash
cd 你的milvus安装目录 # 确保你在这个目录
bash standalone_embed.sh start
```

```bash
sudo systemctl start redis-server
```

步骤5: 启动streamlit服务

```bash
streamlit run ui/streamlit_app.py
```



## 📦 项目结构

.
├── api/  # FastAPI 后端服务
│   └── main.py              # FastAPI 应用主入口
├── config/                  # 系统配置文件
│   └── settings.py          # 全局配置 (模型路径, Milvus, Redis, 应用设置等)
├── data/  # 知识库数据目录
│   └── TCM.json             # 您的知识库文件
├── models/                  # 模型封装与加载逻辑
│   ├── QWEN                 # qwen模型
│   └── BAAI                 # embedding模型和rerank模型
├── services/                # RAG 核心业务逻辑
│   ├── rag_service.py       # RAG 服务核心 (协调各组件)
│   ├── document_processor.py# 文档加载、分块与节点创建
│   ├── retriever.py         # 混合检索与重排逻辑 (Milvus, BM25, Reranker)
│   └── answer_generator.py  # LLM 答案生成
├── utils/                   # 工具类与辅助函数
│   ├── cache.py             # Redis 缓存管理
│   ├── logger.py            # 自定义日志模块
│   └── metrics.py           # 系统运行指标收集
├── Dockerfile               # 构建 FastAPI 和 Streamlit 容器的 Dockerfile
├── docker-compose.yml       # Docker Compose 配置文件 (一键部署所有服务)
├── requirements.txt         # Python 依赖列表
├── run_api.py               # 启动 FastAPI 服务的脚本 (不用于Docker Compose部署)
├── streamlit_app.py         # Streamlit 前端应用
└── logs/  # 日志文件输出目录



## 📜 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 📧 联系方式

如果你有任何问题或建议，欢迎通过 GitHub Issue 提交，或者发送邮件至 1039995947@qq.com。
