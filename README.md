# Traditional-Chinese-medicine-diagnosis-RAG
Traditional Chinese medicine diagnosis-RAG
项目简介：企业级中医知识问答系统 (RAG)

## 项目名称： 企业级中医知识问答系统 (Enterprise Traditional Chinese Medicine RAG System)

## 项目背景与目标：

本项目旨在构建一个高效、精准的企业级中医知识问答系统，解决传统中医文献检索效率低下、信息孤岛严重的问题。通过集成先进的检索增强生成（RAG）技术，利用大规模语言模型（LLM）结合专业知识库，实现用户对复杂中医病症、诊断、治疗等问题的智能问答，提升知识获取效率和决策支持能力。

## 核心技术栈与亮点：


    检索增强生成 (RAG) 架构： 采用混合检索策略（Dense + Sparse），结合语义检索和关键词检索优势，确保从海量中医知识库中召回最相关的上下文信息。

    LLM 集成 (HuggingFaceLLM)： 深度集成 HuggingFaceLLM，加载并利用 Qwen2-7B-Instruct 等大型语言模型进行答案生成，确保回答的准确性、连贯性和专业性。

    向量数据库 (Milvus)： 利用高性能向量数据库 Milvus 存储和管理中医知识文档的嵌入向量，实现高效的语义相似度检索。

    高性能嵌入与重排 (Embedding & Reranking)： 采用 BAAI/bge-large-zh 作为嵌入模型，提升中文文本的语义表示能力；引入 BAAI/bge-reranker-base 进行文档重排，优化检索结果的相关性，提高RAG召回质量。

    缓存机制 (Redis)： 集成 Redis 作为高性能缓存层，存储常见查询结果，显著提升系统响应速度和资源利用率。

    模块化与可扩展架构： 采用清晰的层次化设计（config, models, services, api, utils），确保系统的高内聚、低耦合，易于维护和功能扩展。

    Web 服务 (FastAPI)： 利用 FastAPI 框架构建高性能、异步的 RESTful API 服务，提供稳定、可扩展的后端支撑。

    交互式前端 (Streamlit)： 使用 Streamlit 快速开发直观、用户友好的交互式问答界面，方便用户进行实时提问和结果展示。

## 运行
你需要自己下载Qwen2-7B-Instruct/BAAI/bge-large-zh/BAAI/bge-reranker-base 并将它们放入models中，然后安装requirement.txt,最后在终端或者编译器中运行streamlit。

# Enterprise Traditional Chinese Medicine RAG System (企业级中医知识问答系统)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-orange.svg)
![Milvus](https://img.shields.io/badge/Milvus-v2.4.6-lightgrey.svg)
![Redis](https://img.shields.io/badge/Redis-latest-red)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## 📖 项目简介

本项目旨在构建一个**高效、精准的企业级中医知识问答系统**，利用先进的**检索增强生成 (Retrieval-Augmented Generation, RAG)** 技术，结合大规模语言模型 (LLM) 和专业中医知识库，为用户提供智能、权威的问答服务。系统解决了传统中医文献检索效率低下、信息孤岛严重的问题，大幅提升了知识获取效率和辅助决策能力。

### ✨ 核心功能与特点

*   **智能问答：** 针对用户提出的中医病症、诊断、治疗、药方等问题，提供准确、连贯且有据可查的答案。
*   **混合检索策略：** 结合语义检索 (向量相似度) 和关键词检索 (BM25)，确保从海量知识库中召回最相关、最全面的上下文信息。
*   **高性能 Reranking：** 通过独立的重排模型对初次检索结果进行精细排序，进一步提升上下文与查询的相关性。
*   **实时交互界面：** 提供基于 Streamlit 的用户友好型 Web 界面，支持实时问答与结果展示。
*   **模块化与可扩展：** 清晰的代码结构和分层设计，便于功能扩展、模型替换和维护。
*   **高效缓存机制：** 集成 Redis 缓存，显著提升重复查询的响应速度。
*   **API 服务化：** 基于 FastAPI 提供稳定、高性能的 RESTful API 接口，方便其他应用集成。
*   **GPU 加速：** 利用 PyTorch 和 NVIDIA GPU 硬件加速 LLM 推理和 Embedding 计算。

## 🚀 架构概览

本项目采用前后端分离的微服务架构，核心 RAG 逻辑封装在 FastAPI 后端中，并通过 Streamlit 提供交互界面。

```mermaid
graph TD
    User[用户浏览器] --> |HTTP| Streamlit[Streamlit Frontend];
    Streamlit --> |HTTP Request| FastAPI[FastAPI Backend];
    FastAPI --> |1. 初始化服务<br/>2. 处理查询| RAGService[RAG Service];
    RAGService --> |Embedding Model<br/>(BAAI/bge-large-zh)| Embedding[Embedding Service];
    RAGService --> |LLM (Qwen2-7B-Instruct)<br/>(Answer Generation)| LLMService[LLM Service];
    RAGService --> |Rerank Model<br/>(BAAI/bge-reranker-base)| RerankService[Rerank Service];
    RAGService --> |Vector Search| Milvus[Milvus Vector DB];
    RAGService --> |Cache Lookup/Store| Redis[Redis Cache];

    subgraph Knowledge Base
        Data[TCM.json 文件] --> DocumentProcessor[文档处理与分块];
        DocumentProcessor --> Embedding;
        Embedding --> Milvus;
    end

Enterprise Traditional Chinese Medicine RAG System (企业级中医知识问答系统)

Python
预览

FastAPI
预览

Streamlit
预览

PyTorch
预览

Milvus
预览

Redis
预览

License

预览

📖 项目简介

本项目旨在构建一个高效、精准的企业级中医知识问答系统，利用先进的检索增强生成 (Retrieval-Augmented Generation, RAG) 技术，结合大规模语言模型 (LLM) 和专业中医知识库，为用户提供智能、权威的问答服务。系统解决了传统中医文献检索效率低下、信息孤岛严重的问题，大幅提升了知识获取效率和辅助决策能力。

✨ 核心功能与特点


    智能问答： 针对用户提出的中医病症、诊断、治疗、药方等问题，提供准确、连贯且有据可查的答案。

    混合检索策略： 结合语义检索 (向量相似度) 和关键词检索 (BM25)，确保从海量知识库中召回最相关、最全面的上下文信息。

    高性能 Reranking： 通过独立的重排模型对初次检索结果进行精细排序，进一步提升上下文与查询的相关性。

    实时交互界面： 提供基于 Streamlit 的用户友好型 Web 界面，支持实时问答与结果展示。

    模块化与可扩展： 清晰的代码结构和分层设计，便于功能扩展、模型替换和维护。

    高效缓存机制： 集成 Redis 缓存，显著提升重复查询的响应速度。

    API 服务化： 基于 FastAPI 提供稳定、高性能的 RESTful API 接口，方便其他应用集成。

    GPU 加速： 利用 PyTorch 和 NVIDIA GPU 硬件加速 LLM 推理和 Embedding 计算。


🚀 架构概览

本项目采用前后端分离的微服务架构，核心 RAG 逻辑封装在 FastAPI 后端中，并通过 Streamlit 提供交互界面。

预览
代码

复制

导出图片

Syntax error in textmermaid version 11.6.0


💻 技术栈


    核心语言: Python 3.10+

    RAG 框架: LlamaIndex

    大型语言模型 (LLM): Qwen2-7B-Instruct (通过 HuggingFaceLLM 加载)

    嵌入模型: BAAI/bge-large-zh

    重排模型: BAAI/bge-reranker-base

    向量数据库: Milvus (v2.4.6)

    缓存: Redis

    后端框架: FastAPI (0.110.0), Uvicorn (0.35.0)

    前端框架: Streamlit (1.36.0)

    容器化: Docker, Docker Compose

    GPU 加速: PyTorch (2.5.1+cu121), NVIDIA CUDA

    日志/指标: 自定义日志模块, Prometheus 风格指标收集


⚙️ 快速开始 (使用 Docker Compose)

这是最推荐的部署方式，能够快速搭建所有服务（Milvus, Redis, FastAPI, Streamlit）并解决环境依赖问题。

前提条件


    Git: 用于克隆代码库。

    Docker & Docker Compose: 用于构建和运行容器化服务。
        Docker 安装指南
        Docker Compose 安装指南 (对于新版 Docker Desktop，通常已内置)

    NVIDIA GPU & 驱动: 确保你的服务器上安装了最新的 NVIDIA GPU 驱动。

    NVIDIA Container Toolkit: 允许 Docker 容器访问 GPU。
        NVIDIA Container Toolkit 安装指南
        安装后请确保重启 Docker 服务 (sudo systemctl restart docker)。

    CUDA 工具包: 确保宿主机安装的 CUDA 版本与 Dockerfile 中 PyTorch 指定的 CUDA 版本兼容 (本项目默认 cu121)。


步骤 1: 克隆代码库

bash

复制

git clone https://github.com/你的用户名/你的repo名称.git

cd 你的repo名称 # 例如: cd Enterprise-TCM-RAG-System

步骤 2: 下载并存放大模型文件

由于模型文件较大，我们将其下载到宿主机，并通过 Docker Volume 映射到容器内部，避免每次构建都下载。


    在你的服务器上创建一个专门存放模型的目录，例如：

    bash

复制

mkdir -p /home/ubuntu/models/BAAI

mkdir -p /home/ubuntu/models/qwen


    下载以下模型到对应的目录：
        BAAI/bge-large-zh (嵌入模型): 下载到 /home/ubuntu/models/BAAI/bge-large-zh/
            可通过 Hugging Face Hub 下载：huggingface-cli download BAAI/bge-large-zh --local-dir /home/ubuntu/models/BAAI/bge-large-zh
        BAAI/bge-reranker-base (重排模型): 下载到 /home/ubuntu/models/BAAI/bge-reranker-base/
            可通过 Hugging Face Hub 下载：huggingface-cli download BAAI/bge-reranker-base --local-dir /home/ubuntu/models/BAAI/bge-reranker-base
        Qwen2-7B-Instruct (LLM): 下载到 /home/ubuntu/models/qwen/Qwen2-7B-Instruct/
            可通过 Hugging Face Hub 下载：huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir /home/ubuntu/models/qwen/Qwen2-7B-Instruct


步骤 3: 准备知识库数据

确保你的中医知识库文件（例如 TCM.json）放置在项目根目录下的 data/ 目录中。

bash

复制

# 示例：

mkdir -p data

# 将你的TCM.json文件放到此处

# cp /path/to/your/TCM.json data/TCM.json

步骤 4: 配置文件 (config/settings.py) 调整

config/settings.py 已经为 Docker Compose 进行了适配，它会优先从环境变量读取配置。docker-compose.yml 文件将负责设置这些环境变量。

请确保 config/settings.py 中的 ModelConfig 路径是容器内部的路径 (例如 /app/models/...)，因为 Docker Compose 会将宿主机目录映射到那里。

python

复制

# config/settings.py (确保这些是容器内的路径，与docker-compose.yml中的volumes映射对应)

# ...

@dataclass

class ModelConfig:

    embed_model_path: str = os.getenv("EMBED_MODEL_PATH", "/app/models/BAAI/bge-large-zh")

    llm_model_path: str = os.getenv("LLM_MODEL_PATH", "/app/models/qwen/Qwen2-7B-Instruct")

    rerank_model_path: str = os.getenv("RERANK_MODEL_PATH", "/app/models/BAAI/bge-reranker-base")

    # ...

# ...

@dataclass

class VectorStoreConfig:

    uri: str = os.getenv("MILVUS_URI", "http://milvus:19530") # 使用Docker服务名

    # ...

# ...

@dataclass

class AppConfig:

    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0") # 使用Docker服务名

    api_host: str = "0.0.0.0" # 在容器内部监听所有接口

    api_port: int = 8000

    # ...

步骤 5: 构建并启动服务

在项目根目录下（Enterprise-TCM-RAG-System/），执行以下命令：

bash

复制

docker compose up -d --build


    up -d: 在后台启动所有服务。

    --build: 强制重新构建 Docker 镜像（如果你修改了 Dockerfile 或项目代码，第一次运行也需要）。


这会启动以下服务：redis, milvus, fastapi-backend, streamlit-frontend。

步骤 6: 检查服务状态

bash

复制

docker compose ps

你应该看到所有服务都处于 Up 状态。

步骤 7: 防火墙配置 (如果服务器有防火墙)

如果你的服务器启用了防火墙，请确保开放 8000 (FastAPI) 和 8501 (Streamlit) 端口，以便外部访问。

bash

复制

sudo ufw allow 8000/tcp

sudo ufw allow 8501/tcp

# sudo ufw enable # 如果防火墙未启用，启用它

步骤 8: 访问应用

打开你的浏览器，访问：


    Streamlit 前端: http://你的服务器IP:8501

    FastAPI 健康检查: http://你的服务器IP:8000/health (用于确认后端 API 正常运行)


📦 项目结构

.
├── api/  # FastAPI 后端服务
│   └── main.py              # FastAPI 应用主入口
├── config/                  # 系统配置文件
│   └── settings.py          # 全局配置 (模型路径, Milvus, Redis, 应用设置等)
├── data/  # 知识库数据目录
│   └── TCM.json             # 您的中医知识库文件
├── models/                  # 模型封装与加载逻辑
│   └── embeddings.py        # 嵌入模型封装 (BAAI/bge-large-zh)
# └── vllm_llm.py            # vLLM 相关代码 (已移除，当前使用HuggingFaceLLM)
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


🛠️ 部署后管理

在项目根目录下：


    停止所有服务: docker compose down

    启动所有服务: docker compose up -d

    重启所有服务: docker compose restart

    只重启某个服务: docker compose restart fastapi-backend

    查看所有容器日志: docker compose logs

    查看特定容器日志: docker compose logs -f fastapi-backend

    进入容器内部调试: docker exec -it rag_fastapi_backend bash


💡 故障排除


    API状态: ❌ API连接超时 或 ❌ 错误 (502)：
        检查 FastAPI 日志： docker compose logs -f fastapi-backend。可能是 FastAPI 内部 RAG 服务初始化失败（LLM 加载、Milvus/Redis 连接问题）。
        检查模型路径： 确保宿主机 /home/ubuntu/models 目录下的模型文件存在且完整。
        检查 Docker Compose 配置： 确保 docker-compose.yml 中的 volumes 映射和 environment 变量（尤其是模型路径、Milvus/Redis URI）正确无误。
        GPU 问题： 确保 NVIDIA 驱动、CUDA 和 NVIDIA Container Toolkit 安装正确，且 fastapi-backend 服务配置了 GPU 资源 (deploy.resources.reservations.devices)。

    JSONDecodeError：
        尽管 curl 可能正常，Streamlit 在某些情况下仍可能收到非 JSON 响应。再次检查 FastAPI 日志，确保在任何异常情况下都返回 JSON 格式的错误响应。

    Milvus / Redis 连接问题： 检查 docker compose logs -f milvus 和 docker compose logs -f redis，确保它们正常启动。config/settings.py 中 Milvus 和 Redis 的 URI 应该使用 Docker Compose 的服务名称 (http://milvus:19530 和 redis://redis:6379/0)。


🤝 贡献

欢迎任何形式的贡献！如果你有改进建议、发现了 Bug 或想添加新功能，请通过以下方式参与：


    Fork 本仓库。

    创建新的功能分支 (git checkout -b feature/AmazingFeature)。

    提交你的改动 (git commit -m 'Add some AmazingFeature')。

    推送到分支 (git push origin feature/AmazingFeature)。

    打开 Pull Request。


📜 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

📧 联系方式

如果你有任何问题或建议，欢迎通过 GitHub Issue 提交，或者发送邮件至 1039995947@qq.com。
