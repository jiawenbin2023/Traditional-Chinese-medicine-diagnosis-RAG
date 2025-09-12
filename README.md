# Traditional-Chinese-medicine-diagnosis-RAG
Traditional Chinese medicine diagnosis-RAG
项目简介：企业级中医知识问答系统 (RAG)

#项目名称： 企业级中医知识问答系统 (Enterprise Traditional Chinese Medicine RAG System)

##项目背景与目标：

本项目旨在构建一个高效、精准的企业级中医知识问答系统，解决传统中医文献检索效率低下、信息孤岛严重的问题。通过集成先进的检索增强生成（RAG）技术，利用大规模语言模型（LLM）结合专业知识库，实现用户对复杂中医病症、诊断、治疗等问题的智能问答，提升知识获取效率和决策支持能力。

##核心技术栈与亮点：


    检索增强生成 (RAG) 架构： 采用混合检索策略（Dense + Sparse），结合语义检索和关键词检索优势，确保从海量中医知识库中召回最相关的上下文信息。

    LLM 集成 (HuggingFaceLLM)： 深度集成 HuggingFaceLLM，加载并利用 Qwen2-7B-Instruct 等大型语言模型进行答案生成，确保回答的准确性、连贯性和专业性。

    向量数据库 (Milvus)： 利用高性能向量数据库 Milvus 存储和管理中医知识文档的嵌入向量，实现高效的语义相似度检索。

    高性能嵌入与重排 (Embedding & Reranking)： 采用 BAAI/bge-large-zh 作为嵌入模型，提升中文文本的语义表示能力；引入 BAAI/bge-reranker-base 进行文档重排，优化检索结果的相关性，提高RAG召回质量。

    缓存机制 (Redis)： 集成 Redis 作为高性能缓存层，存储常见查询结果，显著提升系统响应速度和资源利用率。

    模块化与可扩展架构： 采用清晰的层次化设计（config, models, services, api, utils），确保系统的高内聚、低耦合，易于维护和功能扩展。

    Web 服务 (FastAPI)： 利用 FastAPI 框架构建高性能、异步的 RESTful API 服务，提供稳定、可扩展的后端支撑。

    交互式前端 (Streamlit)： 使用 Streamlit 快速开发直观、用户友好的交互式问答界面，方便用户进行实时提问和结果展示。


##负责工作与职责：


    系统架构设计： 参与 RAG 系统的整体架构设计，包括数据流程、模型选型、服务模块划分及前后端分离方案。

    核心模块开发： 负责 EnterpriseRAGService 核心业务逻辑的实现，协调嵌入模型、LLM、检索器、答案生成器等组件的协同工作。

    模型集成与优化： 完成 HuggingFaceLLM 对 Qwen2-7B-Instruct 的集成，并配置 BAAI/bge-large-zh 和 BAAI/bge-reranker-base 用于嵌入和重排，确保模型在 GPU 上的高效运行。

    向量存储与索引： 配置 Milvus 向量数据库，负责文档的嵌入、索引构建及检索策略的实现。

    性能优化与稳定性： 引入 Redis 缓存机制，优化 API 响应速度；通过日志监控和异常处理，提升系统运行的稳定性和可靠性。

    API 接口开发： 使用 FastAPI 设计并实现了 /query, /health, /metrics, /clear_cache 等核心 API 接口。

    前端界面开发： 基于 Streamlit 构建了用户友好的问答界面，实现了与后端 API 的数据交互与结果展示。

    环境搭建与调试： 负责 Python 虚拟环境的搭建、依赖管理（如 pip），并独立解决了复杂的 PyTorch、CUDA 兼容性问题以及前后端通信故障，确保项目顺利部署和运行。


##项目成果：


    成功部署并运行了基于 Qwen2-7B-Instruct 的企业级中医 RAG 问答系统，实现了对中医知识的智能问答能力。

    系统平均查询响应时间显著缩短（具体数据可根据实际运行情况补充），缓存命中率高，有效提升了用户体验。

    通过模块化设计，为后续功能扩展（如多模态输入、知识图谱集成）奠定了良好基础。

    （可选：如果你有实际运行的量化指标，可以补充如下） 在内部测试中，问答准确率达到 [xx]%，召回率达到 [xx]%。


