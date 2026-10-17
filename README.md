# Life Service AI Agent

> 基于大模型的 Re-Act 模式生活服务智能体 —— 自研异步推理引擎 + RAG 检索增强 + Function Calling 工具自动调度

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-47%20passed-brightgreen)](#测试)

---

## 项目简介

用户输入自然语言，Agent 自主推理并调用外部工具完成任务。系统从零实现（**不依赖 LangChain / AutoGen 等 Agent 框架**），核心推理循环、工具注册中心与检索链路均为自研，可清晰追溯每一步决策过程。

**本地运行**：见下方 [快速开始](#快速开始)，启动后访问 `http://localhost:8000`（Web UI）与 `/docs`（API 文档）。

### 核心能力

| 能力 | 实现 |
|---|---|
| **自主推理** | 自研异步 Re-Act 引擎，`Thought → Action → Observation` 循环，最多 10 轮迭代 |
| **工具调度** | Tool Registry 工具注册中心，20 个工具零硬编码自动调度 |
| **知识检索** | FAISS 向量检索 + 滑动窗口分块，Top-3 召回，结果自动注入 System Prompt |
| **流式输出** | SSE 实时推送完整推理链，前端可逐字看到思考过程 |
| **多轮会话** | SQLAlchemy 异步持久化，会话历史可查、可删 |
| **稳定性治理** | 指数退避重试 + 分层超时 + 超时熔断 + 异常隔离 |

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                      浏览器 / API 调用方                       │
│                    http://localhost:8000                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      FastAPI 服务层                           │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────────┐  │
│  │  路由层     │  │   中间件     │  │      静态资源         │  │
│  │ /chat      │  │ RequestID   │  │   index.html (Web UI) │  │
│  │ /chat/stream│ │ Timing      │  │                       │  │
│  │ /conversations│ ErrorHandler│ │                       │  │
│  │ /tools     │  │             │  │                       │  │
│  │ /health    │  └─────────────┘  └──────────────────────┘  │
│  └──────┬─────┘                                              │
└─────────┼────────────────────────────────────────────────────┘
          │ 依赖注入
┌─────────▼────────────────────────────────────────────────────┐
│                    ReActEngine（异步推理引擎）                  │
│                                                              │
│   while 未完成 && 迭代 < 10:                                  │
│     ① 调 LLM（携带工具 Schema 列表）                           │
│     ② LLM 返回 tool_calls？                                   │
│        是 → 执行工具 → Observation 追加 → 回到 ①               │
│        否 → 输出最终答案，结束                                 │
│                                                              │
│   稳定性：指数退避重试(3次) / LLM 60s 超时 / 工具 30s 超时       │
└───┬──────────────────┬───────────────────┬───────────────────┘
    │                  │                   │
┌───▼────────┐  ┌──────▼───────┐  ┌────────▼──────────┐
│ ToolRegistry│  │ RAG Retriever │  │   LLM Client      │
│ 工具注册中心 │  │              │  │                   │
│            │  │ FAISS 向量库  │  │  DashScope        │
│ 20 个工具   │  │ 滑动窗口分块  │  │  (qwen-plus)      │
│ 自动调度    │  │ Top-K 召回    │  │  OpenAI 兼容协议   │
└────────────┘  └──────────────┘  └───────────────────┘
```

### 目录结构

```
src/
├── agent/              # 推理引擎
│   ├── react_engine.py #   Re-Act 核心循环（execute / stream）
│   ├── core.py         #   Agent 编排
│   └── types.py        #   类型定义（AgentEvent / AgentResult）
├── tools/              # 工具层
│   ├── base.py         #   ToolRegistry 注册中心 + ToolDefinition
│   ├── compute_tools.py#   计算类工具（4）
│   ├── file_tools.py   #   文件类工具（7）
│   ├── mcp_tools.py    #   MCP 服务工具（8）
│   └── rag_tool.py     #   检索工具（1）
├── rag/                # 检索增强
│   ├── document_processor.py  # 文档解析 + 滑动窗口分块
│   ├── embedding.py           # 向量化（DashScope Embedding）
│   ├── vector_store.py        # FAISS 索引管理
│   └── retriever.py           # Top-K 召回
├── server/             # Web 服务
│   ├── app.py          #   App 工厂 + lifespan
│   ├── routers/        #   chat / health / tools
│   ├── middleware/     #   请求 ID / 耗时统计 / 错误处理
│   └── static/         #   Web 聊天界面
├── core/               # 基础设施
│   ├── config.py       #   Pydantic Settings 配置管理
│   ├── logging.py      #   structlog 结构化日志
│   └── exceptions.py   #   自定义异常体系
├── llm/                # LLM 客户端封装
├── models/             # SQLAlchemy 数据模型
└── repository/         # 数据访问层
```

---

## 技术栈

| 层次 | 技术选型 |
|---|---|
| **语言** | Python 3.11+（asyncio / AsyncGenerator 全异步） |
| **Agent** | 自研 Re-Act 引擎 · Function Calling · Tool Registry |
| **RAG** | FAISS · DashScope Embedding · 滑动窗口分块 · Top-K 召回 |
| **Web** | FastAPI · Uvicorn · SSE（sse-starlette） |
| **数据** | SQLAlchemy 2.0 Async · aiosqlite |
| **校验** | Pydantic v2 · pydantic-settings |
| **可观测** | structlog 结构化日志 · 请求链路追踪（Request-ID） |
| **稳定性** | tenacity 指数退避重试 · 分层超时 · 熔断隔离 |
| **测试** | pytest · pytest-asyncio · pytest-cov |
| **部署** | Docker 多阶段构建 · docker-compose |

---

## 可用工具（20 个）

| 分类 | 工具 |
|---|---|
| **计算** | `calculate` · `unit_convert` · `date_calc` · `text_stats` |
| **文件** | `read_file` · `write_file` · `delete_file` · `list_directory` · `create_directory` · `search_files` · `search_content` |
| **服务（MCP）** | `get_weather` · `food_recommend` · `mbti_query` · `mbti_questions` · `mbti_calculate` · `get_coordinates` · `city_distance` · `get_current_time` |
| **检索** | `rag_search` |

---

## 快速开始

### 方式一：Docker（推荐）

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY

# 2. 启动
docker compose up -d

# 3. 访问
# Web UI   → http://localhost:8000
# API 文档  → http://localhost:8000/docs
```

### 方式二：本地运行

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2. 安装依赖
pip install -e ".[dev,faiss]"
# 或（不安装为包时）：pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env         # 填入 DASHSCOPE_API_KEY

# 4. 启动服务
make run                     # 生产模式
make dev                     # 开发模式（热重载）
```

---

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | — | **必填**，阿里云百炼 API Key |
| `LLM_MODEL` | `qwen-plus` | LLM 模型名 |
| `LLM_TIMEOUT` | `60` | LLM 调用超时（秒） |
| `LLM_MAX_RETRIES` | `3` | 重试次数 |
| `AGENT_MAX_ITERATIONS` | `10` | Re-Act 最大迭代轮数 |
| `AGENT_TOOL_TIMEOUT` | `30` | 单工具执行超时（秒） |
| `RAG_CHUNK_SIZE` | `500` | 分块大小（字符） |
| `RAG_CHUNK_OVERLAP` | `50` | 分块重叠 |
| `RAG_TOP_K` | `3` | 召回条数 |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/agent.db` | 数据库连接串 |

> 完整配置项见 `.env.example`

---

## API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/v1/chat` | 同步对话，返回完整结果 |
| `GET` | `/api/v1/chat/stream` | **SSE 流式对话**，实时推送推理链 |
| `GET` | `/api/v1/conversations` | 会话列表 |
| `GET` | `/api/v1/conversations/{id}` | 会话详情（含消息历史） |
| `DELETE` | `/api/v1/conversations/{id}` | 删除会话 |
| `GET` | `/api/v1/tools` | 已注册工具清单 |
| `GET` | `/health` | 健康检查 |
| `GET` | `/ready` | 就绪探针（含依赖检查） |

### 调用示例

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "杭州今天天气怎么样？顺便帮我算一下 128 * 36"}'
```

---

## 测试

```bash
make test          # 运行全部测试（47 个用例）
make test-html     # 生成 HTML 覆盖率报告
make check         # lint + test 全量检查
```

覆盖维度：工具注册与调度、计算工具、文件工具、MCP 工具、API 接口、安全校验。

---

## 工程设计要点

**1. 为什么自研 Re-Act 引擎而不是用 LangChain？**

框架封装了推理循环，出问题时难以定位是「Prompt 问题」还是「框架问题」。自研后每一步 `Thought / Action / Observation` 都可见可控，SSE 可以逐事件推送，调试和优化路径清晰。

**2. 工具如何做到零硬编码调度？**

`ToolRegistry` 统一管理工具元数据（名称、描述、参数 JSON Schema）。调用 LLM 时自动把全部工具的 Schema 注入 `tools` 参数，由模型决定调用哪个——新增工具只需注册，不改推理逻辑。

**3. 稳定性如何保障？**

- **指数退避重试**：LLM 网络抖动时 1s → 2s → 4s 重试
- **分层超时**：LLM 调用 60s、单工具执行 30s，避免一个慢工具阻塞整条链路
- **异常隔离**：工具执行失败返回错误 Observation，推理循环继续，不中断整个任务

**4. 可观测性怎么做的？**

- `RequestIDMiddleware` 为每个请求生成唯一 ID，贯穿日志
- `TimingMiddleware` 记录耗时
- structlog 输出结构化日志，便于检索和聚合

---

## License

MIT
