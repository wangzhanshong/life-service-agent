# Life Agent 快速上手指南

> 基于大模型的 Re-Act 模式生活服务 AI Agent — 5 分钟了解全貌

---

## 一句话概述

用户输入自然语言 → Agent 自动推理 + 调用工具（天气/计算/文件/美食/MBTI/RAG）→ 返回结果。支持 Web 聊天界面和 API 调用。

---

## 项目架构图

```
浏览器 (http://localhost:8000)
        │
        ▼
┌──────────────────────────────────────────────┐
│               FastAPI 服务器                    │
│  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ 路由层    │  │ 中间件    │  │ 静态文件     │  │
│  │ /chat    │  │ RequestID │  │ index.html  │  │
│  │ /stream  │  │ Timing    │  │             │  │
│  │ /tools   │  │ ErrorHdr  │  │             │  │
│  │ /health  │  └──────────┘  └─────────────┘  │
│  └────┬─────┘                                  │
│       │ Depends(get_engine)                     │
│       ▼                                         │
│  ┌──────────────────────────────────────┐      │
│  │         ReActEngine (异步引擎)         │      │
│  │                                      │      │
│  │  while 未完成:                        │      │
│  │    ① 调 LLM (带工具列表)              │      │
│  │    ② LLM 返回→有 tool_calls?          │      │
│  │       是→执行工具→追加结果→回到①     │      │
│  │       否→最终答案，结束              │      │
│  └──┬──────────────┬───────────────────┘      │
│     │              │                           │
│     ▼              ▼                           │
│ ┌────────┐  ┌─────────────────────┐           │
│ │LLM客户端│  │   ToolRegistry      │           │
│ │(Async  │  │  ┌─ calculate       │           │
│ │ OpenAI)│  │  ├─ get_weather     │           │
│ │        │  │  ├─ city_distance   │           │
│ │ 重试+  │  │  ├─ mbti_calculate  │           │
│ │ 超时)  │  │  ├─ food_recommend  │           │
│ └────────┘  │  ├─ read_file       │           │
│              │  ├─ write_file      │           │
│              │  ├─ rag_search      │           │
│              │  └─ ...共 19 个     │           │
│              └─────────────────────┘           │
│                                                │
│  数据层: SQLAlchemy Async + SQLite             │
│  配置:   pydantic-settings + .env              │
│  日志:   structlog (fallback→logging)          │
└──────────────────────────────────────────────┘
```

---

## 目录结构速览

```
基于大模型的生活服务综合智能体Agent/
│
├── src/                          ← 核心代码
│   ├── agent/                    ← Agent 引擎
│   │   ├── react_engine.py       ★ ReAct 循环 (最重要)
│   │   ├── core.py               ★ 异步 Agent 基类
│   │   ├── types.py              数据模型 (事件类型)
│   │   └── tool_registry.py      工具注册表 (已弃用，见tools/)
│   │
│   ├── tools/                    ← 工具集合
│   │   ├── base.py               ★ ToolRegistry + ToolDefinition
│   │   ├── mcp_tools.py          ★ 天气/地理/MBTI/美食 4 类工具
│   │   ├── compute_tools.py      数学 + 日期 + 文本统计
│   │   ├── file_tools.py         文件读写/搜索 + 路径沙箱
│   │   └── rag_tool.py           RAG 检索工具
│   │
│   ├── llm/
│   │   └── client.py             ★ LLM 调用 + 重试 + 超时
│   │
│   ├── server/                   ← Web 服务层
│   │   ├── app.py                ★ FastAPI 应用工厂
│   │   ├── dependencies.py       ★ 依赖注入 (get_engine/get_repo)
│   │   ├── routers/
│   │   │   ├── chat.py           ★ 聊天 API + SSE 流式
│   │   │   ├── tools.py          工具列表 API
│   │   │   └── health.py         健康检查
│   │   ├── middleware/           请求ID/耗时/异常处理
│   │   └── static/index.html    ★ Web 聊天界面
│   │
│   ├── rag/                      ← RAG 子系统
│   │   ├── retriever.py          ★ 检索编排
│   │   ├── embedding.py          文本→向量 (DashScope/fallback)
│   │   ├── vector_store.py       FAISS 向量库 (JSON 安全存储)
│   │   └── document_processor.py 文档分块
│   │
│   ├── models/                   ← 数据库模型
│   │   ├── base.py               ★ 异步引擎 + Session 工厂
│   │   ├── conversation.py       会话表
│   │   └── message.py            消息表
│   │
│   ├── repository/
│   │   └── conversation_repo.py  ★ 会话 CRUD (Repository 模式)
│   │
│   └── core/                     ← 基础设施
│       ├── config.py             ★ 配置 (pydantic-settings)
│       ├── exceptions.py         异常体系 (7 种异常)
│       ├── security.py           ★ 路径沙箱 + 输入净化
│       └── logging.py            结构化日志
│
├── tests/                        ← 44 个测试
├── .env                          ← 环境变量 (API Key 等)
├── pyproject.toml                ← 项目配置 + 依赖
├── Makefile                      ← 常用命令
├── Dockerfile / docker-compose   ← 容器化
│
└── (旧版 v1 文件保留)             agent/, compute/, file_ops/, mcp/, main.py ...
    ─────────────────             (功能被 src/ 替代，保留作参考)
```

> 标 ★ 的是**最核心、最值得先读**的文件

---

## 5 个核心文件速览

### 1. 入口：[src/server/app.py](src/server/app.py)
```python
# FastAPI 应用工厂，组装所有组件
app = FastAPI(lifespan=lifespan)   # 启动时创建数据库表 + 初始化RAG
app.add_middleware(...)            # 请求ID / 耗时 / CORS / 异常处理
app.include_router(chat.router)    # /api/v1/chat
app.include_router(tools.router)   # /api/v1/tools
app.include_router(health.router)  # /health
app.mount("/", StaticFiles(...))  # 前端页面
```

### 2. ReAct 引擎：[src/agent/react_engine.py](src/agent/react_engine.py)
```python
class ReActEngine:
    async def stream(self, user_input) -> AsyncGenerator[AgentEvent, None]:
        for iteration in range(max_iterations):
            response = await llm.chat(messages, tools)     # 问 LLM
            if not response.tool_calls:
                yield FINAL_ANSWER; return                 # 没工具调用→完成
            for tc in response.tool_calls:
                result = await registry.execute(tc.name, tc.args)  # 执行工具
                messages.append({"role": "tool", "content": result}) # 追加结果
                yield OBSERVATION
```

### 3. 工具注册：[src/tools/base.py](src/tools/base.py)
```python
class ToolRegistry:
    def register(self, tool: ToolDefinition): ...          # 注册工具
    def execute(self, name, args) -> str:                 # 执行工具 (带超时)
    def to_openai_functions(self) -> List[Dict]:           # 生成 LLM 格式
```

### 4. LLM 客户端：[src/llm/client.py](src/llm/client.py)
```python
class LLMClient:
    @retry(stop=3, wait=exponential)  # 自动重试 3 次
    async def chat(self, messages, tools=None):
        await self._client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            tools=tools,              # 工具列表
            tool_choice="auto",       # LLM 自己决定用不用
        )
```

### 5. 聊天路由：[src/server/routers/chat.py](src/server/routers/chat.py)
```python
# 普通请求 → 返回完整结果
@router.post("/chat")
async def chat(req, engine, repo):
    result = await engine.execute(req.message)
    return ChatResponse(answer=result.answer, ...)

# SSE 流式 → 逐步推送推理过程
@router.get("/chat/stream")
async def chat_stream(message, engine, repo):
    async def gen():
        async for event in engine.stream(message):
            yield {"data": event.model_dump_json()}  # 推送到浏览器
    return EventSourceResponse(gen())
```

---

## 数据流向图

```
用户发送 "北京天气？"
        │
        ▼
┌──────────────────┐
│ FastAPI 接收请求   │ → 中间件注入 request_id → 安全净化输入
└──────┬───────────┘
       │ engine.stream("北京天气？")
       ▼
┌──────────────────┐
│ ReActEngine 启动   │ → 构建 messages = [system + user]
└──────┬───────────┘
       │ llm.chat(messages, tools=19个工具)
       ▼
┌──────────────────┐
│ LLM 返回          │ → content: "我来查一下"
│                   │ → tool_calls: [{name: "get_weather", args: {city: "北京"}}]
└──────┬───────────┘
       │ registry.execute("get_weather", {"city": "北京"})
       ▼
┌──────────────────┐
│ 工具执行          │ → _query_weather("北京")
│                   │ → 返回: {temperature: 22, condition: "晴"}
└──────┬───────────┘
       │ 结果追加到 messages
       ▼
┌──────────────────┐
│ LLM 再次调用      │ → messages 现在包含天气数据
│                   │ → 返回纯文本: "北京今天晴，22°C"
│                   │ → tool_calls = []  ← 循环终止
└──────┬───────────┘
       │ yield FINAL_ANSWER
       ▼
┌──────────────────┐
│ SSE 推送到浏览器   │ → 前端渲染聊天气泡
│ 持久化到 SQLite   │ → 保存 conversation + message
└──────────────────┘
```

---

## 关键数字

| 项目 | 数值 |
|------|------|
| 工具总数 | 19 个（compute×4 + file×7 + mcp×7 + rag×1） |
| API 端点 | 7 个（chat / stream / tools / conversations / health / ready / static） |
| 数据库表 | 2 张（conversations + messages） |
| 中间件 | 3 个（RequestID / Timing / ErrorHandler） |
| 异常类型 | 7 种（Agent / LLM / Tool / Security / Config / RAG / RateLimit） |
| 重试次数 | LLM 最多 3 次，指数退避 1s→2s→4s |
| 超时控制 | LLM 60s / 工具 30s / 连接 10s |
| 测试数量 | 44 个，6 个维度 |

---

## 快速运行

```bash
# 1. 配置 API Key
echo "DASHSCOPE_API_KEY=sk-your-key" > .env

# 2. 启动
make dev
# 或: python -m uvicorn src.server.app:create_app --factory --reload

# 3. 访问
浏览器打开 http://localhost:8000

# 4. 测试
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  --data-raw '{"message":"北京天气"}'

# 5. 跑测试
make test   # 44 passed
```

---

## 怎么改代码？常见需求

| 想做的事 | 改哪个文件 | 怎么改 |
|----------|-----------|--------|
| **新增一个工具** | `src/tools/mcp_tools.py` | 写一个函数 → 在 `create_mcp_tools()` 里注册 |
| **修改系统 Prompt** | `src/llm/client.py` | 改 `SYSTEM_PROMPT` 常量 |
| **更换 LLM 模型** | `.env` 或 `src/core/config.py` | 改 `LLM_MODEL=qwen-max` |
| **调整推理步数** | `.env` | 改 `AGENT_MAX_ITERATIONS=15` |
| **修改界面样式** | `src/server/static/index.html` | 改 CSS，刷新即生效 |
| **添加新的 API 端点** | `src/server/routers/` | 新增路由文件 → 在 `app.py` 注册 |
| **新增异常类型** | `src/core/exceptions.py` | 添加类 → 在 `error_handler.py` 注册 |
| **切换数据库** | `.env` | 改 `DATABASE_URL=postgresql+asyncpg://...` |
