# EAT-RAG

> 基于 LangGraph 构建的 Agentic RAG 智能食谱助手，实现混合检索、查询扩展、相关性评估、流式输出与可观测 RAG Pipeline。

## 项目简介

EAT-RAG 是一个面向本地食谱知识库的 Agentic RAG 项目。系统以本地 Markdown 菜谱数据为知识库，结合向量检索、关键词检索、融合排序、查询改写、相关性评估与 Agent 工具调用，实现菜谱问答、做法查询、食材推荐、烹饪技巧查询等功能。

与传统 RAG 系统相比，本项目在 RAG 外层接入 LangGraph Agent，让大模型根据用户问题自主判断是否需要调用食谱检索工具。同时，系统在检索链路中加入 Step-Back、HyDE、复杂查询扩展与 LLM 相关性评分，用于提升复杂问题、多约束问题和模糊问题下的检索效果。

当前版本支持 FastAPI 后端、SSE 流式输出、前端实时展示 Agent 步骤与 RAG 检索过程，并记录完整 trace，便于调试、评估和后续优化。

---

## 项目特点

### 1. Agent 驱动的 RAG 架构

项目基于 LangGraph 构建 ReAct Agent，由 Agent 自主判断：

* 当前问题是否需要调用食谱知识库
* 应该如何组织检索查询
* 工具返回结果后如何生成最终回答
* 非食谱问题是否直接回答或引导用户回到食谱场景

系统从早期的“规则判断 + 单次 RAG 调用”升级为“Agent 决策 + 工具调用 + 可观测 RAG Pipeline”。

### 2. 混合检索

检索模块同时使用：

* FAISS 向量检索
* BM25 关键词检索
* RRF 融合排序
* 元数据过滤检索

通过向量语义召回与关键词精确匹配结合，提高菜谱名称、食材、步骤、技巧、时间、火候等不同类型问题的召回质量。

### 3. 查询改写与查询扩展

系统支持多种查询优化策略：

* Query Rewrite：将模糊问题改写成更适合检索的查询
* Step-Back：将具体问题抽象为更上位的问题，补充背景信息后重新检索
* HyDE：生成一段假想食谱文档，用于辅助召回相关内容
* Complex：同时执行 Step-Back 和 HyDE，并合并两路检索结果

当初始检索结果相关性不足时，系统会自动触发二次检索。

### 4. 检索结果相关性评估

项目实现了基于 LLM 的 `grade_documents` 模块，用于判断初始检索结果是否足够回答用户问题。

该模块会输出：

* 是否相关
* 置信度
* 相关文档编号
* 判断原因

如果检索结果相关性不足，系统会进入查询扩展阶段，避免低质量检索结果直接进入生成环节。

### 5. 可观测 RAG Pipeline

系统记录完整的 RAG 执行链路，包括：

* 查询路由
* 查询改写
* 初始检索
* 文档相关性评分
* 查询扩展策略
* 二次检索
* 父文档扩展
* 最终回答生成
* 检索耗时与命中文档

前端可以实时展示 Agent step 和 RAG step，方便观察模型是否正确调用工具、检索是否命中、是否触发二次检索。

### 6. 流式交互

后端基于 FastAPI + SSE 实现流式输出，支持：

* token 级别回答输出
* Agent 执行状态推送
* RAG 检索步骤推送
* 最终 trace 返回
* 多轮会话管理

用户可以在前端实时看到回答生成过程和检索链路。

---

## 核心功能

* 本地菜谱知识库构建
* FAISS 向量索引构建
* BM25 关键词检索
* RRF 融合排序
* LangGraph Agent 工具调用
* 食谱问答
* 菜品推荐
* 食材用量查询
* 做法步骤查询
* 火候与时间查询
* 烹饪技巧查询
* 多条件组合推荐
* 多轮上下文对话
* RAG trace 实时展示
* SSE 流式输出

---

## 技术栈

### 大模型与 Agent

* LangGraph
* LangChain
* Qwen API

### 检索与 RAG

* FAISS
* BM25
* RRF 融合排序
* Query Rewrite
* Step-Back
* HyDE
* LLM 相关性评分

### 后端

* FastAPI
* Server-Sent Events
* Pydantic

### 前端

* HTML
* JavaScript
* CSS

### 数据

* Markdown 本地食谱知识库

---

## 系统架构

```text
用户输入
  ↓
FastAPI 流式接口
  ↓
读取会话历史，必要时压缩旧对话
  ↓
LangGraph Agent
  ↓
判断是否调用 recipe_rag_search 工具
  ↓
RecipeRAGTool
  ↓
RecipeRAGSystem.ask_with_trace()
  ↓
RAG Pipeline
  ├── 查询路由
  ├── 查询改写
  ├── 初始检索
  ├── 文档相关性评分
  ├── 查询扩展
  ├── 二次检索
  ├── 父文档扩展
  └── 最终回答生成
  ↓
保存 RAG trace
  ↓
Agent 生成最终回答
  ↓
前端流式展示回答与 trace
```

---

## RAG 执行流程

```text
用户问题
  ↓
query_router
  ↓
query_rewrite
  ↓
retrieve_initial
  ↓
grade_documents
  ↓
如果相关性不足：
  rewrite_question
  ↓
  step_back / hyde / complex
  ↓
  retrieve_expanded
  ↓
parent document expansion
  ↓
generate answer
```

---

## 查询路由

系统会根据用户问题类型进行路由：

| 类型          | 说明             |
| ----------- | -------------- |
| list        | 推荐类、列表类问题      |
| ingredients | 食材、原料、调料、用量问题  |
| steps       | 做法、教程、制作步骤问题   |
| tips        | 技巧、注意事项、失败原因问题 |
| time        | 时间、温度、火候问题     |
| general     | 其他一般问题         |

路由结果会影响后续检索和回答生成方式。

---

## 查询扩展策略

### Step-Back

适用于技巧类、火候类、失败原因类和模糊追问类问题。

例如：

```text
这个怎么做才不会老？
```

系统会先生成更上位的问题和背景答案，再结合原问题重新检索。

### HyDE

适用于推荐类、多约束类、宽泛需求类问题。

例如：

```text
只有鸡蛋和青菜，20分钟内能做什么？
```

系统会生成一段可能出现在菜谱知识库中的假想文档，再用这段文档辅助检索。

### Complex

适用于初始检索明显不足、问题约束较多或语义较复杂的场景。

系统会同时执行 Step-Back 和 HyDE 两路检索，并对结果进行合并、去重和排序。

---

## RAG Trace 示例

```json
{
  "tool_name": "recipe_rag_search",
  "tool_query": "只有鸡蛋和青菜，20分钟内能做什么？",
  "tool_used": true,
  "hit": true,
  "route_type": "list",
  "rewritten_query": "鸡蛋 青菜 20分钟 快手菜 推荐",
  "retrieval_grade": {
    "binary_score": "no",
    "confidence": 0.42,
    "reason": "初检文档只匹配鸡蛋，没有覆盖青菜和时间约束"
  },
  "rewrite_triggered": true,
  "rewrite_strategy": "complex",
  "retrieval_mode": "hybrid+complex",
  "retrieved_chunk_count": 5,
  "retrieved_dish_names": ["青菜炒鸡蛋", "鸡蛋青菜汤"],
  "elapsed_ms": 1830
}
```

---

## 项目结构

```text
.
├── code/
│   ├── backend/
│   │   ├── app.py              # FastAPI 接口、SSE 流式输出、session 管理
│   │   ├── agent.py            # LangGraph Agent 执行器
│   │   ├── rag_tool.py         # recipe_rag_search 工具封装与 trace 记录
│   │   └── schemas.py          # API、Agent、trace 数据结构
│   │
│   ├── frontend/
│   │   ├── index.html          # 前端页面
│   │   ├── app.js              # 对话、SSE 消费、trace 展示
│   │   └── styles.css          # 页面样式
│   │
│   ├── rag_modules/
│   │   ├── data_preparation.py
│   │   ├── index_construction.py
│   │   ├── retrieval_optimization.py
│   │   └── generation_integration.py
│   │
│   ├── main.py                 # RAG 主流程与 ask_with_trace()
│   ├── config.py               # 默认配置
│   └── requirements.txt
│
└── data/
    └── cook/                   # 本地食谱 Markdown 知识库
```

---

## API 说明

### 健康检查

```http
GET /api/health
```

### 数据流说明

```http
GET /api/flow
```

### 普通对话

```http
POST /api/chat
```

请求示例：

```json
{
  "message": "番茄炒蛋怎么做？",
  "session_id": null,
  "include_trace": true
}
```

### 流式对话

```http
POST /api/chat/stream
```

SSE 返回事件：

```text
data: {"type":"agent_step","step":{...}}

data: {"type":"rag_step","step":{...}}

data: {"type":"token","content":"..."}

data: {"type":"final","response":{...}}

data: {"type":"done"}
```

---

## 本地运行

建议使用 Python 3.10 到 3.12。

部分依赖对 Python 3.13 的支持不稳定，尤其是 `langchain-unstructured` 间接依赖的 `onnxruntime`。

### 1. 安装依赖

```bash
cd code
python -m pip install -r requirements.txt
```

### 2. 配置模型 API Key

```bash
export QWEN_API_KEY="your-api-key"
```

### 3. 启动服务

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

### 4. 打开前端页面

```text
http://127.0.0.1:8000/
```

---

## 测试示例

### 1. 寒暄类问题

```text
你好
```

预期结果：

* Agent 直接回答
* 不调用 RAG 工具
* `tool_trace` 为空

### 2. 明确菜谱问题

```text
番茄炒蛋怎么做？
```

预期结果：

* Agent 调用 `recipe_rag_search`
* RAG trace 展示路由、改写、检索和评分过程
* 如果初检结果足够相关，不触发二次检索

### 3. 模糊推荐问题

```text
我想吃点清淡的，但不要太麻烦，有什么推荐？
```

预期结果：

* 系统识别为推荐类问题
* 可能触发 HyDE 或 Complex 查询扩展
* trace 中展示扩展查询内容

### 4. 技巧类问题

```text
鸡胸肉怎么做才不会柴？
```

预期结果：

* 系统识别为技巧类问题
* 可能触发 Step-Back 查询扩展
* trace 中展示 step_back_question 和 step_back_answer

### 5. 多约束问题

```text
只有鸡蛋和青菜，20分钟内能做什么？
```

预期结果：

* 系统识别为推荐类或多约束问题
* 初检不足时触发 Complex 查询扩展
* Step-Back 和 HyDE 两路结果合并后生成回答

### 6. curl 测试流式接口

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"只有鸡蛋和青菜，20分钟内能做什么？","include_trace":true}'
```

---

## 后续优化方向

* 将 RAG Pipeline 拆分为更清晰的 LangGraph 节点
* 为 `grade_documents`、Step-Back、HyDE 构建评测集
* 增加 rerank 模型，提高检索排序质量
* 增加购物清单生成工具
* 增加一周菜单规划工具
* 增加食材替换工具
* 将 session 从内存迁移到 SQLite 或 PostgreSQL
* 将 RAG trace 持久化，用于离线评估
* 优化前端 timeline 展示效果

---

## 项目亮点总结

* 基于 LangGraph 实现 Agentic RAG 架构
* 支持 Agent 自主判断是否调用食谱检索工具
* 使用 FAISS + BM25 + RRF 实现混合检索
* 引入 LLM 相关性评分，控制是否触发二次检索
* 支持 Step-Back、HyDE 和 Complex 查询扩展
* 支持 FastAPI + SSE 流式输出
* 支持前端实时展示 Agent 步骤和 RAG 检索链路
* 支持多轮会话记忆与摘要压缩
* 形成完整的 Agent + RAG 工程化实践闭环
