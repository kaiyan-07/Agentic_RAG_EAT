# EAT_RAG

EAT_RAG 是一个面向本地食谱知识库的 Agentic RAG 项目。系统以本地菜谱 Markdown 数据为知识库，构建向量索引与 BM25 检索，并在 RAG 外层接入 LangGraph Agent，让模型可以自行判断是否调用食谱检索工具，再通过可观测的 RAG pipeline 生成回答。

当前版本已经从早期的“规则判断 + 单次 RAG 调用”升级为：

- LangGraph ReAct Agent 工具调用
- FastAPI + SSE 流式输出
- RAG trace 旁路记录
- 每轮工具调用硬约束
- 会话记忆与摘要压缩
- Step-Back / HyDE / complex 查询扩展
- LLM 二分类 `grade_documents` 相关性门控
- 前端实时展示 Agent step 和 RAG step

## 当前功能

- 基于本地食谱数据构建向量索引
- 使用 FAISS 向量检索 + BM25 检索 + RRF 融合排序
- 支持食谱问答、做法查询、食材用量、火候时间、烹饪技巧和菜品推荐
- 使用 LangGraph Agent 决定是否调用 `recipe_rag_search`
- 将现有 `RecipeRAGSystem.ask_with_trace()` 封装为 Agent 工具
- 每轮最多调用一次 RAG 工具，避免反复检索和成本失控
- 使用 SSE 流式推送 token、Agent 步骤、RAG 步骤和最终 trace
- 使用结构化 `grade_documents` 判断初检结果是否足够相关
- 相关性不足时自动选择 Step-Back、HyDE 或 complex 双路扩展后重新检索
- 前端显示完整 trace，包括路由、查询改写、评分、重写策略、扩展查询、命中文档和耗时

## 项目结构

```text
.
├── code/
│   ├── backend/
│   │   ├── app.py          # FastAPI API、SSE 流式接口、session 管理
│   │   ├── agent.py        # LangGraph Agent 执行器
│   │   ├── rag_tool.py     # recipe_rag_search 工具封装与 trace 旁路
│   │   └── schemas.py      # API / Agent / trace 数据结构
│   ├── frontend/
│   │   ├── index.html      # 简单 Web UI
│   │   ├── app.js          # 对话、SSE 消费、trace 展示
│   │   └── styles.css
│   ├── rag_modules/
│   │   ├── data_preparation.py
│   │   ├── index_construction.py
│   │   ├── retrieval_optimization.py
│   │   └── generation_integration.py  # LLM 生成、路由、评分、Step-Back、HyDE
│   ├── main.py             # RAG 主流程与 ask_with_trace()
│   ├── config.py           # 默认配置
│   └── requirements.txt
└── data/
    └── cook/               # 本地食谱知识库
```

## 整体架构

```text
用户输入
  ↓
FastAPI /api/chat/stream
  ↓
读取 session history，必要时压缩旧对话
  ↓
LangGraph RecipeAgent
  ↓
模型根据 system prompt + 工具描述自行决定是否调用 recipe_rag_search
  ↓
RecipeRAGTool
  ↓
RecipeRAGSystem.ask_with_trace()
  ↓
RAG pipeline:
  query_router
  query_rewrite
  retrieve_initial
  grade_documents
  rewrite_question(step_back / hyde / complex, if needed)
  retrieve_expanded(if needed)
  parent document expansion
  final answer generation
  ↓
RAG trace 旁路保存
  ↓
Agent 生成最终回答
  ↓
前端流式展示回答与 trace
```

## LangGraph Agent 实现

Agent 入口在 `code/backend/agent.py`。

当前 Agent 使用 `langgraph.prebuilt.create_react_agent` 构建 ReAct-style agent。核心目标是让 LLM 自己决定：

- 当前问题是否需要调用工具
- 应该用什么查询调用 `recipe_rag_search`
- 工具返回后如何生成最终回答
- 非食谱问题是否直接拒答或引导用户回到食谱场景

Agent 的 system prompt 明确了工具约束：

- 食谱、食材、步骤、时间、火候、技巧和推荐问题应调用 `recipe_rag_search`
- 每轮最多调用一次 RAG 工具
- 收到 RAG 结果后必须直接产出最终回答
- 不要在 RAG 上下文不足时硬编
- 寒暄或明显非食谱问题不调用工具

代码层也有硬约束。`TurnState.rag_calls` 会记录本轮 RAG 调用次数，`RecipeRAGTool.run()` 会再次检查 `max_calls_per_turn`。即使模型试图重复调用工具，第二次也会返回：

```text
TOOL_CALL_LIMIT_REACHED: use existing retrieval result and answer directly.
```

这意味着工具调用限制不只依赖 prompt。

## 流式输出

流式接口在 `code/backend/app.py`：

```text
POST /api/chat/stream
```

它使用 SSE 返回统一事件流。事件类型包括：

- `agent_step`：Agent 执行状态
- `rag_step`：RAG pipeline 状态
- `token`：模型输出文本 token
- `final`：最终完整响应，包含 answer、tool_trace、agent_steps
- `done`：本轮结束
- `error`：异常信息

`RecipeAgent.astream_events()` 内部会创建一个后台任务执行 LangGraph agent，主生成器只从统一 `output_queue` 读取事件并 yield。这样即使 RAG 工具是同步执行的，也能通过 `emit_event()` 把 RAG 步骤实时推给前端。

前端在 `code/frontend/app.js` 中读取流：

```text
fetch("/api/chat/stream")
  ↓
reader.read()
  ↓
解析 data: {...}
  ↓
token 追加到 assistant 消息
rag_step / agent_step 追加到 trace 面板
final 渲染完整 trace
```

## RAG 工具封装

`code/backend/rag_tool.py` 将已有 RAG 系统封装为 Agent 工具：

```text
recipe_rag_search(query: str) -> str
```

工具内部做了几件事：

- lazy 初始化 `RecipeRAGSystem`
- 调用 `ask_with_trace()`
- 将完整 trace 存在旁路变量中
- 返回给 Agent 的文本保持简洁
- 把完整 trace 返回给 API 和前端
- 记录耗时、命中状态、检索模式和命中文档

这避免把所有 metadata 都塞给模型。Agent 只吃必要上下文，系统仍然能保存完整调试信息。

## RAG Pipeline

核心流程在 `code/main.py` 的 `_run_query_pipeline()`。

当前 RAG pipeline 是：

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
如果相关性不足:
  rewrite_question
  retrieve_expanded
  ↓
parent document expansion
  ↓
generate answer
```

### 查询路由

`GenerationIntegrationModule.query_router()` 会把问题分成：

- `list`：推荐/列表类问题
- `ingredients`：食材、原料、调料、用量
- `steps`：完整做法、教程、制作步骤
- `tips`：技巧、注意事项
- `time`：时间、温度、火候
- `general`：其他一般问题

路由结果会影响后续回答生成方式。

### 初始查询改写

`query_rewrite()` 负责把模糊问题改写成更适合检索的查询。

例如：

```text
推荐个菜
→ 简单家常菜推荐
```

具体明确的问题会保持原样。

### 初始检索

初始检索使用 `RetrievalOptimizationModule.hybrid_search()`：

- FAISS 向量检索
- BM25 关键词检索
- RRF 合并排序

如果问题中包含分类或难度关键词，会走 `metadata_filtered_search()`。

## grade_documents 相关性门控

`grade_documents()` 在 `code/rag_modules/generation_integration.py`。

它是一个 LLM 二分类评分器，用于判断初检结果是否足以回答问题。输出结构类似：

```json
{
  "binary_score": "yes",
  "confidence": 0.83,
  "relevant_doc_indices": [1, 3],
  "reason": "文档包含用户询问菜品的主要食材和步骤"
}
```

含义：

- `binary_score=yes`：检索结果足够相关，可以直接进入父文档扩展和生成回答
- `binary_score=no`：检索结果不足，需要触发查询扩展和二次检索
- `confidence`：0 到 1 的置信度
- `relevant_doc_indices`：相关文档编号
- `reason`：评分原因

RAG pipeline 的触发条件：

```text
binary_score=no
或 confidence < rewrite_confidence_threshold
```

默认阈值是：

```python
rewrite_confidence_threshold = 0.55
```

## Step-Back / HyDE / Complex 查询扩展

当 `grade_documents` 判断初检不足时，会进入重写路由：

```text
retrieval_rewrite_router()
  ↓
选择 step_back / hyde / complex
```

### Step-Back

Step-Back 适合：

- 技巧类问题
- 火候/时间类问题
- 食材替换
- 做法失败原因
- 模糊追问

实现函数：

```python
step_back_expand(query)
```

它会生成：

- `step_back_question`：更上位的退步问题
- `step_back_answer`：对退步问题的背景回答
- `expanded_query`：原问题 + 退步问题 + 背景答案

然后用 `expanded_query` 重新检索。

### HyDE

HyDE 适合：

- 推荐类问题
- 多约束场景
- 宽泛需求
- 初检完全跑偏或命中弱

实现函数：

```python
generate_hypothetical_document(query)
```

它会生成一段可能出现在食谱知识库里的假想文档片段。例如用户问：

```text
只有鸡蛋和青菜，20分钟内能做什么？
```

HyDE 会生成包含可能菜名、食材、步骤、时间和口味关键词的短文本，然后用这段文本去检索。

### Complex

`complex` 表示同时使用 Step-Back 和 HyDE：

```text
step_back_expand()
  ↓
用 expanded_query 检索

generate_hypothetical_document()
  ↓
用 hypothetical_document 检索

两路结果合并、去重
```

合并逻辑在 `RecipeRAGSystem._merge_unique_chunks()`，按 `source + page_content hash` 去重，并保留前 `top_k` 个结果。

## RAG Trace 字段

当前 `tool_trace` 会返回这些关键信息：

- `tool_name`
- `tool_query`
- `tool_used`
- `hit`
- `route_type`
- `rewritten_query`
- `retrieval_grade`
- `rewrite_triggered`
- `rewrite_strategy`
- `rewrite_reason`
- `expanded_query`
- `step_back_question`
- `step_back_answer`
- `hypothetical_document`
- `retrieval_attempts`
- `filters`
- `retrieval_mode`
- `rerank_applied`
- `elapsed_ms`
- `retrieved_chunk_count`
- `retrieved_chunk_summaries`
- `retrieved_parent_count`
- `retrieved_dish_names`
- `retrieved_sources`
- `retrieved_docs`

示例 trace 片段：

```json
{
  "tool_name": "recipe_rag_search",
  "tool_query": "只有鸡蛋和青菜，20分钟内能做什么？",
  "route_type": "list",
  "retrieval_grade": {
    "binary_score": "no",
    "confidence": 0.42,
    "reason": "初检文档只匹配鸡蛋，没有覆盖青菜和时间约束"
  },
  "rewrite_triggered": true,
  "rewrite_strategy": "complex",
  "expanded_query": "只有鸡蛋和青菜，20分钟内能做什么？...",
  "hypothetical_document": "鸡蛋青菜快手菜可选择青菜炒蛋、鸡蛋青菜汤...",
  "retrieval_mode": "hybrid+complex"
}
```

## 会话记忆与摘要压缩

FastAPI 后端用内存字典保存 session：

```python
sessions: dict[str, list[ChatMessage]]
```

当 session 消息超过 `MAX_SESSION_MESSAGES = 50` 时，会把前 `SUMMARY_COMPACT_COUNT = 40` 条压缩成一条 system summary。

摘要保留：

- 用户偏好
- 已确认事实
- 项目背景
- 仍未解决的问题
- 后续待办

每条 assistant 消息可以附带本轮 `rag_trace`，方便后续调试。

## API

### 健康检查

```text
GET /api/health
```

### 数据流说明

```text
GET /api/flow
```

### 普通对话

```text
POST /api/chat
```

请求：

```json
{
  "message": "番茄炒蛋怎么做？",
  "session_id": null,
  "include_trace": true
}
```

### 流式对话

```text
POST /api/chat/stream
```

返回 SSE：

```text
data: {"type":"agent_step","step":{...}}

data: {"type":"rag_step","step":{...}}

data: {"type":"token","content":"..."}

data: {"type":"final","response":{...}}

data: {"type":"done"}
```

## 本地运行

建议使用 Python 3.10 到 3.12。部分依赖对 Python 3.13 的支持不稳定，尤其是 `langchain-unstructured` 间接依赖的 `onnxruntime`。

安装依赖：

```bash
cd code
python -m pip install -r requirements.txt
```

配置模型 API Key：

```bash
export QWEN_API_KEY="your-api-key"
```

启动服务：

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

打开浏览器访问：

```text
http://127.0.0.1:8000/
```

## 测试建议

### 1. 寒暄，不应调用工具

```text
你好
```

预期：

- Agent 直接回答
- `tool_trace` 为空

### 2. 明确菜谱问题，应调用工具

```text
番茄炒蛋怎么做？
```

预期：

- Agent 调用 `recipe_rag_search`
- RAG trace 展示 route、rewrite、retrieve、grade
- 如果初检足够相关，不触发二次检索

### 3. 模糊推荐问题，可能触发 HyDE

```text
我想吃点清淡的，但不要太麻烦，有什么推荐？
```

预期：

- `grade_documents` 可能给出 `binary_score=no`
- `rewrite_strategy` 可能为 `hyde` 或 `complex`
- `hypothetical_document` 有内容

### 4. 技巧/火候问题，可能触发 Step-Back

```text
这个怎么做才不会老？
```

预期：

- 多轮上下文足够时 Agent 会结合历史
- RAG 内部可能选择 `step_back`
- trace 中出现 `step_back_question` 和 `step_back_answer`

### 5. 多约束问题，可能触发 complex

```text
只有鸡蛋和青菜，20分钟内能做什么？
```

预期：

- 如果初检不足，重写路由可能选择 `complex`
- Step-Back 和 HyDE 两路检索结果合并
- `retrieval_attempts` 中出现两条 expanded 检索记录

也可以直接用 curl 测 SSE：

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"只有鸡蛋和青菜，20分钟内能做什么？","include_trace":true}'
```

## Git 回滚点

项目升级过程中已保存两个本地回滚点：

```text
4f0ab00 Save event-stream agent version
7dec1b4 Add LangGraph recipe agent
```

当前工作区在 `7dec1b4` 之后继续修改了 RAG 查询扩展和评分门控。

## 后续优化方向

- 将 RAG pipeline 拆成独立 `rag_pipeline.py`，让节点结构更接近 LangGraph 风格
- 为 `grade_documents`、Step-Back、HyDE 构建 eval dataset
- 增加更多工具，例如购物清单、菜单规划、食材替换
- 将 session 从内存迁移到 SQLite / Postgres
- 将 RAG trace 持久化，便于离线评估
- 为前端增加更清晰的 timeline 视图
