# EAT_RAG

EAT_RAG 是一个面向食谱知识库的 RAG 项目。当前版本在已有食谱检索与生成流程之上增加了一层轻量 Agent，让系统不只是直接问答，而是先判断问题、决定是否调用工具、再把检索和生成过程返回给前端。

## 当前功能

- 基于本地食谱数据构建向量索引
- 支持食谱问答、做法查询、食材用量、火候时间和菜品推荐
- 提供 FastAPI 后端接口
- 提供简单 Web UI，用于对话和查看 Agent / RAG trace
- 保留原有 RAG 流程，包括查询路由、查询改写、混合检索、父文档扩展和生成回答

## Agent 工作流

当前 Agent 层作为 RAG 系统外侧的编排层运行，主要负责对话流转和工具调用：

```text
用户输入
  ↓
FastAPI 接收请求并读取会话历史
  ↓
RecipeAgent 判断问题是否与食谱相关
  ↓
需要食谱知识时调用 recipe_rag_search 工具
  ↓
RecipeRAGTool 调用现有 RecipeRAGSystem.ask_with_trace()
  ↓
RAG 执行查询路由、查询改写、混合检索、父文档扩展和答案生成
  ↓
Agent 整合工具结果、步骤信息和 trace
  ↓
前端展示最终回答与调用过程
```

这一版的 Agent 设计目标是尽量不改动原有 RAG 内核，把 RAG 包装成可调用工具。后续会继续优化 Agent 的判断能力、多轮上下文理解、工具选择、错误恢复，以及更清晰的推理 / 检索过程展示。

## 项目结构

```text
.
├── code/
│   ├── backend/          # FastAPI + Agent 工具调用层
│   ├── frontend/         # 简单 Web UI
│   ├── rag_modules/      # RAG 数据、索引、检索和生成模块
│   ├── main.py           # RAG 系统主入口
│   ├── config.py         # 默认配置
│   └── requirements.txt
└── data/
    └── cook/             # 食谱知识库数据
```

## 本地运行

安装依赖：

```bash
cd code
pip install -r requirements.txt
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

## 本地产物

本地运行会生成一些不需要进入仓库的文件，例如：

- `code/vector_index/`：本地生成的 FAISS 向量索引
- `code/eval_outputs/`：RAGAS 等评估运行结果
- `.env` / `.env.*`：本地环境变量和 API Key
- Python 缓存、编辑器配置、系统临时文件和日志

这些内容已经通过 `.gitignore` 忽略。
