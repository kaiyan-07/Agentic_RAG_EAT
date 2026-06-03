"""
生成集成模块
"""

import json
import os
import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""
    
    def __init__(
        self,
        model_name: str = "qwen-max",
        api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env: str = "QWEN_API_KEY",
        thinking_type: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        """
        初始化生成集成模块
        
        Args:
            model_name: 模型名称
            api_base: OpenAI兼容接口地址
            api_key_env: API Key环境变量名
            thinking_type: 推理模式开关，留空则不传
            temperature: 生成温度
            max_tokens: 最大token数
        """
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.api_key_env = api_key_env
        self.thinking_type = thinking_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()
    
    def _resolve_api_key(self) -> str:
        """解析当前应使用的API Key，并兼容旧环境变量。"""
        candidate_envs = [
            self.api_key_env,
            "DEEPSEEK_API_KEY",
            "AIHUBMIX_API_KEY",
            "MOONSHOT_API_KEY",
            "QWEN_API_KEY",
            "API_KEY",
        ]
        for env_name in candidate_envs:
            api_key = os.getenv(env_name)
            if api_key:
                if env_name != self.api_key_env:
                    logger.warning(f"未找到 {self.api_key_env}，回退使用 {env_name}")
                return api_key
        raise ValueError(
            f"请设置 {self.api_key_env} 环境变量，"
            "或提供兼容的 DEEPSEEK_API_KEY / AIHUBMIX_API_KEY / MOONSHOT_API_KEY / API_KEY"
        )

    def setup_llm(self):
        """初始化大语言模型"""
        logger.info(f"正在初始化LLM: {self.model_name} @ {self.api_base}")

        api_key = self._resolve_api_key()
        model_kwargs = {}
        if self.thinking_type:
            model_kwargs["thinking"] = {"type": self.thinking_type}

        self.llm = ChatOpenAI(
            model_name=self.model_name,
            openai_api_base=self.api_base,
            openai_api_key=api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model_kwargs=model_kwargs,
        )
        
        logger.info("LLM初始化完成")
    
    def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成基础回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            生成的回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请直接回答用户问题，并遵循以下规则：
- 只在用户询问完整做法、步骤或教程时，才展开制作流程
- 如果用户只询问食材、工具、用量、时间、温度、火候或技巧，只回答对应信息
- 不要为了显得完整而输出无关的菜品介绍、完整教程或额外背景
- 可以使用简短列表，让答案清楚可读
- 如果信息不足，请诚实说明，不要补充食谱中没有的信息

回答:""")

        # 使用LCEL构建链
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response
    
    def generate_step_by_step_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成分步骤回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            分步骤的详细回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。优先使用原文中的实用技巧，如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容或重复制作步骤中的信息
- 重点突出实用性和可操作性
- 如果没有额外的技巧要分享，可以省略制作技巧部分

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response

    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 - 让大模型判断是否需要重写查询

        Args:
            query: 原始查询

        Returns:
            重写后的查询或原查询
        """
        prompt = PromptTemplate(
            template="""
你是一个智能查询分析助手。请分析用户的查询，判断是否需要重写以提高食谱搜索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询**（直接返回原查询）：
   - 包含具体菜品名称：如"宫保鸡丁怎么做"、"红烧肉的制作方法"
   - 明确的制作询问：如"蛋炒饭需要什么食材"、"糖醋排骨的步骤"
   - 具体的烹饪技巧：如"如何炒菜不粘锅"、"怎样调制糖醋汁"

2. **模糊不清的查询**（需要重写）：
   - 过于宽泛：如"做菜"、"有什么好吃的"、"推荐个菜"
   - 缺乏具体信息：如"川菜"、"素菜"、"简单的"
   - 口语化表达：如"想吃点什么"、"有饮品推荐吗"

重写原则：
- 保持原意不变
- 增加相关烹饪术语
- 优先推荐简单易做的
- 保持简洁性

示例：
- "做菜" → "简单易做的家常菜谱"
- "有饮品推荐吗" → "简单饮品制作方法"
- "推荐个菜" → "简单家常菜推荐"
- "川菜" → "经典川菜菜谱"
- "宫保鸡丁怎么做" → "宫保鸡丁怎么做"（保持原查询）
- "红烧肉需要什么食材" → "红烧肉需要什么食材"（保持原查询）

请输出最终查询（如果不需要重写就返回原查询）:""",
            input_variables=["query"]
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query).strip()

        # 记录重写结果
        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response



    def query_router(self, query: str) -> str:
        """
        查询路由 - 根据查询类型选择不同的处理方式

        Args:
            query: 用户查询

        Returns:
            路由类型 ('list', 'ingredients', 'steps', 'tips', 'time', 'general')
        """
        prompt = ChatPromptTemplate.from_template("""
根据用户的问题，将其分类为以下六种类型之一：

1. 'list' - 用户想要获取菜品列表或推荐，只需要菜名
   例如：推荐几个素菜、有什么川菜、给我3个简单的菜

2. 'ingredients' - 用户询问食材、原料、工具、配料、调料、用量
   例如：需要什么食材、有哪些必备原料和工具、两只鸡蛋要多少水和盐

3. 'steps' - 用户询问完整做法、制作步骤、流程、教程
   例如：宫保鸡丁怎么做、制作步骤是什么、完整流程

4. 'tips' - 用户询问技巧、注意事项、怎么避免失败
   例如：怎么做才嫩、怎么不粘锅、有什么注意事项

5. 'time' - 用户询问时间、温度、火候、烤多久、蒸多久
   例如：空气炸锅多少度多久、蒸几分钟、火候怎么控制

6. 'general' - 其他一般性问题
   例如：什么是川菜、制作技巧、营养价值

请只返回分类结果：list、ingredients、steps、tips、time 或 general

用户问题: {query}

分类结果:""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(query).strip().lower()

        # 确保返回有效的路由类型
        valid_routes = ['list', 'ingredients', 'steps', 'tips', 'time', 'general']
        if result in valid_routes:
            return result
        else:
            return 'general'  # 默认类型

    def retrieval_rewrite_router(self, query: str, route_type: str, grade: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        判断二次检索应使用 Step-Back 还是 HyDE。

        Returns:
            {
                "strategy": "step_back" | "hyde" | "complex",
                "reason": "...",
            }
        """
        grade_text = json.dumps(grade or {}, ensure_ascii=False)
        prompt = PromptTemplate(
            template="""
你是食谱 RAG 查询重写路由器。请在 Step-Back 与 HyDE 中选择一种二次检索策略。

策略说明：
- step_back：把具体问题抽象成更上位的检索问题，适合技巧、火候、替换、失败原因、模糊追问。
- hyde：生成一段理想答案/假想食谱文档来检索，适合推荐、宽泛需求、多约束场景、初次检索命中弱。
- complex：同时使用 step_back 和 hyde，适合多约束、需要综合场景、初检明显跑偏的问题。

用户问题：{query}
问题类型：{route_type}
初检评分：{grade}

只输出 JSON：
{{
  "strategy": "step_back" 或 "hyde" 或 "complex",
  "reason": "一句话原因"
}}
""",
            input_variables=["query", "route_type", "grade"],
        )
        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke({"query": query, "route_type": route_type, "grade": grade_text}).strip()
        data = self._parse_json_object(raw) or {}
        strategy = str(data.get("strategy") or "none").lower()
        strategy = strategy.replace("-", "_")
        if strategy not in {"step_back", "hyde", "complex"}:
            strategy = "step_back"
        return {
            "strategy": strategy,
            "reason": str(data.get("reason") or ""),
        }

    def step_back_expand(self, query: str) -> Dict[str, str]:
        """生成 Step-Back 退步问题、背景答案和融合后的扩展查询。"""
        step_back_question = self._generate_step_back_question(query)
        step_back_answer = self._answer_step_back_question(step_back_question)
        expanded_query = query
        if step_back_question or step_back_answer:
            expanded_query = (
                f"{query}\n\n"
                f"退步问题：{step_back_question}\n"
                f"退步问题答案：{step_back_answer}"
            )
        return {
            "step_back_question": step_back_question,
            "step_back_answer": step_back_answer,
            "expanded_query": expanded_query,
        }

    def _generate_step_back_question(self, query: str) -> str:
        """生成 Step-Back 抽象问题。"""
        prompt = PromptTemplate(
            template="""
你是食谱检索查询改写器。请把用户问题抽象成更上位、更容易补充背景知识的 Step-Back 退步问题。

要求：
- 保留烹饪目标或约束
- 抽象掉过细的菜名、口语化表达或局部条件
- 输出一句中文问题
- 不要解释

用户问题：{query}

Step-Back 退步问题：
""",
            input_variables=["query"],
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query}).strip()

    def _answer_step_back_question(self, step_back_question: str) -> str:
        """回答 Step-Back 退步问题，作为检索背景知识。"""
        if not step_back_question:
            return ""
        prompt = PromptTemplate(
            template="""
请简要回答以下食谱/烹饪退步问题，提供通用原理或背景知识。

要求：
- 控制在 120 字以内
- 只输出答案，不要解释你的推理过程
- 适合拼接进检索查询

退步问题：{query}

答案：
""",
            input_variables=["query"],
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": step_back_question}).strip()

    def generate_hypothetical_document(self, query: str) -> str:
        """生成 HyDE 假想食谱答案，用作检索查询。"""
        prompt = PromptTemplate(
            template="""
你是食谱 RAG 的 HyDE 查询生成器。请基于用户问题，写一段可能出现在食谱知识库中的假想答案/食谱片段。

要求：
- 使用食谱文档常见措辞
- 包含可能的菜名、食材、步骤、时间、火候或口味关键词
- 不要声称这是事实，只生成用于检索的文本
- 控制在 120 字以内

用户问题：{query}

HyDE 检索文本：
""",
            input_variables=["query"],
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query}).strip()

    def generate_step_back_query(self, query: str) -> str:
        """兼容旧调用：返回 Step-Back 扩展查询。"""
        return self.step_back_expand(query)["expanded_query"]

    def generate_hyde_query(self, query: str) -> str:
        """兼容旧调用：返回 HyDE 假想文档。"""
        return self.generate_hypothetical_document(query)

    def grade_documents(self, query: str, context_docs: List[Document]) -> Dict[str, Any]:
        """
        结构化评估检索文档是否足以回答用户问题。
        """
        if not context_docs:
            return {
                "relevant": False,
                "needs_rewrite": True,
                "confidence": 0.0,
                "relevant_doc_indices": [],
                "reason": "没有检索到文档",
            }

        context = self._build_grading_context(context_docs)
        prompt = PromptTemplate(
            template="""
你是 RAG 检索结果评估器。请判断这些食谱文档是否足以回答用户问题。

评分规则：
- binary_score=yes：至少有一个文档和问题强相关，且足以支撑回答。
- binary_score=no：文档明显不相关、过少、只弱相关，或无法支撑回答。
- confidence：0 到 1。
- relevant_doc_indices：相关文档编号，从 1 开始。

用户问题：
{query}

候选文档：
{context}

只输出 JSON：
{{
  "binary_score": "yes",
  "confidence": 0.83,
  "relevant_doc_indices": [1, 3],
  "reason": "一句话说明"
}}
""",
            input_variables=["query", "context"],
        )
        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke({"query": query, "context": context}).strip()
        data = self._parse_json_object(raw) or {}

        confidence = data.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        relevant_doc_indices = data.get("relevant_doc_indices") or []
        if not isinstance(relevant_doc_indices, list):
            relevant_doc_indices = []

        binary_score = str(data.get("binary_score") or "").strip().lower()
        if binary_score not in {"yes", "no"}:
            relevant = self._coerce_bool(data.get("relevant"))
            needs_rewrite = self._coerce_bool(data.get("needs_rewrite"))
            binary_score = "yes" if relevant and not needs_rewrite else "no"

        return {
            "binary_score": binary_score,
            "relevant": binary_score == "yes",
            "needs_rewrite": binary_score != "yes",
            "confidence": max(0.0, min(1.0, confidence)),
            "relevant_doc_indices": relevant_doc_indices,
            "reason": str(data.get("reason") or ""),
        }

    def _build_grading_context(self, docs: List[Document], max_doc_chars: int = 700) -> str:
        parts = []
        for index, doc in enumerate(docs, 1):
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            source = doc.metadata.get("source", "")
            content = doc.page_content[:max_doc_chars]
            parts.append(f"【文档 {index}】{dish_name}\n来源: {source}\n内容: {content}")
        return "\n\n".join(parts)

    def _parse_json_object(self, text: str) -> Dict[str, Any] | None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            logger.warning("无法解析结构化输出: %s", text)
            return None

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1", "是", "相关", "需要"}
        return bool(value)

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成列表式回答 - 适用于推荐类查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            列表式回答
        """
        if not context_docs:
            return "抱歉，没有找到相关的菜品信息。"

        # 提取菜品名称
        dish_names = []
        for doc in context_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in dish_names:
                dish_names.append(dish_name)

        # 构建简洁的列表回答
        if len(dish_names) == 1:
            return f"为您推荐：{dish_names[0]}"
        elif len(dish_names) <= 3:
            return f"为您推荐以下菜品：\n" + "\n".join([f"{i+1}. {name}" for i, name in enumerate(dish_names)])
        else:
            return f"为您推荐以下菜品：\n" + "\n".join([f"{i+1}. {name}" for i, name in enumerate(dish_names[:3])]) + f"\n\n还有其他 {len(dish_names)-3} 道菜品可供选择。"

    def generate_basic_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成基础回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            生成的回答片段
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请直接回答用户问题，并遵循以下规则：
- 只在用户询问完整做法、步骤或教程时，才展开制作流程
- 如果用户只询问食材、工具、用量、时间、温度、火候或技巧，只回答对应信息
- 不要为了显得完整而输出无关的菜品介绍、完整教程或额外背景
- 可以使用简短列表，让答案清楚可读
- 如果信息不足，请诚实说明，不要补充食谱中没有的信息

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def generate_step_by_step_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成详细步骤回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            详细步骤回答片段
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容
- 重点突出实用性和可操作性

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def _build_context(self, docs: List[Document], max_length: int = 2000) -> str:
        """
        构建上下文字符串
        
        Args:
            docs: 文档列表
            max_length: 最大长度
            
        Returns:
            格式化的上下文字符串
        """
        if not docs:
            return "暂无相关食谱信息。"
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(docs, 1):
            # 添加元数据信息
            metadata_info = f"【食谱 {i}】"
            if 'dish_name' in doc.metadata:
                metadata_info += f" {doc.metadata['dish_name']}"
            if 'category' in doc.metadata:
                metadata_info += f" | 分类: {doc.metadata['category']}"
            if 'difficulty' in doc.metadata:
                metadata_info += f" | 难度: {doc.metadata['difficulty']}"
            
            # 构建文档文本
            doc_text = f"{metadata_info}\n{doc.page_content}\n"
            
            # 检查长度限制
            if current_length + len(doc_text) > max_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n" + "="*50 + "\n".join(context_parts)
