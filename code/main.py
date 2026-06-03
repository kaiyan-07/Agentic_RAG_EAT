"""
RAG系统主程序
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecipeRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self._module_dir = Path(__file__).parent.resolve()
        self.config.data_path = str(self._resolve_project_path(self.config.data_path))
        self.config.index_save_path = str(self._resolve_project_path(self.config.index_save_path))
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查LLM API密钥，并兼容旧环境变量命名
        llm_key_env = self.config.llm_api_key_env
        llm_api_key = (
            os.getenv(llm_key_env)
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("AIHUBMIX_API_KEY")
            or os.getenv("MOONSHOT_API_KEY")
            or os.getenv("API_KEY")
        )
        if not llm_api_key:
            raise ValueError(f"请设置 {llm_key_env} 环境变量")
        if not os.getenv(llm_key_env):
            os.environ[llm_key_env] = llm_api_key

    def _resolve_project_path(self, path_value: str) -> Path:
        """
        解析配置中的相对路径，优先兼容当前工作目录，其次回退到 code 目录。
        """
        path = Path(path_value)
        if path.is_absolute():
            return path

        cwd_candidate = path.resolve()
        if cwd_candidate.exists():
            return cwd_candidate

        return (self._module_dir / path).resolve()
    
    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            api_base=self.config.llm_api_base,
            api_key_env=self.config.llm_api_key_env,
            thinking_type=self.config.llm_thinking_type,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        print("✅ 系统初始化完成！")
    
    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        # 1. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            print("✅ 成功加载已保存的向量索引！")
            # 仍需要加载文档和分块用于检索模块
            print("加载食谱文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
        else:
            print("未找到已保存的索引，开始构建新索引...")

            # 2. 加载文档
            print("加载食谱文档...")
            self.data_module.load_documents()

            # 3. 文本分块
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            # 4. 构建向量索引
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            # 5. 保存索引
            print("保存向量索引...")
            self.index_module.save_index()

        # 6. 初始化检索优化模块
        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        # 7. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("✅ 知识库构建完成！")

    def _log_progress(self, message: str, verbose: bool):
        """按需打印流程日志。"""
        if verbose:
            print(message)

    def _summarize_chunks(self, chunks: List) -> List[str]:
        """生成检索到的子块摘要，便于调试排查。"""
        chunk_info = []
        for chunk in chunks:
            dish_name = chunk.metadata.get('dish_name', '未知菜品')
            content_preview = chunk.page_content[:100].strip()
            if content_preview.startswith('#'):
                title_end = content_preview.find('\n') if '\n' in content_preview else len(content_preview)
                section_title = content_preview[:title_end].replace('#', '').strip()
                chunk_info.append(f"{dish_name}({section_title})")
            else:
                chunk_info.append(f"{dish_name}(内容片段)")
        return chunk_info

    def _emit_rag_step(self, emitter, step_type: str, icon: str, label: str, detail: str) -> None:
        """向上层实时发送 RAG pipeline 步骤。"""
        if not emitter:
            return
        emitter(
            {
                "type": "rag_step",
                "step": {
                    "type": step_type,
                    "icon": icon,
                    "label": label,
                    "detail": detail,
                },
            }
        )

    def _retrieve_chunks(self, query: str, filters: Dict[str, Any], verbose: bool):
        """执行一次带可选元数据过滤的混合检索。"""
        if filters:
            self._log_progress(f"应用过滤条件: {filters}", verbose)
            return self.retrieval_module.metadata_filtered_search(
                query,
                filters,
                top_k=self.config.top_k
            )
        return self.retrieval_module.hybrid_search(query, top_k=self.config.top_k)

    def _merge_unique_chunks(self, chunks: List) -> List:
        """按 source + content 去重，并保留前 top_k 个结果。"""
        seen = set()
        merged = []
        for chunk in chunks:
            key = (chunk.metadata.get("source", ""), hash(chunk.page_content))
            if key in seen:
                continue
            seen.add(key)
            merged.append(chunk)
            if len(merged) >= self.config.top_k:
                break
        return merged

    def _build_retrieval_attempt(self, stage: str, query: str, strategy: str, chunks: List) -> Dict[str, Any]:
        return {
            "stage": stage,
            "query": query,
            "strategy": strategy,
            "chunk_count": len(chunks),
            "chunk_summaries": self._summarize_chunks(chunks),
        }

    def _run_query_pipeline(
        self,
        question: str,
        stream: bool = False,
        verbose: bool = False,
        pipeline_options: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        运行完整问答流程，并返回可供调试复用的trace信息。

        Args:
            question: 用户问题
            stream: 是否使用流式输出
            verbose: 是否打印流程日志
            pipeline_options: 可选流程控制项

        Returns:
            包含检索和生成中间结果的trace字典
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        options = {
            "enable_router": True,
            "enable_rewrite": True,
            "enable_retrieval_grading": True,
            "enable_rewrite_retrieval": True,
            "rewrite_confidence_threshold": 0.55,
            "rag_step_emitter": None,
        }
        if pipeline_options:
            options.update(pipeline_options)
        emit_step = options.get("rag_step_emitter")

        self._log_progress(f"\n❓ 用户问题: {question}", verbose)

        if options["enable_router"]:
            route_type = self.generation_module.query_router(question)
        else:
            route_type = "steps"
        self._log_progress(f"🎯 查询类型: {route_type}", verbose)

        if route_type == 'list' or not options["enable_rewrite"]:
            rewritten_query = question
            if route_type == 'list':
                self._log_progress(f"📝 列表查询保持原样: {question}", verbose)
            else:
                self._log_progress(f"📝 已关闭查询重写，保持原样: {question}", verbose)
        else:
            self._log_progress("🤖 智能分析查询...", verbose)
            rewritten_query = self.generation_module.query_rewrite(question)

        self._emit_rag_step(emit_step, "retrieve_initial", "🔍", "正在检索知识库...", rewritten_query)
        self._log_progress("🔍 检索相关文档...", verbose)
        filters = self._extract_filters_from_query(question)
        relevant_chunks = self._retrieve_chunks(rewritten_query, filters, verbose)
        retrieval_attempts = [
            self._build_retrieval_attempt("initial", rewritten_query, "base", relevant_chunks)
        ]

        relevance_grade = {
            "relevant": bool(relevant_chunks),
            "needs_rewrite": not bool(relevant_chunks),
            "confidence": 1.0 if relevant_chunks else 0.0,
            "relevant_doc_indices": [],
            "reason": "未启用文档相关性评分",
        }
        rewrite_triggered = False
        rewrite_strategy = "none"
        rewrite_reason = ""
        expanded_query = ""
        step_back_question = ""
        step_back_answer = ""
        hypothetical_document = ""

        if options["enable_retrieval_grading"]:
            self._emit_rag_step(emit_step, "grade_documents", "🧪", "正在评估文档相关性...", rewritten_query)
            relevance_grade = self.generation_module.grade_documents(question, relevant_chunks)
            self._log_progress(
                f"🧪 相关性评分: relevant={relevance_grade.get('relevant')} "
                f"confidence={relevance_grade.get('confidence')}",
                verbose,
            )

        should_rewrite = (
            options["enable_rewrite_retrieval"]
            and (
                relevance_grade.get("needs_rewrite")
                or (relevance_grade.get("confidence") or 0.0) < options["rewrite_confidence_threshold"]
            )
        )

        if should_rewrite:
            rewrite_route = self.generation_module.retrieval_rewrite_router(
                question,
                route_type,
                relevance_grade,
            )
            rewrite_strategy = rewrite_route.get("strategy", "none")
            rewrite_reason = rewrite_route.get("reason", "")
            rewrite_triggered = True
            self._emit_rag_step(
                emit_step,
                "rewrite_question",
                "📝",
                "正在重写查询...",
                f"策略: {rewrite_strategy}；原因: {rewrite_reason}",
            )

            expanded_results = []
            if rewrite_strategy in {"step_back", "complex"}:
                step_back = self.generation_module.step_back_expand(question)
                step_back_question = step_back.get("step_back_question", "")
                step_back_answer = step_back.get("step_back_answer", "")
                expanded_query = step_back.get("expanded_query", "")
                self._emit_rag_step(
                    emit_step,
                    "retrieve_expanded",
                    "🔁",
                    "使用 Step-Back 扩展查询重新检索...",
                    expanded_query,
                )
                step_back_chunks = self._retrieve_chunks(expanded_query, filters, verbose)
                expanded_results.extend(step_back_chunks)
                retrieval_attempts.append(
                    self._build_retrieval_attempt("expanded", expanded_query, "step_back", step_back_chunks)
                )

            if rewrite_strategy in {"hyde", "complex"}:
                hypothetical_document = self.generation_module.generate_hypothetical_document(question)
                self._emit_rag_step(
                    emit_step,
                    "retrieve_expanded",
                    "🔁",
                    "使用 HyDE 假想文档重新检索...",
                    hypothetical_document,
                )
                hyde_chunks = self._retrieve_chunks(hypothetical_document, filters, verbose)
                expanded_results.extend(hyde_chunks)
                retrieval_attempts.append(
                    self._build_retrieval_attempt("expanded", hypothetical_document, "hyde", hyde_chunks)
                )

            expanded_chunks = self._merge_unique_chunks(expanded_results)
            if expanded_chunks:
                relevant_chunks = expanded_chunks

        chunk_summaries = self._summarize_chunks(relevant_chunks)
        if relevant_chunks:
            self._log_progress(
                f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_summaries)}",
                verbose
            )
        else:
            self._log_progress("找到 0 个相关文档块", verbose)

        trace: Dict[str, Any] = {
            "question": question,
            "route_type": route_type,
            "rewritten_query": rewritten_query,
            "retrieval_grade": relevance_grade,
            "rewrite_triggered": rewrite_triggered,
            "rewrite_strategy": rewrite_strategy,
            "rewrite_reason": rewrite_reason,
            "expanded_query": expanded_query,
            "step_back_question": step_back_question,
            "step_back_answer": step_back_answer,
            "hypothetical_document": hypothetical_document,
            "retrieval_attempts": retrieval_attempts,
            "filters": filters,
            "retrieved_chunk_count": len(relevant_chunks),
            "retrieved_chunk_summaries": chunk_summaries,
            "retrieved_chunk_contexts": [chunk.page_content for chunk in relevant_chunks],
            "retrieved_chunk_sources": [chunk.metadata.get("source", "") for chunk in relevant_chunks],
            "retrieved_chunk_dish_names": [chunk.metadata.get("dish_name", "未知菜品") for chunk in relevant_chunks],
            "retrieved_chunks": relevant_chunks,
            "retrieved_parent_count": 0,
            "retrieved_parent_contexts": [],
            "retrieved_parent_sources": [],
            "retrieved_parent_dish_names": [],
            "retrieved_parent_docs": [],
            "response": ""
        }

        if not relevant_chunks:
            trace["response"] = "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"
            return trace

        self._log_progress("获取完整文档...", verbose)
        relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
        doc_names = []
        for doc in relevant_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in doc_names:
                doc_names.append(dish_name)

        if doc_names:
            self._log_progress(f"找到文档: {', '.join(doc_names)}", verbose)
        else:
            self._log_progress(f"对应 {len(relevant_docs)} 个完整文档", verbose)

        trace.update({
            "retrieved_parent_count": len(relevant_docs),
            "retrieved_parent_contexts": [doc.page_content for doc in relevant_docs],
            "retrieved_parent_sources": [doc.metadata.get("source", "") for doc in relevant_docs],
            "retrieved_parent_dish_names": doc_names,
            "retrieved_parent_docs": relevant_docs
        })

        if route_type == 'list':
            self._log_progress("📋 生成菜品列表...", verbose)
            response = self.generation_module.generate_list_answer(question, relevant_docs)
        else:
            if route_type == "steps":
                self._log_progress("✍️ 生成详细步骤回答...", verbose)
                response = (
                    self.generation_module.generate_step_by_step_answer_stream(question, relevant_docs)
                    if stream else
                    self.generation_module.generate_step_by_step_answer(question, relevant_docs)
                )
            else:
                self._log_progress("✍️ 生成直接回答...", verbose)
                response = (
                    self.generation_module.generate_basic_answer_stream(question, relevant_docs)
                    if stream else
                    self.generation_module.generate_basic_answer(question, relevant_docs)
                )

        trace["response"] = response
        return trace
    
    def ask_question(self, question: str, stream: bool = False):
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出

        Returns:
            生成的回答或生成器
        """
        trace = self._run_query_pipeline(question, stream=stream, verbose=True)
        return trace["response"]

    def ask_with_trace(
        self,
        question: str,
        stream: bool = False,
        include_raw_documents: bool = False,
        pipeline_options: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        调试专用接口，返回结构化trace信息。

        Args:
            question: 用户问题
            stream: 是否使用流式输出
            include_raw_documents: 是否返回原始Document对象
            pipeline_options: 调试时的可选流程开关

        Returns:
            包含中间流程和最终回答的字典
        """
        trace = self._run_query_pipeline(
            question,
            stream=stream,
            verbose=False,
            pipeline_options=pipeline_options,
        )
        if not include_raw_documents:
            trace = dict(trace)
            trace.pop("retrieved_chunks", None)
            trace.pop("retrieved_parent_docs", None)
        return trace
    
    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户问题中提取元数据过滤条件
        """
        filters = {}
        # 分类关键词
        category_keywords = DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters['category'] = cat
                break

        # 难度关键词
        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters['difficulty'] = diff
                break

        return filters
    
    def search_by_category(self, category: str, query: str = "") -> List[str]:
        """
        按分类搜索菜品
        
        Args:
            category: 菜品分类
            query: 可选的额外查询条件
            
        Returns:
            菜品名称列表
        """
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")
        
        # 使用元数据过滤搜索
        search_query = query if query else category
        filters = {"category": category}
        
        docs = self.retrieval_module.metadata_filtered_search(search_query, filters, top_k=10)
        
        # 提取菜品名称
        dish_names = []
        for doc in docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in dish_names:
                dish_names.append(dish_name)
        
        return dish_names
    
    def get_ingredients_list(self, dish_name: str) -> str:
        """
        获取指定菜品的食材信息

        Args:
            dish_name: 菜品名称

        Returns:
            食材信息
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        # 搜索相关文档
        docs = self.retrieval_module.hybrid_search(dish_name, top_k=3)

        # 生成食材信息
        answer = self.generation_module.generate_basic_answer(f"{dish_name}需要什么食材？", docs)

        return answer
    
    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")
        
        # 初始化系统
        self.initialize_system()
        
        # 构建知识库
        self.build_knowledge_base()
        
        print("\n交互式问答 (输入'退出'结束):")
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break
                
                # 询问是否使用流式输出
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答:")
                if use_stream:
                    # 流式输出
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通输出
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
        
        print("\n感谢使用尝尝咸淡RAG系统！")



def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = RecipeRAGSystem()
        
        # 运行交互式问答
        rag_system.run_interactive()
        
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")

if __name__ == "__main__":
    main()
