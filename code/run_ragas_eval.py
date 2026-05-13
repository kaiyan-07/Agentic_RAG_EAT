"""
Run RAGAS evaluation for the recipe RAG final answers.

This script intentionally stays outside the main RAG pipeline. It:
1. loads a small human-written evaluation dataset,
2. runs the existing RAG system for each question,
3. evaluates only final-answer metrics:
   - faithfulness
   - answer/response relevancy
   - answer correctness

Retrieval hit-rate metrics are intentionally not included here.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from dotenv import load_dotenv

from config import DEFAULT_CONFIG, RAGConfig


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = SCRIPT_DIR / "eval_datasets" / "ragas_recipe_eval_50.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "eval_outputs"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_config(args: argparse.Namespace) -> RAGConfig:
    config = RAGConfig.from_dict(DEFAULT_CONFIG.to_dict())
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.llm_model:
        config.llm_model = args.llm_model
    if args.llm_api_base:
        config.llm_api_base = args.llm_api_base
    if args.llm_api_key_env:
        config.llm_api_key_env = args.llm_api_key_env
    if args.embedding_model:
        config.embedding_model = args.embedding_model
    return config


def collect_rag_outputs(
    samples: List[Dict[str, Any]],
    config: RAGConfig,
) -> List[Dict[str, Any]]:
    from main import RecipeRAGSystem

    rag = RecipeRAGSystem(config)
    rag.initialize_system()
    rag.build_knowledge_base()

    run_records = []
    total = len(samples)
    for index, sample in enumerate(samples, 1):
        question = sample["user_input"]
        print(f"[{index}/{total}] Running RAG: {sample.get('id', question)}")
        trace = rag.ask_with_trace(
            question,
            stream=False,
            include_raw_documents=False,
            pipeline_options={
                "enable_router": True,
                "enable_rewrite": True,
            },
        )

        parent_contexts = trace.get("retrieved_parent_contexts") or []
        run_records.append(
            {
                **sample,
                "response": trace["response"],
                "retrieved_contexts": parent_contexts,
                "route_type": trace.get("route_type"),
                "rewritten_query": trace.get("rewritten_query"),
                "retrieved_parent_sources": trace.get("retrieved_parent_sources", []),
                "retrieved_parent_dish_names": trace.get("retrieved_parent_dish_names", []),
            }
        )
    return run_records


def trim_text(text: str, max_chars: int | None) -> str:
    if not max_chars or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def trim_contexts(contexts: List[str], max_total_chars: int | None) -> List[str]:
    if not max_total_chars:
        return contexts

    trimmed_contexts = []
    used_chars = 0
    for context in contexts:
        remaining = max_total_chars - used_chars
        if remaining <= 0:
            break
        trimmed = trim_text(context, remaining)
        trimmed_contexts.append(trimmed)
        used_chars += len(trimmed)
    return trimmed_contexts


def to_ragas_records(
    run_records: List[Dict[str, Any]],
    max_context_chars: int | None = None,
    max_response_chars: int | None = None,
) -> List[Dict[str, Any]]:
    ragas_records = []
    for record in run_records:
        ragas_records.append(
            {
                "user_input": record["user_input"],
                "retrieved_contexts": trim_contexts(
                    record.get("retrieved_contexts", []),
                    max_context_chars,
                ),
                "response": trim_text(record["response"], max_response_chars),
                "reference": record["reference"],
            }
        )
    return ragas_records


def build_ragas_backends(config: RAGConfig, judge_max_tokens: int) -> Tuple[Any, Any]:
    from langchain_community.chat_models import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    api_key = (
        os.getenv(config.llm_api_key_env)
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("AIHUBMIX_API_KEY")
        or os.getenv("MOONSHOT_API_KEY")
        or os.getenv("API_KEY")
    )
    if not api_key:
        raise ValueError(
            f"Please set {config.llm_api_key_env} or a compatible API key env var "
            "before running RAGAS evaluation."
        )

    llm_kwargs: Dict[str, Any] = {}
    if config.llm_thinking_type:
        llm_kwargs["model_kwargs"] = {"thinking": {"type": config.llm_thinking_type}}

    judge_llm = ChatOpenAI(
        model_name=config.llm_model,
        openai_api_base=config.llm_api_base,
        openai_api_key=api_key,
        temperature=0,
        max_tokens=judge_max_tokens,
        **llm_kwargs,
    )
    judge_embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)

    return LangchainLLMWrapper(judge_llm), LangchainEmbeddingsWrapper(judge_embeddings)


def build_modern_metrics(ragas_llm: Any, ragas_embeddings: Any) -> List[Any]:
    from ragas.metrics import AnswerCorrectness, Faithfulness

    try:
        from ragas.metrics import ResponseRelevancy as RelevancyMetric
    except ImportError:
        from ragas.metrics import AnswerRelevancy as RelevancyMetric

    metric_specs = [
        (Faithfulness, {"llm": ragas_llm}),
        (RelevancyMetric, {"llm": ragas_llm, "embeddings": ragas_embeddings}),
        (AnswerCorrectness, {"llm": ragas_llm, "embeddings": ragas_embeddings}),
    ]

    metrics = []
    for metric_cls, kwargs in metric_specs:
        try:
            metrics.append(metric_cls(**kwargs))
        except TypeError:
            metrics.append(metric_cls())
    return metrics


def evaluate_with_ragas(
    ragas_records: List[Dict[str, Any]],
    config: RAGConfig,
    judge_max_tokens: int,
):
    from ragas import evaluate

    ragas_llm, ragas_embeddings = build_ragas_backends(config, judge_max_tokens)

    try:
        from ragas import EvaluationDataset

        dataset = EvaluationDataset.from_list(ragas_records)
        metrics = build_modern_metrics(ragas_llm, ragas_embeddings)
        try:
            return evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
            )
        except TypeError:
            return evaluate(dataset=dataset, metrics=metrics)
    except ImportError:
        from datasets import Dataset
        from ragas.metrics import answer_correctness, answer_relevancy, faithfulness

        legacy_records = {
            "question": [r["user_input"] for r in ragas_records],
            "contexts": [r["retrieved_contexts"] for r in ragas_records],
            "answer": [r["response"] for r in ragas_records],
            "ground_truth": [r["reference"] for r in ragas_records],
        }
        dataset = Dataset.from_dict(legacy_records)
        return evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, answer_correctness],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )


def save_ragas_result(result: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df = result.to_pandas()
        df.to_csv(path.with_suffix(".csv"), index=False)
        path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
        return
    except Exception:
        pass

    try:
        data = result.to_dict()
    except Exception:
        data = {"result": str(result)}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS final-answer evaluation for RecipeRAG.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples.")
    parser.add_argument(
        "--category",
        choices=["fact", "step", "recommendation"],
        action="append",
        help="Filter by category. Can be provided multiple times.",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-base", default=None)
    parser.add_argument("--llm-api-key-env", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=8192,
        help="Max tokens for the RAGAS judge LLM. Increase this if metrics become nan with LLMDidNotFinishException.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=6000,
        help="Max total characters of retrieved_contexts passed to RAGAS per sample. Use 0 to disable trimming.",
    )
    parser.add_argument(
        "--max-response-chars",
        type=int,
        default=4000,
        help="Max characters of response passed to RAGAS per sample. Use 0 to disable trimming.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only run RAG and save records; skip RAGAS scoring.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = build_config(args)

    samples = read_jsonl(args.dataset)
    if args.category:
        wanted = set(args.category)
        samples = [sample for sample in samples if sample.get("category") in wanted]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("No evaluation samples selected.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_records_path = args.output_dir / f"ragas_run_records_{timestamp}.jsonl"
    ragas_input_path = args.output_dir / f"ragas_input_{timestamp}.jsonl"
    ragas_result_path = args.output_dir / f"ragas_result_{timestamp}.json"

    run_records = collect_rag_outputs(samples, config)
    write_jsonl(run_records_path, run_records)
    print(f"Saved RAG run records: {run_records_path}")

    ragas_records = to_ragas_records(
        run_records,
        max_context_chars=args.max_context_chars or None,
        max_response_chars=args.max_response_chars or None,
    )
    write_jsonl(ragas_input_path, ragas_records)
    print(f"Saved RAGAS input records: {ragas_input_path}")

    if args.collect_only:
        print("Skipped RAGAS scoring because --collect-only was set.")
        return

    result = evaluate_with_ragas(ragas_records, config, args.judge_max_tokens)
    save_ragas_result(result, ragas_result_path)
    print(f"Saved RAGAS result: {ragas_result_path}")
    print(result)


if __name__ == "__main__":
    main()
