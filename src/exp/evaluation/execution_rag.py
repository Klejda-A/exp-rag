"""Module for evaluation of QA datasets."""

import re
from pathlib import Path

import hydra
import litellm
import mlflow
import mlflow.data.pandas_dataset
import numpy as np
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from encourage.llm import BatchInferenceRunner
from encourage.prompts.context import Document
from encourage.prompts import PromptCollection
from encourage.prompts.meta_data import MetaData
from encourage.rag import HybridBM25RAG, BaseRAG, KnownContext, SummarizationContextRAG, SummarizationRAG, RerankerRAG, HydeRAG, HydeRerankerRAG, SelfRAG
from vllm import SamplingParams

from exp.evaluation.config import Config
from exp.evaluation.evaluation import main as evaluation
from exp.utils.file_manager import FileManager
from exp.utils.flatten_dict import flatten_dict
from exp.evaluation.query_rewriting import QueryRewritingRAG
from exp.evaluation.dataset_analysis import context_analysis, conversation_sequence
from transformers import AutoTokenizer

config_path = str((Path(__file__).parents[3] / "conf").resolve())
_escape_re = re.compile(r"\\[nrt]")
_quotes_re = re.compile(r'"([^"]*)"')
_whitespace_re = re.compile(r"\s+")


@hydra.main(version_base=None, config_path=config_path, config_name="defaults")
def main(cfg: Config) -> None:
    """Main function for execution of RAG methods"""
    # Load dataset from Huggingface
    load_dotenv(".env")

    qa_dataset = load_dataset(
        cfg.dataset.name, cfg.dataset.subset, split=cfg.dataset.split
    ).to_pandas()

    litellm._logging._disable_debugging()
    mlflow.openai.autolog()

    sampling_params = SamplingParams(
        temperature=cfg.model.temperature, max_tokens=cfg.model.max_tokens
    )
    runner = BatchInferenceRunner(sampling_params, cfg.model.model_name, base_url=cfg.base_url)
    sys_prompt = FileManager(cfg.dataset.sys_prompt_path).read()

    # Set RAG method
    rag_methods = ['known_context', 'base_implementation', 'hybrid_bm25', "summarization", 
                   'summarization_context', 'reranker_rag', 'hyde_rag', 'hyde_reranker_rag']
    
    rag_method_nr = 1
    rewrite_query = True

    ## Run the Inference
    mlflow.set_tracking_uri(cfg.mlflow.uri)
    mlflow.set_experiment(experiment_name=cfg.mlflow.experiment_id)

    with mlflow.start_run():
        # Log identifying parameters for mlflow
        mlflow.log_params(flatten_dict(cfg))
        mlflow.log_params({"dataset_size": len(qa_dataset)})
        mlflow.log_params({"rag_method": rag_methods[rag_method_nr]})
        mlflow.log_input(
            mlflow.data.pandas_dataset.from_pandas(qa_dataset, name=cfg.dataset.name),
            context="inference",
        )        

        with mlflow.start_span(name="root"):
            meta_datas, contexts = prepare_data(qa_dataset, cfg)

            # context_analysis(contexts, ["questions"], ["answers"], [], AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct"))

            # Set RAG parameters
            top_k = 5   # Contexts retrieved
            batch_size = 3000
            multi_turn_user = True  # Include conversation history in main prompt
            multi_turn_context = True   # Include conversation history in retrieval prompt

            prompts = [prepare_conversation_history(row, multi_turn_user) for row in qa_dataset["messages"]]
            conversation_sequence(prompts, cfg.dataset.name)

            match rag_method_nr:
                case 0:
                    rag = known_context_rag(contexts, cfg, top_k)
                case 1:
                    rag = base_implementation_rag(contexts, cfg, top_k, batch_size)
                case 2:
                    rag = hybrid_bm25_rag(contexts, cfg, top_k, 0.5, 0.5, batch_size)
                case 3:
                    rag = summarization_rag(contexts, cfg, top_k, runner)
                case 4:
                    rag = summarization_context_rag(contexts, cfg, top_k, runner)
                case 5:
                    rag = reranker_rag(contexts, cfg, top_k, runner, 3, batch_size)
                case 6:
                    rag = hyde_rag(contexts, cfg, top_k, runner, batch_size)
                case 7:
                    rag = hyde_reranker_rag(contexts, cfg, top_k, runner, 5, batch_size)

            user_prompts = [prepare_conversation_history(row, multi_turn_user) for row in qa_dataset["messages"]]
            if rewrite_query:
                user_prompts = query_rewriting_rag(cfg, runner, user_prompts)

            responses = rag.run(
                sys_prompt=sys_prompt,
                runner=runner,
                user_prompts=user_prompts,
                meta_datas=meta_datas,
                retrieval_queries=[prepare_conversation_history(row, multi_turn_context) for row in qa_dataset["messages"]],
            )

        # Save the output 
        json_dump = [response.to_dict(truncated=False) for response in responses.response_data]

        FileManager(cfg.output_folder + "/inference_log.json").dump_json(
            json_dump, pydantic_encoder=True
        )
        json_dump = [flatten_dict(response.to_dict(truncated=True)) for response in responses.response_data]

        active_run = mlflow.active_run()
        run_name = active_run.info.run_name if active_run else "responses"
        mlflow.log_table(data=pd.DataFrame(json_dump), artifact_file=f"{run_name}.json")

        # Evaluate the retrieval
        evaluation(cfg)


def clean_text(text):
    """Clean text of formatting characters"""
    text = _escape_re.sub(" ", text)
    text = _quotes_re.sub(r"\1", text)
    text = _whitespace_re.sub(" ", text).strip()
    return text


def prepare_conversation_history(messages, is_multi_turn=True):
    """Format the conversation structure for the prompt"""
    if is_multi_turn == False:
        return "User: " + messages[-1]["content"]

    if len(messages) == 1:
        return "User: " + messages[0]["content"]

    history = "Conversation history: "
    for i in range(len(messages)):
        if messages[i]["role"] == "user":
            if i < (len(messages) - 1):
                history = history + " \n\nUser:" + messages[i]["content"]
            else:
                history = history + "\n\nUser: " + messages[i]["content"]
        else:
            history = history + "\nAssistant: " + messages[i]["content"]
    return history


def prepare_data(df: pd.DataFrame, cfg) -> tuple[list, list]:
    """Prepare metadata for the inference runner"""
    meta_datas = []
    contexts = []
    seen_contexts = set()
    for _, row in df.iterrows():

        ground_truth, titles = prepare_ground_truth(cfg, row)
        ground_truth = clean_text(ground_truth[0])
        if (ground_truth not in seen_contexts):
            ground_truth_doc = Document(content=ground_truth, meta_data=MetaData({"title": titles[0]}))
            contexts.append(ground_truth_doc)
            seen_contexts.add(ground_truth)
        else:
            ground_truth_doc = next((d for d in contexts if d.content == ground_truth), None)

        # used for all rag methods excluding known_context
        for ctx in row["ctxs"]:
            text = clean_text(ctx["text"])
            if (text not in seen_contexts):
                contexts.append(
                    Document(content=text, meta_data=MetaData({"title": ctx["title"]}))
                )
                seen_contexts.add(text)

        if cfg.dataset.multiple_answers:
            meta_data = MetaData(
                {
                    "question": row["messages"][-1]["content"],
                    "reference_answer": list(row["answers"]),
                    "conversation_history": prepare_conversation_history(row["messages"]),
                    "reference_document": ground_truth_doc,
                }
            )
        else:
            meta_data = MetaData(
                {
                    "question": row["messages"][-1]["content"],
                    "reference_answer": row["answers"][0],
                    "conversation_history": prepare_conversation_history(row["messages"]),
                    "reference_document": ground_truth_doc,
                }
            )

        meta_datas.append(meta_data)

    return meta_datas, contexts


def prepare_ground_truth(cfg, row):
    """Retrieve ground truth context"""
    ctxs = []
    titles = []
    if "ground_truth_ctx" in row:
        if isinstance(row["ground_truth_ctx"], np.ndarray):
            for i in range(len(row["ground_truth_ctx"])):
                ctxs.append(row["ground_truth_ctx"][i]["ctx"])
                if "title" in row["ground_truth_ctx"][i]:
                    titles.append(row["ground_truth_ctx"][i]["title"])
                else:
                    titles.append("")
            if len(row["ground_truth_ctx"]) == 0:
                ctxs.append("no context available")
                titles.append("no title available")
            return ctxs, titles
        else:
            if "title" in row["ground_truth_ctx"]:
                return [row["ground_truth_ctx"]["ctx"]], [row["ground_truth_ctx"]["title"]]
            else:
                return [row["ground_truth_ctx"]["ctx"]], [""]
    else:
        if isinstance(row["ctxs"], np.ndarray):
            for i in range(len(row["ctxs"])):
                ctxs.append(row["ctxs"][i]["text"])
                titles.append(row["ctxs"][i]["title"])
            return ctxs, titles
        else:
            return [row["ctxs"][0]["text"]], [row["ctxs"][0]["title"]]

def known_context_rag(contexts, cfg, top_k):
    return KnownContext(
        context_collection=contexts,
        template_name=cfg.template_name,
        collection_name=cfg.dataset.subset,
        embedding_function="sentence-transformers/all-mpnet-base-v2",
        top_k=top_k,
        retrieval_only=False,
    )

def base_implementation_rag(contexts, cfg, top_k, batch_size = 2000):
    return BaseRAG(
        context_collection=contexts,
        template_name=cfg.template_name,
        collection_name=cfg.dataset.subset,
        embedding_function="sentence-transformers/all-mpnet-base-v2",
        top_k=top_k,
        retrieval_only=False,
        batch_size_insert=batch_size,
    )

def hybrid_bm25_rag(contexts, cfg, top_k, alpha = 0.5, beta = 0.5, batch_size = 2000):
    return HybridBM25RAG(
        context_collection=contexts,
        template_name=cfg.template_name,
        collection_name=cfg.dataset.subset,
        embedding_function="sentence-transformers/all-mpnet-base-v2",
        top_k=top_k,
        retrieval_only=False,
        batch_size_insert=batch_size,
        alpha=alpha,
        beta=beta,
    )

def summarization_rag(contexts, cfg, top_k, runner):
    return SummarizationRAG(
        context_collection=contexts,
        template_name="./src/exp/prompts/templates/rag_temp_sum.j2",
        collection_name=cfg.dataset.subset,
        embedding_function="sentence-transformers/all-mpnet-base-v2",
        additional_prompt="Rewrite the context to keep all essential facts and remove only irrelevant or redundant details, without adding new information.", 
        top_k=top_k,
        runner=runner,
        retrieval_only=False,
    )

def summarization_context_rag(contexts, cfg, top_k, runner):
    return SummarizationContextRAG(
        context_collection=contexts,
        template_name="./src/exp/prompts/templates/rag_temp_sum.j2",
        collection_name=cfg.dataset.subset,
        embedding_function="sentence-transformers/all-mpnet-base-v2",
        additional_prompt="Rewrite the context to keep all essential facts and remove only irrelevant or redundant details, without adding new information.", 
        top_k=top_k,
        runner=runner,
        retrieval_only=False,
    )

def reranker_rag(contexts, cfg, top_k, runner, rerank_ratio, batch_size = 2000):
    return RerankerRAG(
        context_collection=contexts,
        template_name=cfg.template_name,
        collection_name=cfg.dataset.subset,
        embedding_function="sentence-transformers/all-mpnet-base-v2",
        top_k=top_k,
        runner=runner,
        rerank_ratio=rerank_ratio,
        retrieval_only=False,
        batch_size_insert=batch_size,
    )

def hyde_rag(contexts, cfg, top_k, runner, batch_size = 2000):
    return HydeRAG(
        context_collection=contexts,
        template_name="./src/exp/prompts/templates/rag_temp_sum.j2",
        collection_name=cfg.dataset.subset,
        embedding_function="sentence-transformers/all-mpnet-base-v2",
        additional_prompt="Please write a passage to answer the question.",
        top_k=top_k,
        runner=runner,
        retrieval_only=False,
        batch_size_insert=batch_size,
    )

def hyde_reranker_rag(contexts, cfg, top_k, runner, rerank_ratio, batch_size = 2000):
    return HydeRerankerRAG(
        context_collection=contexts,
        template_name="./src/exp/prompts/templates/rag_temp_sum.j2",
        collection_name=cfg.dataset.subset,
        embedding_function="sentence-transformers/all-mpnet-base-v2",
        top_k=top_k,
        runner=runner,
        rerank_ratio=rerank_ratio,
        retrieval_only=False,
        batch_size_insert=batch_size,
    )

def self_rag(contexts, cfg, top_k, runner, batch_size = 2000):
    return SelfRAG(
        context_collection=contexts,
        template_name=cfg.template_name,
        collection_name=cfg.dataset.subset,
        embedding_function="sentence-transformers/all-mpnet-base-v2",
        top_k=top_k,
        runner=runner,
        retrieval_only=False,
        batch_size_insert=batch_size,
    )

def query_rewriting_rag(cfg, runner, user_prompts):
    sys_prompt = "Rewrite the last user question to be fully self-contained and clearly answerable. Include only the essential context from the conversation. Do not add information that was not explicitly mentioned. The rewritten question should be short, precise, and reflect exactly what the user wants to know. Output only the rewritten question."
    rag = QueryRewritingRAG(
        template_name="./src/exp/prompts/templates/query_rewriting.j2",
        collection_name=cfg.dataset.subset,
    )

    responses = rag.run(
                sys_prompt=sys_prompt,
                runner=runner,
                user_prompts=user_prompts)
    responses = [r.response for r in responses]

    return responses


if __name__ == "__main__":
    main()
