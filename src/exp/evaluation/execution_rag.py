"""Module for evaluation of QA datasets."""

from pathlib import Path

import hydra
import litellm
import mlflow
import mlflow.data.pandas_dataset
import pandas as pd
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from encourage.llm import BatchInferenceRunner
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context, Document
from encourage.prompts.meta_data import MetaData
from encourage.rag import KnownContext, BaseRAG
from vllm import SamplingParams

from exp.evaluation.config import Config
from exp.evaluation.evaluation import main as evaluation
from exp.utils.file_manager import FileManager
from exp.utils.flatten_dict import flatten_dict

config_path = str((Path(__file__).parents[3] / "conf").resolve())

def prepare_conversation_history(messages, is_multi_turn=True):
    if is_multi_turn == False:
        return("User: " + messages[-1]["content"])

    if len(messages) == 1:
        return("User: " + messages[0]["content"])
    
    history = "Conversation history: "
    # history = ""
    for i in range(len(messages)):
        if messages[i]["role"] == "user":
            if i < (len(messages) - 1):
                history = history + " \n\nUser:" + messages[i]["content"]
            else:
                history = history + "\n\nUser: " + messages[i]["content"]
        else:
            history = history + "\nAssistant: " + messages[i]["content"]
    return(history)


def prepare_data(df: pd.DataFrame, cfg) -> tuple[list, list]:
    """Prepare data for the inference runner."""
    meta_datas = []
    contexts = []
    seen_contexts = set()
    for _, row in df.iterrows():

        ground_truth, titles = prepare_ground_truth(cfg, row)
        ground_truth_doc = Document(content=ground_truth[0], meta_data=MetaData({"title": titles[0]}))
        seen_contexts.add("text")
        contexts.append(ground_truth_doc)

        for ctx in row["ctxs"]:
            text = ctx["text"]
            if text not in seen_contexts:
                contexts.append(Document(content=text, meta_data=MetaData({"title": ctx.get("title")})))
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

        # context=""
        # for ctx in row["ctxs"]:
        #     text = ctx["text"]
        #     if text not in seen_contexts:
        #         context = context + text + " "
        #         seen_contexts.add(text)
        # contexts.append(Document(content=context, meta_data=MetaData({"title": ctx.get("title")})))
    return meta_datas, contexts


def prepare_ground_truth(cfg, row):
    ctxs = []
    titles = []
    if cfg.dataset.subset == "hybridial":
        return ctxs, titles
    elif "ground_truth_ctx" in row:
        if isinstance(row["ground_truth_ctx"], np.ndarray):
            for i in range(len(row["ground_truth_ctx"])):
                ctxs.append(row["ground_truth_ctx"][i]["ctx"])
                titles.append(row["ground_truth_ctx"][i]["title"])
            return ctxs, titles
        else:
            return [row["ground_truth_ctx"]["ctx"]], [row["ground_truth_ctx"]["title"]]
    else:
        if isinstance(row["ctxs"], np.ndarray):
            for i in range(len(row["ctxs"])):
                ctxs.append(row["ctxs"][i]["text"])
                titles.append(row["ctxs"][i]["title"])
            return ctxs, titles
        else:
            return [row["ctxs"][0]["text"]], [row["ctxs"][0]["title"]]


@hydra.main(version_base=None, config_path=config_path, config_name="defaults")
def main(cfg: Config) -> None:
    """Main function for evaluation of QA datasets."""
    # Load dataset from Huggingface
    load_dotenv(".env")
    
    qa_dataset = load_dataset(cfg.dataset.name, cfg.dataset.subset, split=cfg.dataset.split).to_pandas()

    litellm._logging._disable_debugging()
    mlflow.openai.autolog()

    sampling_params = SamplingParams(
        temperature=cfg.model.temperature, max_tokens=cfg.model.max_tokens
    )
    runner = BatchInferenceRunner(sampling_params, cfg.model.model_name, base_url=cfg.base_url)
    sys_prompt = FileManager(cfg.dataset.sys_prompt_path).read()

    ## Run the Inference
    mlflow.set_tracking_uri(cfg.mlflow.uri)
    mlflow.set_experiment(experiment_name=cfg.mlflow.experiment_id)

    with mlflow.start_run():
        mlflow.log_params(flatten_dict(cfg))
        mlflow.log_params({"dataset_size": len(qa_dataset)})
        mlflow.log_input(
            mlflow.data.pandas_dataset.from_pandas(
                qa_dataset, name=cfg.dataset.name
            ),
            context="inference",
        )

        with mlflow.start_span(name="root"):            
            meta_datas, contexts = prepare_data(qa_dataset, cfg)

            rag = BaseRAG(
                        context_collection=contexts,
                        template_name=cfg.template_name,
                        collection_name=cfg.dataset.subset,
                        embedding_function="sentence-transformers/all-mpnet-base-v2",
                        top_k=1,
                        retrieval_only=False,
                        batch_size_insert=2000,
                    )
            
            responses = rag.run(
                runner=runner,
                sys_prompt=sys_prompt,
                user_prompts=[prepare_conversation_history(row, False) for row in qa_dataset["messages"]],
                meta_data=meta_datas,
                retrieval_instruction=[prepare_conversation_history(row, True) for row in qa_dataset["messages"]],
                )

            # rag = KnownContext(
            #             context_collection=contexts,
            #             template_name=cfg.template_name,
            #             collection_name=cfg.dataset.subset,
            #             embedding_function="sentence-transformers/all-mpnet-base-v2",
            #             top_k=1,
            #             retrieval_only=False,
            #         )
            
            # responses = rag.run(
            #     runner=runner,
            #     sys_prompt=sys_prompt,
            #     user_prompts=[prepare_conversation_history(row, False) for row in qa_dataset["messages"]],
            #     meta_data=meta_datas,
            #     retrieval_instruction=[prepare_conversation_history(row, True) for row in qa_dataset["messages"]],
            #     )

            

        # Save the output to hydra folder0
        json_dump = [response.to_dict() for response in responses.response_data]

        FileManager(cfg.output_folder + "/inference_log.json").dump_json(
            json_dump, pydantic_encoder=True
        )
        json_dump = [flatten_dict(response.to_dict()) for response in responses.response_data]

        active_run = mlflow.active_run()
        run_name = active_run.info.run_name if active_run else "responses"
        mlflow.log_table(data=pd.DataFrame(json_dump), artifact_file=f"{run_name}.json")

        # Evaluate the retrieval
        evaluation(cfg)


if __name__ == "__main__":
    main()
