"""Module for evaluation of retrieval and QA results with MLflow tracking."""

import json
import logging
from pathlib import Path
from typing import Any

import hydra
import mlflow
import numpy as np
from encourage.llm import BatchInferenceRunner, Response
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics import Metric, MetricOutput
from vllm import SamplingParams

from exp.evaluation.config import Config
from exp.evaluation.factory_helper import load_metrics
from exp.utils.file_manager import FileManager
from exp.utils.flatten_dict import flatten_dict

logger = logging.getLogger(__name__)
config_path = str((Path(__file__).parents[3] / "conf").resolve())


class F1_Individual(Metric):
    """Computes the F1 score for the generated answers."""

    def __init__(self) -> None:
        super().__init__(
            name="f1",
            description="F1 score for the generated answers",
            required_meta_data=["reference_answer"],
        )

        self.metric = self.load_metric()

    def load_metric(self) -> Any:
        """Loads the F1 metric."""
        from evaluate import load

        return load("squad_v2")

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)
        # Initialize empty lists for formatted predictions and references
        formatted_predictions = []
        formatted_references = []

        # Use zip to iterate over predictions and references
        for i, r in enumerate(responses):
            formatted_predictions.append(
                {
                    "id": str(i),
                    "prediction_text": r.response,
                    "no_answer_probability": 0.0,
                }
            )
            formatted_references.append(
                {
                    "id": str(i),
                    "answers": [{"text": str(r.meta_data["reference_answer"]), "answer_start": 0}],
                }
            )

        # Call the compute function with formatted data
        output = []
        for pred, ref in zip(formatted_predictions, formatted_references):
            score = self.metric.compute(predictions=[pred], references=[ref])
            output.append(score["f1"])

        # output = self.metric.compute(
        #     predictions=formatted_predictions,
        #     references=formatted_references,
        # )

        if output is None:
            return MetricOutput(score=0.0, raw=[])
        return MetricOutput(score=float(np.mean(output) / 100), raw=output)


@hydra.main(version_base=None, config_path=config_path, config_name="defaults")
def main(cfg: Config) -> None:
    """Main function for evaluation of QA results with MLflow tracking."""
    # Set MLflow tracking configuration
    mlflow.set_tracking_uri(cfg.mlflow.uri)
    mlflow.set_experiment(experiment_name=cfg.mlflow.experiment_id)

    # Run the evaluation with MLflow tracking
    if mlflow.active_run().info.run_id if mlflow.active_run() else None:  # type: ignore
        evaluation(cfg)
    else:
        with mlflow.start_run():
            evaluation(cfg)


def get_most_correct_answer(responses):
    f1 = load_metrics(["F1"])[0]
    for response in responses:
        results = []
        answers = response["meta_data"]["reference_answer"]
        for answer in answers:
            temp = json.loads(json.dumps(response))
            temp["meta_data"]["reference_answer"] = answer
            r = [Response.from_dict(temp)]
            results.append(f1(r).score)

        response["meta_data"]["reference_answer"] = answers[results.index(max(results))]
    return responses


def evaluation(cfg: Config) -> None:
    """Evaluate the QA results with MLflow tracking."""
    flat_config = flatten_dict(cfg)
    mlflow.log_params(flat_config)

    with mlflow.start_span(name="loading_results"):
        results_path = Path(cfg.output_folder)
        if not results_path.exists() or not results_path.is_dir():
            raise ValueError(f"Results folder not found: {results_path}")

        # Convert to ResponseDataCollection format
        responses_json = FileManager(list(results_path.glob("inference_log.json"))[0]).load_json()

        if cfg.dataset.multiple_answers:
            responses_json = get_most_correct_answer(responses_json)

        responses = [Response.from_dict(item) for item in responses_json]

        logger.info(f"Loaded {len(responses)} responses!")

    sampling_params = SamplingParams(
        temperature=cfg.model.temperature, max_tokens=cfg.model.max_tokens
    )
    runner = BatchInferenceRunner(sampling_params, cfg.model.model_name, base_url=cfg.base_url)

    # Load metrics
    metrics: list[Metric] = load_metrics(cfg.metrics, runner)
    metrics_log = []
    f1 = F1_Individual()
    for metric in metrics:
        if metric.name == "f1":
            result: MetricOutput = f1(responses)
        else:
            result: MetricOutput = metric(responses)
        metrics_log.append({metric.name: result.to_dict()})

        mlflow.log_metric(metric.name, result.score)

    FileManager(cfg.output_folder + "/metrics_log.json").dump_json(
        metrics_log, pydantic_encoder=True
    )


if __name__ == "__main__":
    main()
