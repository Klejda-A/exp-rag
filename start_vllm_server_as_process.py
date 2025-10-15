"""Main script for starting the VLLM server."""

import os
import subprocess

import hydra
from dotenv import load_dotenv

from src.exp.evaluation.config import Config

# Load environment variables from .env file
load_dotenv(".env")
API_KEY = os.getenv("VLLM_API_KEY")
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES")
HUGGINGFACE_CACHE_DIR = os.getenv(
    "HUGGINGFACE_CACHE_DIR", os.path.expanduser("~/.cache/huggingface")
)
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

if not API_KEY:
    raise ValueError(
        "API key not found in the .env file. Please add VLLM_API_KEY=<your_api_key> to the file."
    )


@hydra.main(version_base=None, config_path="conf", config_name="defaults")
def main(cfg: Config) -> None:
    """Main function for starting the VLLM server."""
    # Ensure the Hugging Face cache directory exists
    if not os.path.exists(HUGGINGFACE_CACHE_DIR):
        os.makedirs(HUGGINGFACE_CACHE_DIR, exist_ok=True)

    # Ensure permissions for the cache directory
    os.chmod(HUGGINGFACE_CACHE_DIR, 0o777)

    print("quant" + str(cfg.model.quantization))
    # Construct the Docker command
    vllm_command = [
        "vllm serve",
        cfg.model.model_name,
        "--dtype",
        "auto",
        "--api-key",
        API_KEY,
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        cfg.model.tool_call_parser,
        "--gpu-memory-utilization",
        str(cfg.model.gpu_memory_utilization),
        "--port",
        str(cfg.vllm_port),
        "--max-model-len",
        str(cfg.model.max_model_len),
        "--tensor-parallel-size",
        str(cfg.model.tensor_parallel_size),
    ]

    # Execute the serve command
    try:
        print(f"Running vllm command: {' '.join(vllm_command)}")
        subprocess.run(" ".join(vllm_command), shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")


if __name__ == "__main__":
    main()
