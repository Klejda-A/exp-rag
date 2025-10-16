# 🧪 Exp Template

This repository contains the code for Exp Template.

- 📦 **[uv](https://github.com/astral-sh/uv)**
  - A Python package manager replacing `pip` and `poetry`.
- 📝 **[pyproject](https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml/)**
  - For project management.
- ⚙️ **[hydra](https://hydra.cc)**
  - For flexible configuration management.
- 📊 **[mlflow](https://mlflow.org)**
  - For experiment tracking and visualization online.
- 🌱 **[encourage](https://github.com/uhh-hcds/encourage)**
  - A custom library for handling LLM inference, prompt handling, and utility functions.

---

## 🚀 Initialization

### 📦 UV

To initialize the environment using `uv`, run the following command:

```bash
uv venv --python 3.12
uv init
uv sync
```

## ⚡ Usage

When using this template you have to declare all your configuration parameters in the `conf/defaults.yaml` file. Also modify the `conf/model/defaults.yaml` and `conf/data/defaults.yaml` files to fit your needs.

### Run LLM

To run a LLM you can use config from the launch.json file. If you want to run it without it you can use the following command:

```bash
 CUDA_VISIBLE_DEVICES=1 uv run start_vllm_server_as_process.py model=llama3  
```

### Run Evaluation
To run the execution of the model you can use the following command:

```bash
uv run src/scivqa/evaluation/execution_rag.py
```

This has the default Base Implementation method which can be changed in the `execution_rag.py` file.

If something broke in the evaluation you can use the following command to run the evaluation again:

```bash
uv run src/scivqa/evaluation/evaluate.py
```

But you have to change the `output_folder` in the `defaults.yaml` to the folder where the execution results are stored. 

