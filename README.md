# BLAH9 – Evaluating the Correlation Between Consistency and Accuracy of LLMs on the EPOP Corpus

Evaluation of the correlation between consistency and accuracy of Large Language Models (LLMs) on the EPOP corpus, as part of BLAH9.

This repository contains all data, prompts, predictions, and evaluation scripts for the BLAH9 study on analyzing the relationship between **consistency** and **accuracy** of Large Language Models (LLMs) using the **EPOP corpus**.

## 📁 Repository Structure

### `dataset/`
Contains the complete set of publicly available training and test data from the EPOP corpus.

- `EPOP_documents/`: Raw source texts used in the experiments.
- `EPOP_json/`: Reference annotations (ground truth) for evaluation.

### `prediction/`
Houses the prediction results from four LLMs:

- `gpt-4o-mini`
- `deepseek`
- `kimi`
- `qwen3`

Each subfolder includes the model outputs on the EPOP test data.

### `prompt/`
Includes all prompt templates used in the experiments for different models and settings.

### `script/`
All evaluation scripts are included here.

- `llm_prediction_EPOP.py`: Script to generate predictions using the OpenAI API.
- `kbeval.py`, `kbeval-main-experiment.py`: Scripts for evaluating **accuracy** based on comparison with reference annotations.
- `kappa.py`: Script for measuring **consistency** across repeated generations (e.g., using Cohen's Kappa or similar metrics).
- `correlation_test.py`: Performs statistical correlation analysis between consistency and accuracy scores across models.

## 📊 Goals of This Project

- To investigate how consistent LLMs are when presented with the same prompts.
- To quantify how prediction consistency relates to accuracy in structured information extraction.
- To provide reproducible benchmarks using the EPOP dataset.

## 🔍 Citation

This work was conducted as part of the [BLAH9](https://blah9.linkedannotation.org/) workshop. Please cite appropriately if you use this dataset or code.

---

Feel free to open an issue if you have questions or need help reproducing the results.
