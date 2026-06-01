# How Language Models Fail: Token-Level Signatures of Committed and Persistent Reasoning Failures

Code and data for the EMNLP 2025 paper submission "How Language Models Fail: Token-Level Signatures of Committed and Persistent Reasoning Failures."

## Overview

We characterize LLM reasoning failures through token-level uncertainty signals extracted from chain-of-thought traces, finding that failures emerge through two empirically distinguishable modes: committed failure, where a model locks onto an incorrect reasoning path early, and persistent uncertainty, where uncertainty accumulates throughout the trace. We identify the commitment point, the position in a reasoning trace beyond which additional tokens hurt rather than help failure detection, and demonstrate direct implications for self-consistency.

## Repository Structure
├── <dataset>_async_vllm.py       # Inference pipelines for all model-dataset configurations
├── analyze_updated_dataset_agnostic.py  # Main analysis script: computes PR-AUC curves,
│                                 # bootstrap CIs (10,000 samples), and failure mode classification
├── results/
│   ├── *.csv / *.xlsx            # Inference outputs: token-level uncertainty features
│   │                             # and failure labels per configuration
│   └── analysis/
│       ├── failure_modes/        # Per-configuration failure mode classification outputs
│       │   └── *.txt             # Committed/persistent classification with bootstrap
│       │                         # confidence intervals and ΔPR-AUC statistics
│       ├── self_consistency/     # Self-consistency triage and complementarity results
│       └── plots/                # PR-AUC curve plots for all configurations

## Reproducing Results

**Step 1: Run inference**
```bash
python <dataset>_async_vllm.py
```
Output `.csv` / `.xlsx` files are saved to `results/`.

**Step 2: Run failure mode analysis**
```bash
python analyze_updated_dataset_agnostic.py --input results/<output_file>
```
This computes token-level uncertainty features over early windows, fits a stratified logistic regression classifier with 5-fold cross-validation, computes PR-AUC at each window size and runs 10,000-iteration paired bootstrap to produce confidence intervals on ΔPR-AUC. Classification outputs are saved to `results/analysis/failure_modes/`.

## Requirements

```bash
pip install -r requirements.txt
```

Open-weight models are served via vLLM. GPT-4o and Gemini2.5-Pro experiments use the OpenAI API with `top_logprobs=20`. All experiments use `temperature=0.6`.

