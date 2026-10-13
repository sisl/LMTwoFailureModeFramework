# How Language Models Fail: Token-Level Signatures of Committed and Persistent Reasoning Failures

Code and data for the paper submission "How Language Models Fail: Token-Level Signatures of Committed and Persistent Reasoning Failures."

## Overview

Failures in language model reasoning emerge through distinct patterns that leave identifiable signatures in the reasoning trace. We characterize these failures using token-level uncertainty signals, finding they arise through two empirically distinguishable modes. We denote the first failure type as "committed failure", in which an early prefix of the reasoning trace is most predictive of eventual failure. A central diagnostic signature is the commitment point, beyond which considering additional tokens hurts failure detection. The second is persistent uncertainty, where the full trace rather than any early prefix is most predictive of failure. We separate these two modes across 16 model-dataset configurations spanning five model families and four reasoning domains with a pre-specified, selection-free statistic (Mann-Whitney p = 0.021), and show that answer-scoring errors can manufacture apparent early-commitment signals that disappear under correct scoring. Since these signatures identify how a model is failing, they show which failure detection strategy can be applied. On GPQA, single-completion confidence identifies inputs on which self-consistency can be skipped in the committed regime; on MATH-500 it cannot, and cross-completion agreement is the stronger signal. These results offer a foundation for understanding when LLM reasoning failures become detectable and for adapting detection strategies accordingly.

## Repository Structure
```
├── *.py                                    # Inference pipelines for all model-dataset configurations
├── analyze_updated_dataset_agnostic.py     # Computes PR-AUC curves, bootstrap CIs (10,000 samples),
│                                           # and failure mode classification
└── results/
    ├── *.csv / *.xlsx                      # Inference outputs: token-level uncertainty features
    │                                       # and failure labels per configuration
    └── analysis/
        ├── failure_modes/                  # Per-configuration failure mode classifications
        │   └── *.txt                       # Committed/persistent classification with bootstrap
        │                                   # CIs and ΔPR-AUC statistics
        ├── self_consistency/               # Self-consistency triage and complementarity results
        └── plots/                          # PR-AUC curve plots for all configurations
```

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

