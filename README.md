# LLMFailureMode

This repository contains raw experimental results and analysis scripts for studying LLM failure modes across multiple benchmarks.

## Repository Structure

- `results/`  
  Contains all raw experiment outputs and result files.

- `failure_mode/`  
  Contains `.txt` files with detailed failure mode analyses.

## Analysis

To analyze raw result files, use:

```bash
python analyze_updated_dataset_agnostic.py\
```

Use math_async_vllm.py for math 500, gpqa_async_vllm.py for gpqa, livebench_async_vllm.py for livecodebench and updated_gsm8k_vllm.py for gsm8k runs to generate their output. 
