python ../../analyze_updated_dataset_agnostic.py --file ../gsm8k_qwen3.5-2b.final.csv --prefinal --out_prefix gsm8k_qwen3.5-2b > gsm8k_qwen3.5-2b.final.txt

python ../../analyze_updated_dataset_agnostic.py --file ../gsm8k_qwen9b.final.csv --prefinal --out_prefix gsm8k_qwen9b > gsm8k_qwen3.5-9b.final.txt

python ../../analyze_updated_dataset_agnostic.py --file ../gsm8k_llama3.1-8B.final.csv --prefinal --out_prefix gsm8k_llama3.1-8B > gsm8k_llama3.1-8b.final.txt

python ../../analyze_updated_dataset_agnostic.py --file ../gsm8k_gpt-oss-20b_final_prefinal_fixed.csv  --prefinal --out_prefix gsm8k_gpt-oss-20b_final_prefinal_fixed > gsm8k_gpt-oss-20b_v2.final.txt

python ../../analyze_updated_dataset_agnostic.py --file ../math500_qwen3.5-2b.final.xlsx  --prefinal --out_prefix math500_qwen3.5-2b > math500_qwen3.5-2b.final.txt

python ../../analyze_updated_dataset_agnostic.py --file ../math500_qwen3-9b.final.xlsx  --prefinal --out_prefix math500_qwen3.5-9b > math500_qwen3.5-9b.final.txt

python ../../analyze_updated_dataset_agnostic.py --file ../math500_llama3.1-8B.final.xlsx  --prefinal --out_prefix math500_llama3.1-8B > math500_llama3.1-8b.final.txt

python ../../analyze_updated_final.py --file ../math500_gemma4-31b.final.xlsx --model_name "Gemma4-31b / MATH-500" --level_stratify > math500_gemma4-31b_levels.final.txt

python ../../analyze_updated_dataset_agnostic.py --file ../gpqa_qwen3.5-2b.final.xlsx  --prefinal --out_prefix gpqa_qwen3.5-2b > gpqa_qwen3.5-2b.final.txt

python ../../analyze_updated_dataset_agnostic.py --file ../gpqa_qwen3.5-9b.fixed.xlsx  --prefinal --out_prefix gpqa_qwen3.5-9b > gpqa_qwen3.5-9b.final.txt

python ../../analyze_updated_dataset_agnostic.py --file ../gpqa_llama3.1-8B.final.xlsx    --prefinal --out_prefix gpqa_llama3.1-8b > gpqa_llama3.1-8b.final.txt

python ../../analyze_updated_dataset_agnostic.py --file ../gpqa_gemma4-31b.final.xlsx    --prefinal --out_prefix gpqa_gemma4-31b > gpqa_gemma4-31b.final.txt

python ../../analyze_updated_dataset_agnostic.py --file ../gpqa_qwen3_5-9b_fixed.xlsx  --prefinal --out_prefix gpqa_qwen3_5-9b_fixed > gpqa_qwen3.5-9b.fixed.txt