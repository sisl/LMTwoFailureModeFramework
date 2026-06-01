#!/usr/bin/env python3
"""
livecodebench_vllm_async.py  —  LiveCodeBench async pipeline.

Async version of livecodebench_vllm.py. Sends --concurrency requests to
vLLM in parallel via AsyncOpenAI + asyncio.Semaphore.

Code execution (subprocess) is offloaded to a thread pool via
asyncio.get_event_loop().run_in_executor so it never blocks the event loop.
"""

import argparse
import ast
import asyncio
import io
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import textwrap
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from datasets import load_dataset
from openai import AsyncOpenAI

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABLATION_WINDOWS: List[int] = [128, 256, 400, 512, 1024, 2048]
DIFFICULTY_CHOICES = ["easy", "medium", "hard"]

DEFAULT_SYSTEM = (
    "You are a careful reasoning assistant. "
    "Think through the problem step by step, then give your final answer.\n\n"
    "Write your solution inside a ```python ... ``` code block.\n\n"
    "Rules:\n"
    "- Reason first, then write code. Do not interleave reasoning and code.\n"
    "- Only one ```python ... ``` block in your response.\n"
    "- The code block must contain a complete, runnable Python solution."
)

DEFAULT_VOCAB_SIZE = 200

EXEC_TIMEOUT_S    = 10
COMPILE_TIMEOUT_S = 10


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("lcb_vllm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def logsumexp(logps: List[float]) -> float:
    m = max(logps)
    return m + math.log(sum(math.exp(lp - m) for lp in logps))


def safe_stats(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": float("nan"), "max": float("nan"), "std": float("nan")}
    mean = sum(xs) / len(xs)
    mx   = max(xs)
    var  = sum((x - mean) ** 2 for x in xs) / len(xs)
    return {"mean": float(mean), "max": float(mx), "std": float(math.sqrt(var))}


# ---------------------------------------------------------------------------
# Uncertainty signals
# ---------------------------------------------------------------------------

def entropy_from_full_vocab_logps(vocab_logps: List[float]) -> float:
    lse = logsumexp(vocab_logps)
    ent = 0.0
    for lp in vocab_logps:
        p = math.exp(lp - lse)
        if p > 0.0:
            ent -= p * math.log(p)
    return ent


def margin_from_full_vocab_logps(vocab_logps: List[float]) -> float:
    top1 = -float("inf")
    top2 = -float("inf")
    for lp in vocab_logps:
        if lp > top1:
            top2, top1 = top1, lp
        elif lp > top2:
            top2 = lp
    lse = logsumexp(vocab_logps)
    p1  = math.exp(top1 - lse)
    p2  = math.exp(top2 - lse) if top2 != -float("inf") else 0.0
    return p1 - p2


def nucleus_size_from_full_vocab_logps(
    vocab_logps: List[float], nucleus_p: float = 0.9
) -> int:
    lse   = logsumexp(vocab_logps)
    probs = sorted([math.exp(lp - lse) for lp in vocab_logps], reverse=True)
    s = 0.0
    for i, p in enumerate(probs, start=1):
        s += p
        if s >= nucleus_p:
            return i
    return len(probs)


def near_tie_count_from_full_vocab_logps(
    vocab_logps: List[float], tie_delta: float = 0.1
) -> int:
    top1 = max(vocab_logps)
    return max(0, sum(1 for lp in vocab_logps if lp >= top1 - tie_delta) - 1)


def _compute_signals_from_token_list(
    token_logprobs: List[Any],
    *,
    nucleus_p: float,
    tie_delta: float,
    suffix: str,
) -> Dict[str, float]:
    entropies, margins, nlls, nucleus_sizes, near_ties, forks = [], [], [], [], [], []
    for tok in token_logprobs:
        lp_chosen = float(tok.logprob)
        nlls.append(-lp_chosen)
        top = getattr(tok, "top_logprobs", None)
        if not top:
            continue
        vocab_logps = [float(x.logprob) for x in top]
        entropies.append(entropy_from_full_vocab_logps(vocab_logps))
        margins.append(margin_from_full_vocab_logps(vocab_logps))
        ns = nucleus_size_from_full_vocab_logps(vocab_logps, nucleus_p=nucleus_p)
        nt = near_tie_count_from_full_vocab_logps(vocab_logps, tie_delta=tie_delta)
        nucleus_sizes.append(float(ns))
        near_ties.append(float(nt))
        forks.append(1 if nt > 0 else 0)

    ent  = safe_stats(entropies)
    mar  = safe_stats(margins)
    nll  = safe_stats(nlls)
    nuc  = safe_stats(nucleus_sizes)
    tie  = safe_stats(near_ties)
    fork_rate = float(sum(forks) / len(forks)) if forks else float("nan")

    return {
        f"entropy_mean{suffix}":      ent["mean"],
        f"entropy_max{suffix}":       ent["max"],
        f"entropy_std{suffix}":       ent["std"],
        f"margin_mean{suffix}":       mar["mean"],
        f"margin_max{suffix}":        mar["max"],
        f"margin_std{suffix}":        mar["std"],
        f"nll_mean{suffix}":          nll["mean"],
        f"nll_max{suffix}":           nll["max"],
        f"nll_std{suffix}":           nll["std"],
        f"fork_rate{suffix}":         fork_rate,
        f"nucleus_size_mean{suffix}": nuc["mean"],
        f"nucleus_size_max{suffix}":  nuc["max"],
        f"near_tie_mean{suffix}":     tie["mean"],
        f"near_tie_max{suffix}":      tie["max"],
    }


def compute_signals_multiwindow(
    token_logprobs: List[Any],
    *,
    windows: List[int],
    nucleus_p: float,
    tie_delta: float,
    suffix: str = "",
) -> Dict[str, float]:
    result = _compute_signals_from_token_list(
        token_logprobs, nucleus_p=nucleus_p, tie_delta=tie_delta, suffix=suffix
    )
    result[f"full_len{suffix}"] = float(len(token_logprobs))
    for T in windows:
        early = token_logprobs[:T]
        result.update(_compute_signals_from_token_list(
            early, nucleus_p=nucleus_p, tie_delta=tie_delta, suffix=f"{suffix}_w{T}"
        ))
        result[f"window_len{suffix}_w{T}"] = float(len(early))
    return result


def _make_nan_signals_multiwindow(
    windows: List[int], suffix: str = "",
    nucleus_p: float = 0.9, tie_delta: float = 0.1,
) -> Dict[str, float]:
    result = _compute_signals_from_token_list(
        [], nucleus_p=nucleus_p, tie_delta=tie_delta, suffix=suffix
    )
    result[f"full_len{suffix}"] = float("nan")
    for T in windows:
        result.update(_compute_signals_from_token_list(
            [], nucleus_p=nucleus_p, tie_delta=tie_delta, suffix=f"{suffix}_w{T}"
        ))
        result[f"window_len{suffix}_w{T}"] = float("nan")
    return result


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _parse_test_cases(raw: Any) -> List[Dict[str, str]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        try:
            import base64, zlib, pickle
            decompressed = zlib.decompress(base64.b64decode(raw))
            unpickled = pickle.loads(decompressed)
            if isinstance(unpickled, str):
                parsed = json.loads(unpickled)
                if isinstance(parsed, list):
                    return parsed
            elif isinstance(unpickled, list):
                return unpickled
        except Exception:
            pass
    return []


def load_livecodebench_examples(
    difficulties: List[str],
    jsonl_files: Optional[List[str]] = None,
    min_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if jsonl_files is None:
        jsonl_files = [
            "hf://datasets/livecodebench/code_generation_lite/test.jsonl",
            "hf://datasets/livecodebench/code_generation_lite/test2.jsonl",
            "hf://datasets/livecodebench/code_generation_lite/test3.jsonl",
            "hf://datasets/livecodebench/code_generation_lite/test4.jsonl",
            "hf://datasets/livecodebench/code_generation_lite/test5.jsonl",
            "hf://datasets/livecodebench/code_generation_lite/test6.jsonl",
        ]

    ds = load_dataset("json", data_files=jsonl_files, split="train")
    difficulties_lower = {d.lower() for d in difficulties}

    import datetime
    cutoff_dt = None
    if min_date is not None:
        cutoff_dt = datetime.datetime.strptime(min_date, "%Y-%m-%d")

    examples = []
    for ex in ds:
        diff = str(ex.get("difficulty", "")).lower()
        if diff not in difficulties_lower:
            continue

        if cutoff_dt is not None:
            contest_date = ex.get("contest_date")
            if contest_date is None or contest_date < cutoff_dt:
                continue

        question_id      = str(ex.get("question_id", len(examples)))
        question_title   = str(ex.get("question_title", "")).strip()
        question_content = str(ex.get("question_content", "")).strip()
        starter_code     = str(ex.get("starter_code", "") or "").strip()
        platform         = str(ex.get("platform", "unknown")).strip()

        private_tests = _parse_test_cases(ex.get("private_test_cases"))
        public_tests  = _parse_test_cases(ex.get("public_test_cases"))
        test_cases    = private_tests if private_tests else public_tests

        prompt_parts = [question_content]
        if starter_code:
            prompt_parts.append(f"\nStarter code:\n```python\n{starter_code}\n```")
        formatted = "\n\n".join(prompt_parts)

        examples.append({
            "question_id":      question_id,
            "question_title":   question_title,
            "question_content": question_content,
            "question":         formatted,
            "difficulty":       diff,
            "platform":         platform,
            "starter_code":     starter_code,
            "test_cases":       test_cases,
            "n_tests":          len(test_cases),
        })

    return examples


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code(text: str) -> Tuple[Optional[str], str]:
    matches = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip(), "python_fence"
    matches = re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    if matches:
        code_blocks = [m.strip() for m in matches if len(m.strip()) > 20]
        if code_blocks:
            return code_blocks[-1], "generic_fence"
    return None, "none"


# ---------------------------------------------------------------------------
# Prefinal stripping
# ---------------------------------------------------------------------------

def find_prefinal_char_cutoff(assistant_text: str) -> Optional[int]:
    m = re.search(r"```python", assistant_text)
    if m:
        idx        = m.start()
        line_start = assistant_text.rfind("\n", 0, idx)
        return line_start + 1 if line_start != -1 else idx

    m = re.search(r"```", assistant_text)
    if m:
        idx        = m.start()
        line_start = assistant_text.rfind("\n", 0, idx)
        return line_start + 1 if line_start != -1 else idx

    m = re.search(
        r"^(def\s+\w+|class\s+Solution|import\s+sys|import\s+os|from\s+\w+\s+import)",
        assistant_text, re.MULTILINE
    )
    if m:
        idx        = m.start()
        line_start = assistant_text.rfind("\n", 0, idx)
        return line_start + 1 if line_start != -1 else idx

    return None


def truncate_token_logprobs_to_char_cutoff(
    token_logprobs: List[Any], cutoff_char_idx: int
) -> List[Any]:
    acc, out = 0, []
    for tok in token_logprobs:
        t = getattr(tok, "token", "") or ""
        if acc + len(t) <= cutoff_char_idx:
            out.append(tok)
            acc += len(t)
        else:
            break
    return out


# ---------------------------------------------------------------------------
# Code execution  (blocking — called via run_in_executor)
# ---------------------------------------------------------------------------

def _run_test_case(
    code: str,
    test_input: str,
    expected_output: str,
    timeout: float,
) -> Tuple[bool, str]:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        actual   = result.stdout.strip()
        expected = expected_output.strip()

        def normalise(s: str) -> str:
            return "\n".join(" ".join(line.split()) for line in s.splitlines()).strip()

        passed = normalise(actual) == normalise(expected)
        status = "pass" if passed else f"wrong_answer(got={actual!r:.50}, exp={expected!r:.50})"
        return passed, status

    except subprocess.TimeoutExpired:
        return False, f"timeout(>{timeout}s)"
    except Exception as e:
        return False, f"error({e})"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def evaluate_correctness(
    code: Optional[str],
    test_cases: List[Dict[str, str]],
    timeout: float = EXEC_TIMEOUT_S,
    max_tests: int = 10,
) -> Tuple[int, str]:
    if code is None:
        return 0, "no_code_extracted"
    try:
        ast.parse(code)
    except SyntaxError as e:
        return 0, f"syntax_error({e})"
    if not test_cases:
        return -1, "no_test_cases"
    tests_to_run = test_cases[:max_tests]
    for tc in tests_to_run:
        inp = str(tc.get("input", tc.get("stdin", "")))
        out = str(tc.get("output", tc.get("stdout", "")))
        passed, status = _run_test_case(code, inp, out, timeout)
        if not passed:
            return 0, f"failed_test({status})"
    return 1, f"all_pass({len(tests_to_run)}_tests)"


# ---------------------------------------------------------------------------
# Degenerate detection
# ---------------------------------------------------------------------------

def detect_degenerate(text: str) -> bool:
    if re.search(r"(.)\1{30,}", text):
        return True
    if re.search(r"\d{20,}", text):
        return True
    return False


# ---------------------------------------------------------------------------
# Async vLLM call
# ---------------------------------------------------------------------------

async def call_vllm_chat(
    client: AsyncOpenAI,
    *,
    model: str,
    system_msg: str,
    user_msg: str,
    max_tokens: int,
    temperature: float,
    vocab_size: int,
    timeout_s: float,
    retries: int,
    logger: logging.Logger,
    enable_thinking: bool = False,
) -> Tuple[str, str, List[Any], List[Any], bool, Optional[str]]:
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    extra_body: Dict[str, Any] = {}
    if enable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp   = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=True,
                top_logprobs=vocab_size,
                timeout=timeout_s,
                **({"extra_body": extra_body} if extra_body else {}),
            )
            choice = resp.choices[0]
            response_text     = choice.message.content or ""
            thinking_text     = getattr(choice.message, "reasoning_content", None) or ""
            response_logprobs = []
            thinking_logprobs = []
            if getattr(choice, "logprobs", None) is not None:
                response_logprobs = choice.logprobs.content or []
                thinking_logprobs = getattr(
                    choice.logprobs, "reasoning_content", None
                ) or []
            hit_max       = bool(getattr(choice, "finish_reason", None) == "length")
            finish_reason = getattr(choice, "finish_reason", None)
            return (
                thinking_text, response_text,
                thinking_logprobs, response_logprobs,
                hit_max, finish_reason,
            )
        except Exception as e:
            last_err = e
            logger.warning(f"vLLM call failed (attempt {attempt+1}/{retries+1}): {e}")
            await asyncio.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"vLLM call failed after retries: {last_err}")


# ---------------------------------------------------------------------------
# Async worker: one problem under semaphore
# ---------------------------------------------------------------------------

async def process_one(
    sem: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    client: AsyncOpenAI,
    args,
    logger: logging.Logger,
    i: int,
    n: int,
    ex: Dict[str, Any],
    start_time: float,
    comp_idx: int = 0,
) -> Optional[Dict[str, Any]]:
    async with sem:
        t0 = time.time()
        try:
            (
                thinking_text, response_text,
                thinking_logprobs, response_logprobs,
                hit_max_tokens, finish_reason,
            ) = await call_vllm_chat(
                client,
                model=args.model,
                system_msg=args.system,
                user_msg=ex["question"],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                vocab_size=args.vocab_size,
                timeout_s=args.timeout_s,
                retries=args.retries,
                logger=logger,
                enable_thinking=args.enable_thinking,
            )
        except Exception as e:
            logger.error(f"[{i+1}/{n}] Fatal error after retries: {e} — skipping.")
            return None

        if thinking_text:
            assistant_text = f"<think>{thinking_text}</think>\n{response_text}"
        else:
            assistant_text = response_text

        token_logprobs = list(thinking_logprobs) + list(response_logprobs)

        extract_source = response_text if response_text else assistant_text
        extracted_code, extraction_method = extract_code(extract_source)

        # Code execution is blocking (subprocess) — run in thread pool
        if args.no_execute:
            correct     = float("nan")
            exec_status = "skipped"
        else:
            loop = asyncio.get_event_loop()
            correct_int, exec_status = await loop.run_in_executor(
                executor,
                evaluate_correctness,
                extracted_code,
                ex["test_cases"],
                args.exec_timeout,
                args.max_tests,
            )
            correct = float("nan") if correct_int == -1 else int(correct_int)

        # Uncertainty signals
        full_signals = compute_signals_multiwindow(
            token_logprobs,
            windows=args.windows,
            nucleus_p=args.nucleus_p,
            tie_delta=args.tie_delta,
            suffix="",
        )

        cutoff = find_prefinal_char_cutoff(assistant_text)
        if cutoff is not None and cutoff > 0:
            prefinal_toks = truncate_token_logprobs_to_char_cutoff(
                token_logprobs, cutoff
            )
            prefinal_signals = compute_signals_multiwindow(
                prefinal_toks,
                windows=args.windows,
                nucleus_p=args.nucleus_p,
                tie_delta=args.tie_delta,
                suffix="_prefinal",
            )
            prefinal_len = len(prefinal_toks)
        else:
            prefinal_signals = _make_nan_signals_multiwindow(
                args.windows, suffix="_prefinal"
            )
            prefinal_len = 0

        thinking_len = 0
        response_len = 0
        thinking_signals: Dict[str, Any] = {}
        response_signals: Dict[str, Any] = {}

        if args.thinking_mode:
            thinking_len = len(thinking_logprobs)
            response_len = len(response_logprobs)
            thinking_signals = (
                compute_signals_multiwindow(
                    list(thinking_logprobs),
                    windows=args.windows,
                    nucleus_p=args.nucleus_p,
                    tie_delta=args.tie_delta,
                    suffix="_thinking",
                ) if thinking_logprobs else
                _make_nan_signals_multiwindow(args.windows, suffix="_thinking")
            )
            response_signals = (
                compute_signals_multiwindow(
                    list(response_logprobs),
                    windows=args.windows,
                    nucleus_p=args.nucleus_p,
                    tie_delta=args.tie_delta,
                    suffix="_response",
                ) if response_logprobs else
                _make_nan_signals_multiwindow(args.windows, suffix="_response")
            )
        else:
            response_len = len(token_logprobs)

        generated_len = len(token_logprobs)
        deg = int(detect_degenerate(assistant_text))
        dt  = time.time() - t0
        elapsed = time.time() - start_time

        correct_str = str(int(correct)) if not (
            isinstance(correct, float) and math.isnan(correct)
        ) else "nan"
        logger.info(
            f"[{i+1}/{n} c{comp_idx+1}/{args.n_completions}] {ex['difficulty']:6s} correct={correct_str} "
            f"extr={extraction_method} exec={exec_status[:40]} "
            f"gen={generated_len} prefinal={prefinal_len} deg={deg} "
            f"lat={dt:.1f}s elapsed={elapsed/3600:.2f}h"
        )

        return {
            "checkpoint_key":    f"{ex['question_id']}::{comp_idx}",
            "completion_idx":    comp_idx,
            "question_id":       ex["question_id"],
            "question_title":    ex["question_title"],
            "difficulty":        ex["difficulty"],
            "platform":          ex["platform"],
            "n_tests":           ex["n_tests"],
            "extracted_code":    (extracted_code or "")[:2000],
            "extraction_method": extraction_method,
            "correct":           correct,
            "exec_status":       exec_status,
            "assistant_output":  assistant_text[:4000],
            "thinking_text":     thinking_text[:2000] if thinking_text else "",
            "response_text":     response_text[:2000],
            "generated_len":     int(generated_len),
            "thinking_len":      int(thinking_len),
            "response_len":      int(response_len),
            "prefinal_len":      int(prefinal_len),
            "hit_max_tokens":    int(bool(hit_max_tokens)),
            "finish_reason":     str(finish_reason) if finish_reason else "",
            "degenerate":        int(deg),
            "latency_s":         float(dt),
            **full_signals,
            **prefinal_signals,
            **thinking_signals,
            **response_signals,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main() -> None:
    ap = argparse.ArgumentParser(
        description="LiveCodeBench — async uncertainty pipeline (Easy + Medium)"
    )
    ap.add_argument("--base_url",        type=str)
    ap.add_argument("--model",           type=str)
    ap.add_argument("--system",          type=str,   default=DEFAULT_SYSTEM)
    ap.add_argument("--difficulty",      type=str,   nargs="+",
                    default=["easy", "medium"], choices=DIFFICULTY_CHOICES)
    ap.add_argument("--n",               type=int,   default=0,
                    help="Max problems (0 = all in selected difficulties)")
    ap.add_argument("--n_completions",   type=int,   default=1,
                    help="Completions per problem (default: 1; set to e.g. 15 for self-consistency)")
    ap.add_argument("--concurrency",     type=int,   default=8,
                    help="Max parallel in-flight requests to vLLM (default: 8)")
    ap.add_argument("--exec_workers",    type=int,   default=4,
                    help="Thread pool size for subprocess code execution (default: 4)")
    ap.add_argument("--max_tokens",      type=int,   default=16384)
    ap.add_argument("--temperature",     type=float, default=0.6)
    ap.add_argument("--vocab_size",      type=int,   default=DEFAULT_VOCAB_SIZE)
    ap.add_argument("--nucleus_p",       type=float, default=0.7)
    ap.add_argument("--tie_delta",       type=float, default=0.5)
    ap.add_argument("--save_every",      type=int,   default=50,
                    help="Checkpoint after every N completed rows.")
    ap.add_argument("--out_prefix",      type=str,   default="lcb_vllm")
    ap.add_argument("--timeout_s",       type=float, default=900.0)
    ap.add_argument("--exec_timeout",    type=float, default=EXEC_TIMEOUT_S)
    ap.add_argument("--max_tests",       type=int,   default=10)
    ap.add_argument("--no_execute",      action="store_true", default=False)
    ap.add_argument("--retries",         type=int,   default=2)
    ap.add_argument("--thinking_mode",   action="store_true", default=False)
    ap.add_argument("--enable_thinking", action="store_true", default=False)
    ap.add_argument("--windows",         type=int,   nargs="+", default=ABLATION_WINDOWS)
    ap.add_argument("--jsonl_files",     type=str,   nargs="+", default=[
        "hf://datasets/livecodebench/code_generation_lite/test.jsonl",
        "hf://datasets/livecodebench/code_generation_lite/test2.jsonl",
        "hf://datasets/livecodebench/code_generation_lite/test3.jsonl",
        "hf://datasets/livecodebench/code_generation_lite/test4.jsonl",
        "hf://datasets/livecodebench/code_generation_lite/test5.jsonl",
        "hf://datasets/livecodebench/code_generation_lite/test6.jsonl",
    ])
    ap.add_argument("--min_date",        type=str,   default="2024-01-01")
    args = ap.parse_args()

    log_path  = f"{args.out_prefix}.run.log"
    csv_path  = f"{args.out_prefix}.partial.csv"
    xlsx_path = f"{args.out_prefix}.final.xlsx"
    logger    = setup_logger(log_path)

    logger.info("=" * 70)
    logger.info("LiveCodeBench — async uncertainty pipeline")
    logger.info(f"  model={args.model}  base_url={args.base_url}")
    logger.info(f"  difficulty={args.difficulty}  n_completions={args.n_completions}  concurrency={args.concurrency}")
    logger.info(f"  exec_workers={args.exec_workers}  no_execute={args.no_execute}")
    logger.info(f"  max_tokens={args.max_tokens}  temp={args.temperature}")
    logger.info(f"  windows={args.windows}  vocab_size={args.vocab_size}")
    logger.info(f"  thinking_mode={args.thinking_mode}  enable_thinking={args.enable_thinking}")
    logger.info("=" * 70)

    if args.n_completions > 1 and args.temperature <= 0.0:
        logger.warning(
            "n_completions>1 with temperature=0.0 gives identical completions — "
            "self-consistency will be trivially 1.0. Use temperature >= 0.6."
        )
    if args.no_execute:
        logger.warning(
            "Code execution disabled (--no_execute). "
            "correct=NaN for all rows. Cannot use for failure prediction."
        )

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    logger.info(f"Loading LiveCodeBench (difficulties={args.difficulty}) ...")
    examples = load_livecodebench_examples(
        difficulties=args.difficulty,
        jsonl_files=args.jsonl_files,
        min_date=args.min_date,
    )
    if args.n > 0:
        examples = examples[: args.n]
    n = len(examples)
    total_calls = n * args.n_completions
    logger.info(f"Loaded {n} problems × {args.n_completions} completions = {total_calls} total calls.")

    diff_counts = Counter(ex["difficulty"] for ex in examples)
    for diff, cnt in sorted(diff_counts.items()):
        logger.info(f"  {diff}: {cnt} problems")

    # Resume from checkpoint — key is "question_id::comp_idx"
    rows: List[Dict[str, Any]] = []
    completed_keys: set = set()
    if os.path.exists(csv_path):
        try:
            existing = pd.read_csv(csv_path)
            if len(existing) > 0 and "checkpoint_key" in existing.columns:
                rows = existing.to_dict(orient="records")
                completed_keys = set(existing["checkpoint_key"].astype(str).tolist())
                logger.info(f"Resuming: {len(rows)} rows already done.")
            elif len(existing) > 0 and "question_id" in existing.columns:
                # Backwards compat: single-completion run without checkpoint_key
                rows = existing.to_dict(orient="records")
                completed_keys = {f"{r['question_id']}::0" for r in rows}
                logger.info(f"Resuming from single-completion checkpoint: {len(rows)} rows.")
        except Exception as e:
            logger.warning(f"Could not load checkpoint ({e}) — starting fresh.")

    tasks_to_run = [
        (i, ex, comp_idx)
        for i, ex in enumerate(examples)
        for comp_idx in range(args.n_completions)
        if f"{ex['question_id']}::{comp_idx}" not in completed_keys
    ]
    logger.info(f"Tasks remaining: {len(tasks_to_run)} / {total_calls}")

    sem       = asyncio.Semaphore(args.concurrency)
    rows_lock = asyncio.Lock()
    executor  = ThreadPoolExecutor(max_workers=args.exec_workers)
    start_time = time.time()
    completed_count = 0

    async def run_and_checkpoint(i: int, ex: Dict[str, Any], comp_idx: int):
        nonlocal completed_count
        row = await process_one(
            sem, executor, client, args, logger, i, n, ex, start_time, comp_idx
        )
        if row is not None:
            async with rows_lock:
                rows.append(row)
                completed_count += 1
                if completed_count % args.save_every == 0:
                    pd.DataFrame(rows).to_csv(csv_path, index=False)
                    logger.info(
                        f"Checkpoint saved: {csv_path} "
                        f"(rows={len(rows)}, completed={completed_count}/{len(tasks_to_run)})"
                    )

    await asyncio.gather(*[
        run_and_checkpoint(i, ex, comp_idx)
        for i, ex, comp_idx in tasks_to_run
    ])
    executor.shutdown(wait=False)

    # Final save
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"All tasks done. Saved {len(rows)} rows to {csv_path}")

    df = pd.DataFrame(rows)

    # Self-consistency columns (only meaningful when n_completions > 1)
    if args.n_completions > 1:
        logger.info("Computing self-consistency columns ...")
        def sc_for_group(g):
            codes = g["extracted_code"].dropna().tolist()
            corrects = g["correct"].dropna().tolist()
            if not corrects:
                g["sc_correct"] = float("nan")
                g["sc_agreement"] = float("nan")
                return g
            # sc_correct: majority vote on correctness (pass=1, fail=0)
            # For code, we don't have a canonical "answer string" to vote on,
            # so we use pass-rate: sc_correct=1 if majority of completions pass
            pass_count = sum(1 for c in corrects if c == 1)
            total = len(corrects)
            sc_correct = int(pass_count > total / 2)
            g["sc_correct"] = sc_correct
            g["sc_agreement"] = max(pass_count, total - pass_count) / total
            return g
        df = df.groupby("question_id", group_keys=False).apply(sc_for_group)

        sc_acc  = df.drop_duplicates("question_id")["sc_correct"].dropna().mean()
        ind_acc = df["correct"].dropna().mean()
        logger.info(f"Individual accuracy:       {ind_acc:.3f}")
        logger.info(f"Self-consistency accuracy: {sc_acc:.3f}  (majority over {args.n_completions} completions)")

    df.to_excel(xlsx_path, index=False)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved final XLSX: {xlsx_path}")

    # Summary
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info(f"  Problems run: {df['question_id'].nunique()}  Total rows: {len(df)}")
    for diff in sorted(df["difficulty"].unique()):
        sub   = df[df["difficulty"] == diff]
        valid = sub["correct"].dropna()
        if len(valid) > 0:
            acc = valid.mean()
            logger.info(
                f"  {diff}: n={sub['question_id'].nunique()} problems  "
                f"rows={len(sub)}  pass_rate={100*acc:.1f}%"
            )
        else:
            logger.info(f"  {diff}: n={sub['question_id'].nunique()} problems  (no execution results)")
    no_code = int((df["extraction_method"] == "none").sum())
    logger.info(f"  No code extracted: {no_code} ({100*no_code/max(len(df),1):.1f}%)")
    logger.info("  Extraction breakdown:")
    for method, cnt in df["extraction_method"].value_counts().items():
        logger.info(f"    {method}: {cnt}")
    logger.info(f"  Mean generated_len: {df['generated_len'].mean():.0f} tokens")
    logger.info(f"  Mean prefinal_len:  {df['prefinal_len'].mean():.0f} tokens")
    logger.info(f"  Output: {xlsx_path}")
    logger.info("=" * 70)
    logger.info("Done.")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()