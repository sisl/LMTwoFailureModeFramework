#!/usr/bin/env python3

import argparse
import asyncio
import logging
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset
from openai import AsyncOpenAI

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False



DEFAULT_SYSTEM = (
    """Reason through the problem step by step to arrive at an answer.

    At the end, output a final line exactly in this format:
    Final: \\boxed{<answer>}

    Rules:
    - Do not use \\boxed{ } except in the final line.
    - Put only the final answer inside the box.
    - The answer may be a number, fraction, expression, or other mathematical object."""
)

DEFAULT_VOCAB_SIZE = 200

# Multi-window ablation sizes — all computed on the SAME trace
DEFAULT_WINDOWS: list = [128, 256, 400, 512, 1024, 2048]


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("gsm8k_vllm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger



def logsumexp(logps: List[float]) -> float:
    m = max(logps)
    return m + math.log(sum(math.exp(lp - m) for lp in logps))


def safe_stats(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": float("nan"), "max": float("nan"), "std": float("nan")}
    mean = sum(xs) / len(xs)
    mx = max(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    return {"mean": float(mean), "max": float(mx), "std": float(math.sqrt(var))}


def detect_degenerate(text: str) -> bool:
    if re.search(r"(.)\1{30,}", text):
        return True
    if re.search(r"\d{20,}", text):
        return True
    return False

def extract_llm_answer(text: str) -> Tuple[Optional[str], str]:
    m = re.search(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    if m:
        return m.group(1).strip(), "boxed"

    tail = text[-300:]
    m = re.search(r"=\s*([^\n=]{1,80})\s*$", tail)
    if m:
        return m.group(1).strip(), "equals"

    m = re.search(
        r"(?:the\s+answer\s+is|answer\s*[:=])\s*([^\n]{1,80})",
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).strip(), "phrase"

    m = re.search(r"\*\*([^*]{1,80})\*\*", text)
    if m:
        return m.group(1).strip(), "bold"

    return None, "none"


def extract_true_target(gt: str) -> str:
    m = re.search(r"####\s*([-+]?\S+)", gt)
    if m:
        return m.group(1).strip()
    return gt.strip()


def _normalise_math_answer(s: str) -> str:
    s = s.strip().rstrip(".")
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\math\w+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_thinking_response(assistant_text: str) -> Tuple[str, str]:
    m = re.search(r"<think>(.*?)</think>(.*)", assistant_text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"(.*?)</think>(.*)", assistant_text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", assistant_text.strip()


def split_token_logprobs_at_think_end(
    token_logprobs: List[Any],
    assistant_text: str,
) -> Tuple[List[Any], List[Any]]:
    m = re.search(r"(?:<think>)?(.*?)</think>", assistant_text, re.DOTALL)
    if not m:
        return [], token_logprobs
    think_end_char = m.end()
    thinking_toks = truncate_token_logprobs_to_char_cutoff(token_logprobs, think_end_char)
    response_toks = token_logprobs[len(thinking_toks):]
    return thinking_toks, response_toks


def is_correct(pred: Optional[str], gt: str) -> bool:
    if pred is None:
        return False
    pred_n = _normalise_math_answer(pred)
    gt_n   = _normalise_math_answer(gt)
    if pred_n == gt_n:
        return True
    def _to_float(s: str) -> Optional[float]:
        try:
            return float(s.replace(",", ""))
        except Exception:
            return None
    pf, gf = _to_float(pred_n), _to_float(gt_n)
    if pf is not None and gf is not None:
        return abs(pf - gf) < 1e-6
    return pred_n.lower() == gt_n.lower()


def build_transcript(system_msg: str, user_msg: str, assistant_msg: str) -> str:
    return f"system\n{system_msg}\n\nuser\n{user_msg}\n\nassistant\n{assistant_msg}\n"


def find_preboxed_char_cutoff(assistant_text: str) -> Optional[int]:
    m = re.search(r"\\boxed\{", assistant_text)
    if not m:
        return None
    idx = m.start()
    line_start = assistant_text.rfind("\n", 0, idx)
    if line_start == -1:
        line_start = 0
    else:
        line_start = line_start + 1
    return line_start


def truncate_token_logprobs_to_char_cutoff(token_logprobs: List[Any], cutoff_char_idx: int) -> List[Any]:
    acc = 0
    out: List[Any] = []
    for tok in token_logprobs:
        t = getattr(tok, "token", "")
        if t is None:
            t = ""
        next_acc = acc + len(t)
        if next_acc <= cutoff_char_idx:
            out.append(tok)
            acc = next_acc
        else:
            break
    return out


def top_logprobs_to_logps(top_logprobs: List[Any]) -> List[float]:
    return [float(x.logprob) for x in top_logprobs]


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
            top2 = top1
            top1 = lp
        elif lp > top2:
            top2 = lp
    lse = logsumexp(vocab_logps)
    p1 = math.exp(top1 - lse)
    p2 = math.exp(top2 - lse) if top2 != -float("inf") else 0.0
    return p1 - p2


def nucleus_size_from_full_vocab_logps(vocab_logps: List[float], nucleus_p: float = 0.9) -> int:
    lse = logsumexp(vocab_logps)
    probs = [math.exp(lp - lse) for lp in vocab_logps]
    probs.sort(reverse=True)
    s = 0.0
    for i, p in enumerate(probs, start=1):
        s += p
        if s >= nucleus_p:
            return i
    return len(probs)


def near_tie_count_from_full_vocab_logps(vocab_logps: List[float], tie_delta: float = 0.1) -> int:
    top1 = max(vocab_logps)
    cnt = 0
    for lp in vocab_logps:
        if lp >= top1 - tie_delta:
            cnt += 1
    return max(0, cnt - 1)


def _compute_signals_from_token_list(
    token_logprobs: List[Any],
    *,
    nucleus_p: float,
    tie_delta: float,
    suffix: str,
) -> Dict[str, float]:
    entropies: List[float] = []
    margins: List[float] = []
    nlls: List[float] = []
    nucleus_sizes: List[float] = []
    near_ties: List[float] = []
    forks: List[int] = []

    for tok in token_logprobs:
        lp_chosen = float(tok.logprob)
        nlls.append(-lp_chosen)
        top = getattr(tok, "top_logprobs", None)
        if not top:
            continue
        vocab_logps = top_logprobs_to_logps(top)
        entropies.append(entropy_from_full_vocab_logps(vocab_logps))
        margins.append(margin_from_full_vocab_logps(vocab_logps))
        ns = nucleus_size_from_full_vocab_logps(vocab_logps, nucleus_p=nucleus_p)
        nt = near_tie_count_from_full_vocab_logps(vocab_logps, tie_delta=tie_delta)
        nucleus_sizes.append(float(ns))
        near_ties.append(float(nt))
        forks.append(1 if nt > 0 else 0)

    ent = safe_stats(entropies)
    mar = safe_stats(margins)
    nll = safe_stats(nlls)
    nuc = safe_stats(nucleus_sizes)
    tie = safe_stats(near_ties)
    fork_rate = float(sum(forks) / len(forks)) if forks else float("nan")

    return {
        f"entropy_mean{suffix}": ent["mean"],
        f"entropy_max{suffix}": ent["max"],
        f"entropy_std{suffix}": ent["std"],
        f"margin_mean{suffix}": mar["mean"],
        f"margin_max{suffix}": mar["max"],
        f"margin_std{suffix}": mar["std"],
        f"nll_mean{suffix}": nll["mean"],
        f"nll_max{suffix}": nll["max"],
        f"nll_std{suffix}": nll["std"],
        f"fork_rate{suffix}": fork_rate,
        f"nucleus_size_mean{suffix}": nuc["mean"],
        f"nucleus_size_max{suffix}": nuc["max"],
        f"near_tie_mean{suffix}": tie["mean"],
        f"near_tie_max{suffix}": tie["max"],
    }


def compute_signals_from_token_logprobs(
    token_logprobs: List[Any],
    *,
    early_T: int = 64,
    nucleus_p: float = 0.9,
    tie_delta: float = 0.1,
    suffix: str = "",
) -> Dict[str, float]:
    full = _compute_signals_from_token_list(
        token_logprobs, nucleus_p=nucleus_p, tie_delta=tie_delta, suffix=suffix
    )
    early = _compute_signals_from_token_list(
        token_logprobs[:early_T], nucleus_p=nucleus_p, tie_delta=tie_delta, suffix=f"{suffix}_early"
    )
    return {f"early_T{suffix}": float(early_T), **full, **early}


def compute_signals_multiwindow(
    token_logprobs: List[Any],
    *,
    windows: List[int],
    nucleus_p: float = 0.9,
    tie_delta: float = 0.1,
    suffix: str = "",
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    result.update(_compute_signals_from_token_list(
        token_logprobs, nucleus_p=nucleus_p, tie_delta=tie_delta, suffix=suffix
    ))
    result[f"full_len{suffix}"] = float(len(token_logprobs))
    for T in windows:
        early = token_logprobs[:T]
        result.update(_compute_signals_from_token_list(
            early, nucleus_p=nucleus_p, tie_delta=tie_delta, suffix=f"{suffix}_w{T}"
        ))
        result[f"window_len{suffix}_w{T}"] = float(len(early))
    return result


def _make_nan_signals_multiwindow(
    windows: List[int],
    suffix: str = "",
    nucleus_p: float = 0.9,
    tie_delta: float = 0.1,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    result.update(_compute_signals_from_token_list(
        [], nucleus_p=nucleus_p, tie_delta=tie_delta, suffix=suffix
    ))
    result[f"full_len{suffix}"] = float("nan")
    for T in windows:
        result.update(_compute_signals_from_token_list(
            [], nucleus_p=nucleus_p, tie_delta=tie_delta, suffix=f"{suffix}_w{T}"
        ))
        result[f"window_len{suffix}_w{T}"] = float("nan")
    return result


def _make_nan_signals(suffix: str, early_T: float) -> Dict[str, float]:
    template = _compute_signals_from_token_list(
        [], nucleus_p=0.9, tie_delta=0.1, suffix=suffix
    )
    early_template = _compute_signals_from_token_list(
        [], nucleus_p=0.9, tie_delta=0.1, suffix=f"{suffix}_early"
    )
    return {f"early_T{suffix}": early_T, **template, **early_template}


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
        {"role": "user", "content": user_msg},
    ]

    extra_body = {}
    if enable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = await client.chat.completions.create(
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
            response_text = choice.message.content or ""
            thinking_text = getattr(choice.message, "reasoning_content", None) or ""
            response_logprobs: List[Any] = []
            thinking_logprobs: List[Any] = []
            if getattr(choice, "logprobs", None) is not None and choice.logprobs is not None:
                response_logprobs = choice.logprobs.content or []
                thinking_logprobs = (
                    getattr(choice.logprobs, "reasoning_content", None) or []
                )
            finish_reason = getattr(choice, "finish_reason", None)
            hit_max = bool(finish_reason == "length")
            return thinking_text, response_text, thinking_logprobs, response_logprobs, hit_max, finish_reason

        except Exception as e:
            last_err = e
            logger.warning(f"vLLM call failed (attempt {attempt+1}/{retries+1}): {e}")
            await asyncio.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"vLLM call failed after retries: {last_err}")

#do not rely on these metrics, use analyze_updated_dataset_agnostic.py to analyze
def compute_evaluation_metrics(
    rows: List[Dict[str, Any]],
    logger: logging.Logger,
) -> None:
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not found")
        return

    import numpy as np

    labels = [r["correct"] for r in rows]
    if len(set(labels)) < 2:
        logger.warning("Only one class present in labels — skipping evaluation metrics.")
        return

    y_true = np.array(labels, dtype=int)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    logger.info(f"Class distribution: correct={n_pos} ({100*n_pos/len(y_true):.1f}%), incorrect={n_neg} ({100*n_neg/len(y_true):.1f}%)")

    def _safe_get(row: Dict, key: str) -> float:
        v = row.get(key, float("nan"))
        return float(v) if v is not None else float("nan")

    entropy_scores = np.array([_safe_get(r, "entropy_mean_early") for r in rows])
    margin_scores  = np.array([_safe_get(r, "margin_mean_early") for r in rows])
    nll_scores     = np.array([_safe_get(r, "nll_mean_early") for r in rows])
    length_scores  = np.array([float(r.get("generated_len", 0)) for r in rows])

    def _normalise(arr: "np.ndarray") -> "np.ndarray":
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0 or finite.max() == finite.min():
            return np.zeros_like(arr)
        return (arr - finite.min()) / (finite.max() - finite.min())

    composite = (
        _normalise(entropy_scores)
        - _normalise(margin_scores)
        + _normalise(nll_scores)
    ) / 3.0

    valid_mask = np.isfinite(composite)
    if valid_mask.sum() < 2:
        logger.warning("Too many NaN values in features — skipping evaluation metrics.")
        return

    y_valid = y_true[valid_mask]
    score_valid = composite[valid_mask]
    length_valid = length_scores[valid_mask]

    auroc_uncertainty = roc_auc_score(y_valid, score_valid)
    auroc_length      = roc_auc_score(y_valid, _normalise(length_valid))
    prauc_uncertainty = average_precision_score(y_valid, score_valid)
    prauc_length      = average_precision_score(y_valid, _normalise(length_valid))
    score_prob = _normalise(score_valid)
    brier = brier_score_loss(y_valid, score_prob)

    logger.info("=" * 60)
    logger.info("The following is incorrect, run analyze_data for true predictive scores!")
    logger.info("EVALUATION METRICS (composite uncertainty score, early window)")
    logger.info(f"  AUROC       (uncertainty): {auroc_uncertainty:.4f}")
    logger.info(f"  AUROC       (length only): {auroc_length:.4f}")
    logger.info(f"  PR-AUC      (uncertainty): {prauc_uncertainty:.4f}  ← preferred under class imbalance")
    logger.info(f"  PR-AUC      (length only): {prauc_length:.4f}")
    logger.info(f"  Brier Score (uncertainty): {brier:.4f}  ← calibration (lower=better)")
    logger.info(f"  Baseline PR-AUC (random) : {n_pos/len(y_true):.4f}  ← positive class rate")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Async worker: one (idx, example) pair under semaphore
# ---------------------------------------------------------------------------

async def process_one(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    args,
    logger: logging.Logger,
    i: int,
    n: int,
    ex: Any,
    start_time: float,
    comp_idx: int = 0,
) -> Optional[Dict[str, Any]]:
    async with sem:
        q = ex["problem"]
        gt = extract_true_target(ex["answer"])
        level = int(ex.get("level", 0))
        subject = str(ex.get("subject", ""))

        t0 = time.time()
        try:
            thinking_text, response_text, thinking_logprobs, response_logprobs, hit_max_tokens, finish_reason = await call_vllm_chat(
                client,
                model=args.model,
                system_msg=args.system,
                user_msg=q,
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
        full_text = build_transcript(args.system, q, assistant_text)

        extract_source = response_text if response_text else assistant_text
        boxed, extraction_method = extract_llm_answer(extract_source)
        correct = is_correct(boxed, gt)

        full_metrics = compute_signals_multiwindow(
            token_logprobs,
            windows=args.windows,
            nucleus_p=args.nucleus_p,
            tie_delta=args.tie_delta,
            suffix="",
        )

        cutoff = find_preboxed_char_cutoff(assistant_text)
        if cutoff is not None:
            preboxed_tok = truncate_token_logprobs_to_char_cutoff(token_logprobs, cutoff)
            preboxed_metrics = compute_signals_multiwindow(
                preboxed_tok,
                windows=args.windows,
                nucleus_p=args.nucleus_p,
                tie_delta=args.tie_delta,
                suffix="_prefinal",
            )
            preboxed_len = len(preboxed_tok)
        else:
            preboxed_metrics = _make_nan_signals_multiwindow(
                args.windows, suffix="_prefinal",
                nucleus_p=args.nucleus_p, tie_delta=args.tie_delta,
            )
            preboxed_len = 0

        if args.thinking_mode:
            thinking_len = len(thinking_logprobs)
            response_len = len(response_logprobs)
            thinking_metrics = (
                compute_signals_multiwindow(
                    list(thinking_logprobs),
                    windows=args.windows,
                    nucleus_p=args.nucleus_p,
                    tie_delta=args.tie_delta,
                    suffix="_thinking",
                ) if thinking_logprobs else
                _make_nan_signals_multiwindow(
                    args.windows, suffix="_thinking",
                    nucleus_p=args.nucleus_p, tie_delta=args.tie_delta,
                )
            )
            response_metrics = (
                compute_signals_multiwindow(
                    list(response_logprobs),
                    windows=args.windows,
                    nucleus_p=args.nucleus_p,
                    tie_delta=args.tie_delta,
                    suffix="_response",
                ) if response_logprobs else
                _make_nan_signals_multiwindow(
                    args.windows, suffix="_response",
                    nucleus_p=args.nucleus_p, tie_delta=args.tie_delta,
                )
            )
        else:
            thinking_len = 0
            response_len = len(token_logprobs)
            thinking_metrics = {}
            response_metrics = {}

        generated_len = len(token_logprobs)
        deg = int(detect_degenerate(assistant_text))
        dt = time.time() - t0
        elapsed = time.time() - start_time

        think_info = f" think_len={thinking_len} resp_len={response_len}" if args.thinking_mode else ""
        logger.info(
            f"[{i+1}/{n} c{comp_idx+1}/{args.n_completions}] level={level} subj={subject[:12]} correct={int(correct)} "
            f"boxed={boxed} extract={extraction_method} "
            f"gen_len={generated_len}{think_info} prefinal_len={preboxed_len} "
            f"hit_max={int(bool(hit_max_tokens))} deg={deg} "
            f"latency={dt:.2f}s elapsed={elapsed/3600:.2f}h"
        )

        return {
            "checkpoint_key": f"{i}::{comp_idx}",
            "idx": i,
            "completion_idx": comp_idx,
            "level": level,
            "subject": subject,
            "question": q,
            "full_text": full_text,
            "assistant_output": assistant_text,
            "thinking_text": thinking_text,
            "response_text": response_text,
            "boxed_answer": boxed,
            "extraction_method": extraction_method,
            "true_answer": gt,
            "correct": int(correct),
            "generated_len": int(generated_len),
            "thinking_len": int(thinking_len),
            "response_len": int(response_len),
            "prefinal_len": int(preboxed_len),
            "preboxed_generated_len": int(preboxed_len),
            "hit_max_tokens": int(bool(hit_max_tokens)),
            "finish_reason": str(finish_reason) if finish_reason is not None else "",
            "degenerate": int(deg),
            "latency_s": float(dt),
            **full_metrics,
            **preboxed_metrics,
            **thinking_metrics,
            **response_metrics,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", type=str)
    ap.add_argument("--model", type=str)
    ap.add_argument("--system", type=str, default=DEFAULT_SYSTEM)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n_completions", type=int, default=1,
                    help="Completions per problem (default: 1; set to 15 for self-consistency)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="Max parallel in-flight requests to vLLM (default: 8)")
    ap.add_argument("--max_tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    ap.add_argument("--early_T", type=int, default=256)
    ap.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE)
    ap.add_argument("--nucleus_p", type=float, default=0.7)
    ap.add_argument("--tie_delta", type=float, default=0.5)
    ap.add_argument("--save_every", type=int, default=50,
                    help="Checkpoint after every N completed rows.")
    ap.add_argument("--out_prefix", type=str, default="math500_vllm")
    ap.add_argument("--timeout_s", type=float, default=1200.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--levels", type=int, nargs="+", default=None)
    ap.add_argument("--thinking_mode", action="store_true", default=False)
    ap.add_argument("--enable_thinking", action="store_true", default=False)
    args = ap.parse_args()

    log_path = f"{args.out_prefix}.run.log"
    logger = setup_logger(log_path)

    logger.info("Starting MATH-500 async multi-window vLLM pipeline")
    logger.info(f"base_url={args.base_url}  model={args.model}")
    logger.info(f"n={args.n}  n_completions={args.n_completions}  concurrency={args.concurrency}  max_tokens={args.max_tokens}  temp={args.temperature}")
    logger.info(f"levels={args.levels if args.levels else 'all'}")
    logger.info(f"windows={args.windows}  vocab_size={args.vocab_size}")
    logger.info(f"nucleus_p={args.nucleus_p}  tie_delta={args.tie_delta}")
    logger.info(f"thinking_mode={args.thinking_mode}  enable_thinking={args.enable_thinking}")
    logger.info(f"save_every={args.save_every}  timeout_s={args.timeout_s}  retries={args.retries}")

    if args.n_completions > 1 and args.temperature <= 0.0:
        logger.warning(
            "n_completions>1 with temperature=0.0 gives identical completions — "
            "self-consistency will be trivially 1.0. Use temperature >= 0.6."
        )
    if args.thinking_mode and not args.enable_thinking:
        logger.warning(
            "thinking_mode=True but enable_thinking=False — the model may not generate "
            "a thinking block, so _thinking columns will be NaN."
        )
    if args.thinking_mode and args.temperature == 0.0:
        logger.warning(
            "thinking_mode=True with temperature=0.0. Qwen3 thinking models recommend "
            "temperature=0.6 — greedy decoding can cause degradation."
        )
    if args.thinking_mode and args.max_tokens < 8192:
        logger.warning(
            f"thinking_mode=True with max_tokens={args.max_tokens}. "
            "Consider --max_tokens 32768."
        )

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if args.levels:
        ds = ds.filter(lambda ex: ex["level"] in args.levels)
        logger.info(f"Filtered to levels {args.levels}: {len(ds)} problems remaining")
    n = min(args.n, len(ds))
    total_calls = n * args.n_completions
    logger.info(f"Loaded MATH-500 test split: {len(ds)} problems total; running n={n} × {args.n_completions} completions = {total_calls} calls")

    csv_path  = f"{args.out_prefix}.partial.csv"
    xlsx_path = f"{args.out_prefix}.final.xlsx"

    # Resume from checkpoint — key is "idx::comp_idx"
    rows: List[Dict[str, Any]] = []
    completed_keys: set = set()
    if os.path.exists(csv_path):
        try:
            existing = pd.read_csv(csv_path)
            if len(existing) > 0 and "checkpoint_key" in existing.columns:
                rows = existing.to_dict(orient="records")
                completed_keys = set(existing["checkpoint_key"].astype(str).tolist())
                logger.info(
                    f"Resuming from checkpoint: {len(rows)} completed rows. "
                    f"Skipping already completed (idx, completion) pairs."
                )
            elif len(existing) > 0 and "idx" in existing.columns:
                # Backwards compat: single-completion run without checkpoint_key
                rows = existing.to_dict(orient="records")
                completed_keys = {f"{int(r['idx'])}::0" for r in rows}
                logger.info(
                    f"Resuming from single-completion checkpoint: {len(rows)} rows."
                )
        except Exception as e:
            logger.warning(f"Could not load checkpoint CSV ({e}) — starting fresh.")

    tasks_to_run = [
        (i, comp_idx)
        for i in range(n)
        for comp_idx in range(args.n_completions)
        if f"{i}::{comp_idx}" not in completed_keys
    ]
    logger.info(f"Tasks remaining: {len(tasks_to_run)} / {total_calls}")

    sem = asyncio.Semaphore(args.concurrency)
    start_time = time.time()
    rows_lock = asyncio.Lock()
    completed_count = 0

    async def run_and_checkpoint(i: int, comp_idx: int):
        nonlocal completed_count
        row = await process_one(sem, client, args, logger, i, n, ds[i], start_time, comp_idx)
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

    await asyncio.gather(*[run_and_checkpoint(i, comp_idx) for i, comp_idx in tasks_to_run])

    # Final save
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"All tasks done. Saved {len(rows)} rows to {csv_path}")

    df = pd.DataFrame(rows)

    # Self-consistency columns (only meaningful when n_completions > 1)
    if args.n_completions > 1:
        logger.info("Computing self-consistency columns ...")
        def sc_for_group(g):
            answers = g["boxed_answer"].dropna().tolist()
            if not answers:
                g["sc_majority_answer"] = None
                g["sc_correct"] = float("nan")
                g["sc_agreement"] = float("nan")
                return g
            from collections import Counter
            counts = Counter(answers)
            majority, majority_count = counts.most_common(1)[0]
            # sc_correct: majority answer matches true answer
            true_ans = g["true_answer"].iloc[0]
            sc_correct = int(is_correct(majority, true_ans))
            g["sc_majority_answer"] = majority
            g["sc_correct"] = sc_correct
            g["sc_agreement"] = majority_count / len(answers)
            return g
        df = df.groupby("idx", group_keys=False).apply(sc_for_group)

        # Summary
        sc_acc = df.drop_duplicates("idx")["sc_correct"].dropna().mean()
        ind_acc = df["correct"].mean()
        logger.info(f"Individual accuracy:      {ind_acc:.3f}")
        logger.info(f"Self-consistency accuracy: {sc_acc:.3f}  (majority over {args.n_completions} completions)")

    df.to_excel(xlsx_path, index=False)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved final XLSX: {xlsx_path}")

    if rows:
        method_counts: Dict[str, int] = {}
        for r in rows:
            m = r.get("extraction_method", "unknown")
            method_counts[m] = method_counts.get(m, 0) + 1
        logger.info("Answer extraction breakdown:")
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {method}: {count} ({100*count/len(rows):.1f}%)")

    compute_evaluation_metrics(rows, logger)
    logger.info("Done.")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()