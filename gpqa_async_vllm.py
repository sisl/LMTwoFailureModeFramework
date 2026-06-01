#!/usr/bin/env python3
"""
gpqa_vllm_multi.py  —  GPQA Diamond with N completions per question.

Key design decisions vs the single-completion pipeline:
  - One row per (question, completion) — NOT one row per question. but this can be run with n = 1 for single completion
    This means the output CSV has n_questions × n_completions rows (198 × 15 = 2970).
  - Checkpoint key is "question_id::completion_idx" so resuming works correctly
    at completion granularity rather than question granularity.
  - Option order is FIXED across completions for the same question
    (same seed → same shuffle) so answer letter is stable and majority vote
    is meaningful. Variation across completions comes only from temperature.
  - Majority vote and self-consistency columns are computed in post-processing
    (see bottom of file), not during inference, to keep the hot loop simple.
  - All uncertainty signal columns are identical to gpqa_vllm.py so the
    existing analysis scripts (gpqa_stratified_analysis.py etc.) work unchanged
    on the per-completion rows.

Self-consistency post-processing (run automatically at end):
  For each question, across all n_completions:
    - majority_answer:        most common predicted letter
    - majority_correct:       1 if majority_answer == correct_letter
    - agreement_rate:         fraction of completions agreeing with majority
    - n_correct_completions:  how many completions got it right
    - p_correct:              n_correct / n_completions  (problem-level accuracy)

  These columns are added to every row belonging to the same question.
  This lets you directly compare agreement_rate as a failure predictor
  against uncertainty features using the same PR-AUC framework.

  To resume a partial run just re-run with the same --out_prefix;
  completed (question_id, completion_idx) pairs are skipped.
"""

import argparse
import asyncio
import logging
import math
import os
import random
import re
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset
from openai import AsyncOpenAI

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants  (identical to gpqa_vllm.py)
# ---------------------------------------------------------------------------

ABLATION_WINDOWS: List[int] = [128, 256, 400, 512, 1024, 2048]

DEFAULT_SYSTEM = (
    "You are a careful reasoning assistant. "
    "Think through the problem step by step, then give your final answer.\n\n"
    "The question is multiple choice. At the end of your response, output exactly:\n"
    "Final Answer: <letter>\n\n"
    "Where <letter> is one of A, B, C, or D.\n\n"
    "Rules:\n"
    "- Only use 'Final Answer:' once, on the last line.\n"
    "- The final answer must be a single letter: A, B, C, or D."
)

DEFAULT_VOCAB_SIZE = 200


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("gpqa_multi")
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
# Math helpers  (identical to gpqa_vllm.py)
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
# Uncertainty signals  (identical to gpqa_vllm.py)
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

    ent = safe_stats(entropies)
    mar = safe_stats(margins)
    nll = safe_stats(nlls)
    nuc = safe_stats(nucleus_sizes)
    tie = safe_stats(near_ties)
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
# GPQA loading  
# ---------------------------------------------------------------------------

def _assign_option_letters(
    example: Dict[str, Any], rng: random.Random
) -> Tuple[Dict[str, str], str]:
    correct_text = str(example.get("Correct Answer", "")).strip()
    inc1 = str(example.get("Incorrect Answer 1", "")).strip()
    inc2 = str(example.get("Incorrect Answer 2", "")).strip()
    inc3 = str(example.get("Incorrect Answer 3", "")).strip()
    texts = [correct_text, inc1, inc2, inc3]
    rng.shuffle(texts)
    letters = ["A", "B", "C", "D"]
    options = {letter: text for letter, text in zip(letters, texts)}
    correct_letter = letters[texts.index(correct_text)]
    return options, correct_letter


def _format_question(question: str, options: Dict[str, str]) -> str:
    option_text = "\n".join(f"({k}) {v}" for k, v in sorted(options.items()))
    return f"{question}\n\n{option_text}"


def load_gpqa_examples(seed: int = 42) -> List[Dict[str, Any]]:
    ds = load_dataset(
        "Idavidrein/gpqa", "gpqa_diamond", split="train", trust_remote_code=True
    )
    examples = []
    for i, ex in enumerate(ds):
        raw_question  = str(ex.get("Question", "")).strip()
        subject       = str(ex.get("Subdomain", ex.get("Domain", "unknown"))).strip()
        question_id   = str(ex.get("Record ID", i))
        per_q_seed    = seed ^ (hash(question_id) & 0xFFFFFFFF)
        rng           = random.Random(per_q_seed)
        options, correct_letter = _assign_option_letters(ex, rng)
        formatted     = _format_question(raw_question, options)
        examples.append({
            "question_id":    question_id,
            "subject":        subject,
            "raw_question":   raw_question,
            "question":       formatted,
            "options":        options,
            "correct_letter": correct_letter,
            "correct_text":   options[correct_letter],
        })
    return examples


# ---------------------------------------------------------------------------
# Answer extraction  
# ---------------------------------------------------------------------------

def extract_letter_answer(text: str) -> Tuple[Optional[str], str]:
    m = re.search(
        r"Final\s+Answer\s*:\s*\(?\s*([A-Da-d])\s*\)?", text, re.IGNORECASE
    )
    if m:
        return m.group(1).upper(), "final_answer"
    m = re.search(
        r"(?:the\s+answer\s+is|answer\s*[=:])\s*\(?\s*([A-Da-d])\s*\)?",
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).upper(), "phrase"
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    for line in reversed(lines):
        m = re.fullmatch(r"\(?([A-Da-d])\)?\.?", line)
        if m:
            return m.group(1).upper(), "last_line_letter"
    matches = re.findall(r"\(([A-Da-d])\)", text)
    if matches:
        return matches[-1].upper(), "bracketed"
    return None, "none"


def is_correct(pred: Optional[str], correct_letter: str) -> bool:
    if pred is None:
        return False
    return pred.strip().upper() == correct_letter.strip().upper()


# ---------------------------------------------------------------------------
# Thinking-model helpers
# ---------------------------------------------------------------------------

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


def find_prefinal_char_cutoff(assistant_text: str) -> Optional[int]:
    m = re.search(r"Final\s+Answer\s*:", assistant_text, re.IGNORECASE)
    if not m:
        return None
    idx        = m.start()
    line_start = assistant_text.rfind("\n", 0, idx)
    return line_start + 1 if line_start != -1 else 0


def detect_degenerate(text: str) -> bool:
    if re.search(r"(.)\1{30,}", text):
        return True
    if re.search(r"\d{20,}", text):
        return True
    return False


# ---------------------------------------------------------------------------
# vLLM call
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
            response_text  = choice.message.content or ""
            thinking_text  = getattr(choice.message, "reasoning_content", None) or ""
            response_logprobs  = []
            thinking_logprobs  = []
            if getattr(choice, "logprobs", None) is not None:
                response_logprobs  = choice.logprobs.content or []
                thinking_logprobs  = getattr(
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
# Self-consistency post-processing
# ---------------------------------------------------------------------------

def add_self_consistency_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each question (grouped by question_id), compute:
      - majority_answer:         most common predicted letter across completions
      - majority_correct:        1 if majority_answer == correct_letter
      - agreement_rate:          fraction of completions matching majority answer
      - n_correct_completions:   count of correct completions
      - p_correct:               n_correct / n_completions

    These are added as new columns on every row belonging to the question,
    enabling direct comparison of agreement_rate vs uncertainty features
    as failure predictors using the same PR-AUC framework.
    """
    sc_rows = []
    for qid, grp in df.groupby("question_id", sort=False):
        preds         = grp["predicted_answer"].fillna("None").tolist()
        correct_letter = grp["correct_letter"].iloc[0]
        n_comp        = len(grp)

        # Majority vote
        counter        = Counter(p for p in preds if p != "None")
        if counter:
            majority_answer  = counter.most_common(1)[0][0]
            agreement_rate   = counter[majority_answer] / n_comp
        else:
            majority_answer  = None
            agreement_rate   = float("nan")

        majority_correct     = int(
            majority_answer is not None
            and majority_answer.upper() == correct_letter.upper()
        )
        n_correct_completions = int(grp["correct"].sum())
        p_correct             = n_correct_completions / n_comp

        for idx in grp.index:
            sc_rows.append({
                "index":                  idx,
                "majority_answer":        majority_answer,
                "majority_correct":       majority_correct,
                "agreement_rate":         agreement_rate,
                "n_correct_completions":  n_correct_completions,
                "p_correct":              p_correct,
                "n_completions":          n_comp,
            })

    sc_df = pd.DataFrame(sc_rows).set_index("index")
    for col in ["majority_answer", "majority_correct", "agreement_rate",
                "n_correct_completions", "p_correct", "n_completions"]:
        df[col] = sc_df[col]

    return df


def log_self_consistency_summary(
    df: pd.DataFrame, logger: logging.Logger
) -> None:
    """
    Logs a summary of self-consistency results including:
      - Majority vote accuracy vs single-sample accuracy
      - Agreement rate distribution for correct vs incorrect problems
      - Per-subject breakdown
    """
    if "majority_correct" not in df.columns:
        return

    # Collapse to one row per question for question-level stats
    q_df = df.drop_duplicates("question_id")
    n_q  = len(q_df)

    single_acc   = q_df["correct"].mean()          # mean of first completion
    mv_acc       = q_df["majority_correct"].mean()
    p_correct_mean = q_df["p_correct"].mean()

    logger.info("=" * 70)
    logger.info("SELF-CONSISTENCY SUMMARY")
    logger.info(f"  Questions:                {n_q}")
    logger.info(f"  Completions per question: {int(q_df['n_completions'].iloc[0])}")
    logger.info(f"  Single-sample accuracy:   {100*single_acc:.1f}%")
    logger.info(f"  Majority-vote accuracy:   {100*mv_acc:.1f}%")
    logger.info(f"  Mean p(correct):          {100*p_correct_mean:.1f}%")
    logger.info(f"  MV lift over single:      {100*(mv_acc - single_acc):+.1f}pp")

    # Agreement rate distribution: correct vs incorrect problems
    correct_q   = q_df[q_df["majority_correct"] == 1]["agreement_rate"]
    incorrect_q = q_df[q_df["majority_correct"] == 0]["agreement_rate"]
    if len(correct_q) > 0 and len(incorrect_q) > 0:
        logger.info(
            f"  Agreement rate — MV correct:   "
            f"mean={correct_q.mean():.3f}  median={correct_q.median():.3f}"
        )
        logger.info(
            f"  Agreement rate — MV incorrect: "
            f"mean={incorrect_q.mean():.3f}  median={incorrect_q.median():.3f}"
        )

    # Problems where model is always wrong (agreement=1.0, majority_correct=0)
    # — these are the committed failures in self-consistency space
    always_wrong = q_df[
        (q_df["agreement_rate"] >= 0.999) & (q_df["majority_correct"] == 0)
    ]
    logger.info(
        f"  Committed wrong (agreement=1.0, MV incorrect): "
        f"{len(always_wrong)}/{n_q} ({100*len(always_wrong)/n_q:.1f}%)"
    )

    # Per-subject
    logger.info("  Per-subject majority-vote accuracy:")
    for subj, grp in q_df.groupby("subject"):
        mv  = grp["majority_correct"].mean()
        ss  = grp["correct"].mean()
        logger.info(
            f"    {subj:<40} MV={100*mv:.1f}%  SS={100*ss:.1f}%  "
            f"lift={100*(mv-ss):+.1f}pp  n={len(grp)}"
        )

    # Agreement rate as failure predictor (PR-AUC vs uncertainty baseline)
    if SKLEARN_AVAILABLE:
        import numpy as np
        # Use q_df for question-level — one row per question
        y_fail = (q_df["majority_correct"] == 0).astype(int).values
        if y_fail.sum() > 1 and y_fail.sum() < len(y_fail):
            baseline = y_fail.mean()
            # Higher agreement among WRONG answers = worse
            # Use 1 - agreement_rate as score (low agreement = uncertain = more likely wrong)
            # But for committed failures, agreement IS high even when wrong.
            # So we test BOTH directions:
            ar = q_df["agreement_rate"].values
            try:
                auroc_ar = roc_auc_score(y_fail, 1 - ar)   # low agreement → failure
                prauc_ar = average_precision_score(y_fail, 1 - ar)
                logger.info(
                    f"  Self-consistency as failure predictor "
                    f"(1 - agreement_rate):"
                )
                logger.info(
                    f"    AUROC={auroc_ar:.4f}  PR-AUC={prauc_ar:.4f}  "
                    f"Baseline={baseline:.4f}  Lift={prauc_ar-baseline:+.4f}"
                )
                # Also try p_correct directly
                pc = q_df["p_correct"].values
                auroc_pc = roc_auc_score(y_fail, 1 - pc)
                prauc_pc = average_precision_score(y_fail, 1 - pc)
                logger.info(
                    f"  p(correct) across completions as failure predictor:"
                )
                logger.info(
                    f"    AUROC={auroc_pc:.4f}  PR-AUC={prauc_pc:.4f}  "
                    f"Baseline={baseline:.4f}  Lift={prauc_pc-baseline:+.4f}"
                )
            except Exception as e:
                logger.warning(f"  Could not compute SC PR-AUC: {e}")

    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def process_one(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    args,
    logger: logging.Logger,
    q_idx: int,
    n_questions: int,
    ex: Dict[str, Any],
    comp_idx: int,
    start_time: float,
) -> Optional[Dict[str, Any]]:
    """Run one (question, completion) pair under the semaphore and return a row dict."""
    async with sem:
        qid            = ex["question_id"]
        q              = ex["question"]
        correct_letter = ex["correct_letter"]
        subject        = ex["subject"]

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
            logger.error(
                f"[q{q_idx+1}/{n_questions} comp{comp_idx+1}] "
                f"Fatal error after retries: {e} — skipping."
            )
            return None

        if thinking_text:
            assistant_text = f"<think>{thinking_text}</think>\n{response_text}"
        else:
            assistant_text = response_text

        token_logprobs = list(thinking_logprobs) + list(response_logprobs)

        extract_source = response_text if response_text else assistant_text
        pred, extraction_method = extract_letter_answer(extract_source)
        correct = is_correct(pred, correct_letter)

        full_signals = compute_signals_multiwindow(
            token_logprobs,
            windows=args.windows,
            nucleus_p=args.nucleus_p,
            tie_delta=args.tie_delta,
            suffix="",
        )

        cutoff = find_prefinal_char_cutoff(assistant_text)
        if cutoff is not None:
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

        ck = f"{qid}::{comp_idx}"
        row: Dict[str, Any] = {
            "checkpoint_key":    ck,
            "question_id":       qid,
            "completion_idx":    comp_idx,
            "subject":           subject,
            "question":          ex["raw_question"],
            "question_formatted":q,
            "correct_letter":    correct_letter,
            "correct_text":      ex["correct_text"],
            "predicted_answer":  pred,
            "extraction_method": extraction_method,
            "correct":           int(correct),
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

        think_info = (
            f" think={thinking_len} resp={response_len}"
            if args.thinking_mode else ""
        )
        logger.info(
            f"[q{q_idx+1}/{n_questions} c{comp_idx+1}/{args.n_completions}] "
            f"subj={subject[:20]} correct={int(correct)} "
            f"pred={str(pred)} gt={correct_letter} "
            f"gen={generated_len}{think_info} "
            f"prefinal={prefinal_len} deg={deg} "
            f"lat={dt:.1f}s elapsed={elapsed/3600:.2f}h"
        )
        return row


async def async_main() -> None:
    ap = argparse.ArgumentParser(
        description="GPQA Diamond — multi-completion pipeline (async/parallel)"
    )
    ap.add_argument("--base_url",        type=str)
    ap.add_argument("--model",           type=str)
    ap.add_argument("--system",          type=str,   default=DEFAULT_SYSTEM)
    ap.add_argument("--n",               type=int,   default=0,
                    help="Max questions (0 = all 198)")
    ap.add_argument("--n_completions",   type=int,   default=15,
                    help="Completions per question (default: 15)")
    ap.add_argument("--concurrency",     type=int,   default=8,
                    help="Max parallel in-flight requests to vLLM (default: 8)")
    ap.add_argument("--max_tokens",      type=int,   default=8192)
    ap.add_argument("--temperature",     type=float, default=0.6)
    ap.add_argument("--vocab_size",      type=int,   default=DEFAULT_VOCAB_SIZE)
    ap.add_argument("--nucleus_p",       type=float, default=0.7)
    ap.add_argument("--tie_delta",       type=float, default=0.5)
    ap.add_argument("--save_every",      type=int,   default=50,
                    help="Checkpoint after every N completed rows.")
    ap.add_argument("--out_prefix",      type=str,   default="gpqa_multi")
    ap.add_argument("--timeout_s",       type=float, default=1200.0)
    ap.add_argument("--retries",         type=int,   default=2)
    ap.add_argument("--thinking_mode",   action="store_true", default=False)
    ap.add_argument("--enable_thinking", action="store_true", default=False)
    ap.add_argument("--windows",         type=int,   nargs="+", default=ABLATION_WINDOWS)
    ap.add_argument("--seed",            type=int,   default=42)
    args = ap.parse_args()

    if args.temperature <= 0.0:
        print(
            "WARNING: temperature=0 gives identical completions — "
            "self-consistency will be trivially 1.0. Use temperature >= 0.6.",
            file=sys.stderr,
        )

    log_path = f"{args.out_prefix}.run.log"
    logger   = setup_logger(log_path)

    logger.info("=" * 70)
    logger.info("GPQA Diamond — async multi-completion pipeline")
    logger.info(f"  model={args.model}  base_url={args.base_url}")
    logger.info(f"  n_questions={args.n} (0=all)  n_completions={args.n_completions}")
    logger.info(f"  concurrency={args.concurrency}")
    logger.info(f"  max_tokens={args.max_tokens}  temp={args.temperature}")
    logger.info(f"  windows={args.windows}  vocab_size={args.vocab_size}")
    logger.info(f"  thinking_mode={args.thinking_mode}  enable_thinking={args.enable_thinking}")
    logger.info(f"  seed={args.seed}")
    logger.info("=" * 70)

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    logger.info("Loading GPQA Diamond ...")
    examples = load_gpqa_examples(seed=args.seed)
    if args.n > 0:
        examples = examples[: args.n]
    n_questions = len(examples)
    total_calls = n_questions * args.n_completions
    logger.info(f"Loaded {n_questions} questions × {args.n_completions} completions "
                f"= {total_calls} total inference calls.")

    csv_path  = f"{args.out_prefix}.partial.csv"
    xlsx_path = f"{args.out_prefix}.final.xlsx"

    # Resume from checkpoint
    rows: List[Dict[str, Any]] = []
    completed_keys: set = set()
    if os.path.exists(csv_path):
        try:
            existing = pd.read_csv(csv_path)
            if len(existing) > 0 and "checkpoint_key" in existing.columns:
                rows = existing.to_dict(orient="records")
                completed_keys = set(existing["checkpoint_key"].tolist())
                logger.info(
                    f"Resuming: {len(rows)} rows already done "
                    f"({len(completed_keys)} unique (question, completion) pairs)."
                )
        except Exception as e:
            logger.warning(f"Could not load checkpoint ({e}) — starting fresh.")

    # Build task list (skip already-completed)
    tasks_to_run = []
    for q_idx, ex in enumerate(examples):
        for comp_idx in range(args.n_completions):
            ck = f"{ex['question_id']}::{comp_idx}"
            if ck not in completed_keys:
                tasks_to_run.append((q_idx, ex, comp_idx))

    logger.info(f"Tasks remaining: {len(tasks_to_run)} / {total_calls}")

    sem = asyncio.Semaphore(args.concurrency)
    start_time = time.time()
    rows_lock = asyncio.Lock()
    completed_count = 0

    async def run_and_checkpoint(q_idx, ex, comp_idx):
        nonlocal completed_count
        row = await process_one(
            sem, client, args, logger,
            q_idx, n_questions, ex, comp_idx, start_time,
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
        run_and_checkpoint(q_idx, ex, comp_idx)
        for q_idx, ex, comp_idx in tasks_to_run
    ])

    # Final checkpoint before post-processing
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"All tasks done. Saved {len(rows)} rows to {csv_path}")

    # Post-processing: self-consistency columns
    logger.info("Computing self-consistency columns ...")
    df = pd.DataFrame(rows)
    df = add_self_consistency_columns(df)

    df.to_excel(xlsx_path, index=False)
    df.to_csv(csv_path, index=False)
    logger.info(f"Final output saved: {xlsx_path}")

    log_self_consistency_summary(df, logger)
    logger.info("Done.")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()