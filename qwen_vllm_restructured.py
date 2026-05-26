#!/usr/bin/env python3
"""
gsm8k_vllm_full_pipeline.py

Runs GSM8K through a vLLM OpenAI-compatible endpoint, collects assistant outputs,
FULL-vocab logprobs (top_logprobs=vocab_size), computes uncertainty signals + forking
signals, and computes both:
  (a) full assistant generation metrics
  (b) preboxed metrics (truncate BEFORE the line containing \boxed{...})

Also computes end-of-run evaluation metrics: AUROC, PR-AUC, Brier score.

Answer extraction uses a cascade of strategies:
  1. \boxed{...} (primary)
  2. "= <number>" at end of response (fallback)
  3. "the answer is <number>" (fallback)
  4. Bold **<number>** formatting (fallback)

Outputs:
  - <out_prefix>.run.log          (progress log)
  - <out_prefix>.partial.csv      (incremental checkpoints)
  - <out_prefix>.final.xlsx       (final spreadsheet)

Usage:
  python gsm8k_vllm_full_pipeline.py \
    --base_url "http://machine.stanford.edu:PORT/v1" \
    --model "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    --split test \
    --n 200 \
    --max_tokens 512 \
    --temperature 0.0 \
    --early_T 64 \
    --vocab_size 151936 \
    --save_every 20
"""

import argparse
import logging
import math
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset
from openai import OpenAI

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ----------------------------
# Prompt
# ----------------------------

DEFAULT_SYSTEM = (
    """Reason through the question step by step to arrive at an answer.

    At the end, output a final line exactly in this format:
    Final: \\boxed{<number>}

    Rules:
    - Do not use \\boxed{ } except in the final line.
    - Put only the final numeric answer inside the box."""
)

DEFAULT_VOCAB_SIZE = 151_936


# ----------------------------
# Logging
# ----------------------------

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


# ----------------------------
# Utility
# ----------------------------

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
    # long repeated characters or digit spam
    if re.search(r"(.)\1{30,}", text):
        return True
    if re.search(r"\d{20,}", text):
        return True
    return False


def extract_llm_answer(text: str) -> Tuple[Optional[str], str]:
    """
    Extract the final answer from model output using a cascade of strategies.

    Returns:
        (answer, extraction_method) where extraction_method is one of:
            "boxed"   - matched \\boxed{...}
            "equals"  - matched "= <number>" near end of response
            "phrase"  - matched "the answer is <number>"
            "bold"    - matched **<number>**
            "none"    - no answer found
    """
    # Strategy 1: boxed answer (primary)
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip(), "boxed"

    # Strategy 2: "= <number>" at end of response (last 300 chars to avoid mid-text false positives)
    tail = text[-300:]
    m = re.search(r"=\s*([-+]?\d[\d,]*\.?\d*)\s*$", tail)
    if m:
        return m.group(1).strip(), "equals"

    # Strategy 3: "the answer is <number>"
    m = re.search(r"(?:the\s+answer\s+is|answer\s*[:=])\s*([-+]?\d[\d,]*\.?\d*)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip(), "phrase"

    # Strategy 4: bold **<number>**
    m = re.search(r"\*\*([-+]?\d[\d,]*\.?\d*)\*\*", text)
    if m:
        return m.group(1).strip(), "bold"

    return None, "none"


def extract_true_target(gt_full: str) -> str:
    m = re.search(r"####\s*([-+]?\d+)", gt_full)
    return m.group(1).strip() if m else gt_full.strip()


def is_correct(pred: Optional[str], gt: str) -> bool:
    if pred is None:
        return False
    try:
        return int(pred.replace(",", "")) == int(gt.replace(",", ""))
    except Exception:
        return pred.strip() == gt.strip()


def build_transcript(system_msg: str, user_msg: str, assistant_msg: str) -> str:
    return f"system\n{system_msg}\n\nuser\n{user_msg}\n\nassistant\n{assistant_msg}\n"


# ----------------------------
# Preboxed truncation helpers
# ----------------------------

def find_preboxed_char_cutoff(assistant_text: str) -> Optional[int]:
    """
    Returns a character index cutoff such that assistant_text[:cutoff]
    excludes the entire line that contains the final \boxed{...}.

    If no boxed answer exists, returns None.
    """
    m = re.search(r"\\boxed\{", assistant_text)
    if not m:
        return None

    # Find the start of the line containing the boxed answer
    idx = m.start()
    line_start = assistant_text.rfind("\n", 0, idx)
    if line_start == -1:
        line_start = 0
    else:
        line_start = line_start + 1  # char after '\n'

    # We want to exclude that whole line and anything after it:
    return line_start


def truncate_token_logprobs_to_char_cutoff(token_logprobs: List[Any], cutoff_char_idx: int) -> List[Any]:
    """
    token_logprobs is a list of ChatCompletionTokenLogprob for generated tokens.
    We truncate to align with a character cutoff in the concatenated token strings.

    We assume tok.token is the text token (already decoded piece).
    This is not perfect in all tokenization edge cases, but works well in practice
    for vLLM/OpenAI token strings.
    """
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


# ----------------------------
# Full-vocab metrics per token
# ----------------------------

def top_logprobs_to_logps(top_logprobs: List[Any]) -> List[float]:
    return [float(x.logprob) for x in top_logprobs]


def entropy_from_full_vocab_logps(vocab_logps: List[float]) -> float:
    lse = logsumexp(vocab_logps)
    # H = -sum p log p ; p = exp(lp-lse)
    # compute in prob space; full vocab means big list; this is heavy but exact
    ent = 0.0
    for lp in vocab_logps:
        p = math.exp(lp - lse)
        if p > 0.0:
            ent -= p * math.log(p)
    return ent


def margin_from_full_vocab_logps(vocab_logps: List[float]) -> float:
    # find top1/top2 logps
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
    """
    Nucleus size = smallest k such that sum_{i=1..k} p_i >= nucleus_p.
    """
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
    """
    Near-tie count: number of alternative tokens whose logprob is within tie_delta
    of the top-1 token logprob (excluding the top token itself).

    tie_delta in NATURAL LOG units.
    Example: tie_delta=0.1 means within exp(-0.1) ~ 0.90x of top token prob in ratio terms.
    """
    top1 = max(vocab_logps)
    # count tokens with lp >= top1 - tie_delta, excluding those equal to top1 (top itself; but ties can happen)
    cnt = 0
    for lp in vocab_logps:
        if lp >= top1 - tie_delta:
            cnt += 1
    # exclude one for the top token (approx)
    return max(0, cnt - 1)





def _compute_signals_from_token_list(
    token_logprobs: List[Any],
    *,
    nucleus_p: float,
    tie_delta: float,
    suffix: str,
) -> Dict[str, float]:
    """
    Core signal computation over an arbitrary token list.
    Shared by both full-window and early-window computations.
    """
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
    """
    Computes entropy, margin, nll, fork_rate, nucleus_size, near_tie
    over the full token list and over the first early_T tokens.

    Returns dict with keys suffixed by `suffix` and `{suffix}_early`.
    """
    full = _compute_signals_from_token_list(
        token_logprobs, nucleus_p=nucleus_p, tie_delta=tie_delta, suffix=suffix
    )
    early = _compute_signals_from_token_list(
        token_logprobs[:early_T], nucleus_p=nucleus_p, tie_delta=tie_delta, suffix=f"{suffix}_early"
    )
    return {f"early_T{suffix}": float(early_T), **full, **early}


def _make_nan_signals(suffix: str, early_T: float) -> Dict[str, float]:
    """
    Returns a dict of NaN signal values with the given suffix, derived
    dynamically from _compute_signals_from_token_list key names.
    Avoids hardcoding keys that can drift out of sync when features change.
    """
    template = _compute_signals_from_token_list(
        [], nucleus_p=0.9, tie_delta=0.1, suffix=suffix
    )
    early_template = _compute_signals_from_token_list(
        [], nucleus_p=0.9, tie_delta=0.1, suffix=f"{suffix}_early"
    )
    return {f"early_T{suffix}": early_T, **template, **early_template}


# ----------------------------
# vLLM call
# ----------------------------

def call_vllm_chat(
    client: OpenAI,
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
) -> Tuple[str, List[Any], bool, Optional[str]]:
    """
    Returns:
      assistant_text,
      token_logprobs (generated tokens),
      hit_max_tokens (best effort),
      finish_reason (if provided)
    """
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=True,
                top_logprobs=vocab_size,
                timeout=timeout_s,
            )
            choice = resp.choices[0]
            assistant_text = choice.message.content or ""

            token_logprobs = []
            if getattr(choice, "logprobs", None) is not None and choice.logprobs is not None:
                token_logprobs = choice.logprobs.content or []

            finish_reason = getattr(choice, "finish_reason", None)
            hit_max = bool(finish_reason == "length")

            return assistant_text, token_logprobs, hit_max, finish_reason

        except Exception as e:
            last_err = e
            logger.warning(f"vLLM call failed (attempt {attempt+1}/{retries+1}): {e}")
            time.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"vLLM call failed after retries: {last_err}")


# ----------------------------
# Evaluation metrics (Step 1)
# ----------------------------

def compute_evaluation_metrics(
    rows: List[Dict[str, Any]],
    logger: logging.Logger,
) -> None:
    """
    Computes and logs end-of-run evaluation metrics:
      - AUROC
      - PR-AUC (Average Precision) — preferred under class imbalance
      - Brier Score — measures calibration

    Uses uncertainty features to build a simple composite score (mean of
    normalised margin_mean_early and negative entropy_mean_early) for
    ranking, consistent with the finding that these are the strongest features.

    Requires scikit-learn. Skips gracefully if not installed.
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not found — skipping evaluation metrics. pip install scikit-learn")
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

    # Build a composite score from the strongest early features.
    # Higher score = more likely incorrect (positive class).
    # We use entropy_mean_early (high = uncertain = likely wrong)
    # and margin_mean_early (low = uncertain = likely wrong, so we negate).
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

    # Composite: high entropy + low margin + high NLL -> likely incorrect
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

    # AUROC
    auroc_uncertainty = roc_auc_score(y_valid, score_valid)
    auroc_length      = roc_auc_score(y_valid, _normalise(length_valid))

    # PR-AUC (Average Precision) — better metric under class imbalance
    prauc_uncertainty = average_precision_score(y_valid, score_valid)
    prauc_length      = average_precision_score(y_valid, _normalise(length_valid))

    # Brier score — measures calibration (lower is better; 0.25 = random at 50/50)
    # Normalise composite to [0,1] to use as a probability estimate
    score_prob = _normalise(score_valid)
    brier = brier_score_loss(y_valid, score_prob)

    logger.info("=" * 60)
    logger.info("EVALUATION METRICS (composite uncertainty score, early window)")
    logger.info(f"  AUROC       (uncertainty): {auroc_uncertainty:.4f}")
    logger.info(f"  AUROC       (length only): {auroc_length:.4f}")
    logger.info(f"  PR-AUC      (uncertainty): {prauc_uncertainty:.4f}  ← preferred under class imbalance")
    logger.info(f"  PR-AUC      (length only): {prauc_length:.4f}")
    logger.info(f"  Brier Score (uncertainty): {brier:.4f}  ← calibration (lower=better)")
    logger.info(f"  Baseline PR-AUC (random) : {n_pos/len(y_true):.4f}  ← positive class rate")
    logger.info("=" * 60)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", type=str, default="http://toulouse.stanford.edu:11401/v1")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--system", type=str, default=DEFAULT_SYSTEM)
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--early_T", type=int, default=64)
    ap.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE)
    ap.add_argument("--nucleus_p", type=float, default=0.7)
    ap.add_argument("--tie_delta", type=float, default=0.5)
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--out_prefix", type=str, default="gsm8k_vllm")
    ap.add_argument("--timeout_s", type=float, default=180.0)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    log_path = f"{args.out_prefix}.run.log"
    logger = setup_logger(log_path)

    logger.info("Starting GSM8K vLLM pipeline (FULL vocab logprobs + preboxed)")
    logger.info(f"base_url={args.base_url}")
    logger.info(f"model={args.model}")
    logger.info(f"split={args.split} n={args.n} max_tokens={args.max_tokens} temp={args.temperature}")
    logger.info(f"early_T={args.early_T} vocab_size={args.vocab_size}")
    logger.info(f"nucleus_p={args.nucleus_p} tie_delta={args.tie_delta}")
    logger.info(f"save_every={args.save_every} timeout_s={args.timeout_s} retries={args.retries}")
    logger.info(f"Logging to: {log_path}")

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    n = min(args.n, len(ds))
    logger.info(f"Loaded dataset split={args.split} size={len(ds)}; running n={n}")

    csv_path = f"{args.out_prefix}.partial.csv"
    xlsx_path = f"{args.out_prefix}.final.xlsx"

    rows: List[Dict[str, Any]] = []
    start_time = time.time()

    for i in range(n):
        ex = ds[i]
        q = ex["question"]
        gt = extract_true_target(ex["answer"])

        t0 = time.time()
        assistant_text, token_logprobs, hit_max_tokens, finish_reason = call_vllm_chat(
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
        )

        full_text = build_transcript(args.system, q, assistant_text)
        boxed, extraction_method = extract_llm_answer(assistant_text)
        correct = is_correct(boxed, gt)

        # FULL metrics
        full_metrics = compute_signals_from_token_logprobs(
            token_logprobs,
            early_T=args.early_T,
            nucleus_p=args.nucleus_p,
            tie_delta=args.tie_delta,
            suffix="",
        )

        # PREBOXED metrics
        cutoff = find_preboxed_char_cutoff(assistant_text)
        if cutoff is not None:
            preboxed_tok = truncate_token_logprobs_to_char_cutoff(token_logprobs, cutoff)
            preboxed_metrics = compute_signals_from_token_logprobs(
                preboxed_tok,
                early_T=args.early_T,
                nucleus_p=args.nucleus_p,
                tie_delta=args.tie_delta,
                suffix="_preboxed",
            )
            preboxed_len = len(preboxed_tok)
        else:
            preboxed_metrics = _make_nan_signals("_preboxed", float(args.early_T))
            preboxed_len = 0

        generated_len = len(token_logprobs)
        deg = int(detect_degenerate(assistant_text))
        dt = time.time() - t0
        elapsed = time.time() - start_time

        row = {
            "idx": i,
            "question": q,
            "full_text": full_text,
            "assistant_output": assistant_text,
            "boxed_answer": boxed,
            "extraction_method": extraction_method,
            "true_answer": gt,
            "correct": int(correct),

            "generated_len": int(generated_len),
            "preboxed_generated_len": int(preboxed_len),
            "hit_max_tokens": int(bool(hit_max_tokens)),
            "finish_reason": str(finish_reason) if finish_reason is not None else "",
            "degenerate": int(deg),
            "latency_s": float(dt),

            **full_metrics,
            **preboxed_metrics,
        }

        rows.append(row)

        logger.info(
            f"[{i+1}/{n}] correct={int(correct)} boxed={boxed} extract={extraction_method} "
            f"gen_len={generated_len} preboxed_len={preboxed_len} "
            f"hit_max={int(bool(hit_max_tokens))} deg={deg} "
            f"latency={dt:.2f}s elapsed={elapsed/3600:.2f}h"
        )

        # periodic checkpoint
        if (i + 1) % args.save_every == 0 or (i + 1) == n:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            logger.info(f"Saved checkpoint CSV: {csv_path} (rows={len(rows)})")

    df = pd.DataFrame(rows)
    df.to_excel(xlsx_path, index=False)
    logger.info(f"Saved final XLSX: {xlsx_path}")

    # Log extraction method breakdown
    if rows:
        method_counts = {}
        for r in rows:
            m = r.get("extraction_method", "unknown")
            method_counts[m] = method_counts.get(m, 0) + 1
        logger.info("Answer extraction breakdown:")
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {method}: {count} ({100*count/len(rows):.1f}%)")

    # Compute and log evaluation metrics (AUROC, PR-AUC, Brier)
    compute_evaluation_metrics(rows, logger)

    logger.info("Done.")


if __name__ == "__main__":
    main()