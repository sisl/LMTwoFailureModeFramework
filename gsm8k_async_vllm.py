#!/usr/bin/env python3
"""
gsm8k_async_vllm.py  —  GSM8K single-trace window ablation pipeline.

Generates ONE completion per problem and computes uncertainty features at
each window prefix of that same trace. This is methodologically sounder
than the old approach (which used a separate trace per window), because
every window is a prefix of the same reasoning path and differences across
windows reflect genuine prefix-length effects rather than trace-to-trace
variance.

Key design decisions:
  - One row per problem.
  - All window features extracted from a single generation (same token sequence).
  - Windows are prefixes: window_w128 uses token_logprobs[:128], etc.
  - Prefinal stripping: features also computed on the trace up to (but not
    including) the "Final:" answer line, to avoid the length confound.
  - Compatible with Llama models (no thinking_mode / reasoning_content).
  - Checkpointing by question index; safe to resume with --resume.

Output columns (per window T in ABLATION_WINDOWS):
  entropy_mean_wT, entropy_max_wT, entropy_std_wT
  margin_mean_wT,  margin_max_wT,  margin_std_wT
  nll_mean_wT,     nll_max_wT,     nll_std_wT
  nucleus_size_mean_wT, nucleus_size_max_wT
  near_tie_mean_wT,     near_tie_max_wT
  fork_rate_wT
  window_len_wT          (actual tokens available, <= T)

Plus full-trace versions (suffix "") and prefinal versions (suffix "_prefinal").
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
    from sklearn.metrics import average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABLATION_WINDOWS: List[int] = [128, 256, 400, 512, 1024, 2048]

DEFAULT_SYSTEM = (
    "Reason through the question step by step to arrive at an answer.\n\n"
    "At the end, output a final line exactly in this format:\n"
    "Final: \\boxed{<number>}\n\n"
    "Rules:\n"
    "- Do not use \\boxed{ } except in the final line.\n"
    "- Put only the final numeric answer inside the box."
)

DEFAULT_VOCAB_SIZE = 200


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("gsm8k_vllm")
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

def entropy_from_vocab_logps(vocab_logps: List[float]) -> float:
    lse = logsumexp(vocab_logps)
    ent = 0.0
    for lp in vocab_logps:
        p = math.exp(lp - lse)
        if p > 0.0:
            ent -= p * math.log(p)
    return ent


def margin_from_vocab_logps(vocab_logps: List[float]) -> float:
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


def nucleus_size_from_vocab_logps(
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


def near_tie_count_from_vocab_logps(
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
    """Compute all uncertainty features over a list of token logprob objects."""
    entropies, margins, nlls, nucleus_sizes, near_ties, forks = [], [], [], [], [], []

    for tok in token_logprobs:
        lp_chosen = float(tok.logprob)
        nlls.append(-lp_chosen)
        top = getattr(tok, "top_logprobs", None)
        if not top:
            continue
        vocab_logps = [float(x.logprob) for x in top]
        entropies.append(entropy_from_vocab_logps(vocab_logps))
        margins.append(margin_from_vocab_logps(vocab_logps))
        ns = nucleus_size_from_vocab_logps(vocab_logps, nucleus_p=nucleus_p)
        nt = near_tie_count_from_vocab_logps(vocab_logps, tie_delta=tie_delta)
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
    """
    Compute uncertainty features over the full token list AND each window prefix.
    All windows share the same underlying token sequence — this is the key
    methodological fix vs the old pipeline.
    """
    # Full-trace signals
    result = _compute_signals_from_token_list(
        token_logprobs, nucleus_p=nucleus_p, tie_delta=tie_delta, suffix=suffix
    )
    result[f"full_len{suffix}"] = float(len(token_logprobs))

    # Window prefix signals
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
    """Return NaN-filled signal dict (used when prefinal cutoff is unavailable)."""
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
# Prefinal stripping  (GSM8K variant: strip from "Final:" line)
# ---------------------------------------------------------------------------

def find_prefinal_char_cutoff(assistant_text: str, last_occurrence: bool = False) -> Optional[int]:
    """
    Return the char index just before the line containing "Final: \boxed{...}".
    Returns None if the pattern is not found.

    last_occurrence: if True, use the LAST match rather than the first.
        Set this when using thinking mode, since the model may write \boxed{}
        inside the reasoning trace before repeating it in the final answer.
    """
    finder = (lambda p, t, **kw: list(re.finditer(p, t, **kw))) if last_occurrence \
             else (lambda p, t, **kw: [re.search(p, t, **kw)])

    matches = [m for m in finder(r"Final\s*:", assistant_text, flags=re.IGNORECASE) if m]
    if not matches:
        matches = [m for m in finder(r"\\boxed\{", assistant_text) if m]
    if not matches:
        return None

    m          = matches[-1] if last_occurrence else matches[0]
    idx        = m.start()
    line_start = assistant_text.rfind("\n", 0, idx)
    return line_start + 1 if line_start != -1 else 0


def truncate_token_logprobs_to_char_cutoff(
    token_logprobs: List[Any], cutoff_char_idx: int
) -> List[Any]:
    """Keep only tokens whose cumulative character offset is <= cutoff_char_idx."""
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
# Answer extraction  (GSM8K)
# ---------------------------------------------------------------------------

def extract_gsm8k_answer(text: str) -> Tuple[Optional[str], str]:
    """Extract the numeric answer from model output."""
    # Primary: \boxed{<number>}
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip(), "boxed"

    # Secondary: "Final: <number>" (last line)
    m = re.search(r"Final\s*:\s*([\d,.\-]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip(), "final_line"

    # Tertiary: last number in text
    nums = re.findall(r"[\d,]+(?:\.\d+)?", text.replace(",", ""))
    if nums:
        return nums[-1].strip(), "last_number"

    return None, "none"


def normalize_number(s: str) -> Optional[str]:
    """Normalize a numeric string for comparison."""
    if s is None:
        return None
    s = s.replace(",", "").strip()
    try:
        val = float(s)
        # Return as int string if it's a whole number
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        return s


def extract_true_answer(raw_answer: str) -> str:
    """Extract the ground-truth numeric answer from GSM8K's '#### <number>' format."""
    m = re.search(r"####\s*([\d,.\-]+)", raw_answer)
    if m:
        return m.group(1).strip().replace(",", "")
    return raw_answer.strip()


def is_correct(pred: Optional[str], gt: str) -> bool:
    if pred is None:
        return False
    return normalize_number(pred) == normalize_number(gt)


# ---------------------------------------------------------------------------
# Degenerate output detection
# ---------------------------------------------------------------------------

def detect_degenerate(text: str) -> bool:
    if re.search(r"(.)\1{30,}", text):
        return True
    if re.search(r"\d{20,}", text):
        return True
    return False


# ---------------------------------------------------------------------------
# vLLM call
# ---------------------------------------------------------------------------

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
    thinking_mode: bool = False,
) -> Tuple[str, List[Any], bool, Optional[str]]:
    """
    Call vLLM chat endpoint and return (assistant_text, token_logprobs, hit_max, finish_reason).

    thinking_mode: if True, enables Qwen3-style thinking. The full token sequence
        (reasoning_content + content) is concatenated as assistant_text, and logprobs
        cover that same combined sequence. For non-thinking models, set to False
        (default) — compatible with Llama, Qwen-Instruct, etc.
    """
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    extra_body = None if thinking_mode else {"chat_template_kwargs": {"enable_thinking": False}}

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
                extra_body=extra_body,
            )
            choice = resp.choices[0]

            # DEBUG: dump raw response on first call to diagnose where vLLM
            # puts reasoning_content for this server version.
            if thinking_mode and not getattr(call_vllm_chat, "_debug_dumped", False):
                call_vllm_chat._debug_dumped = True
                try:
                    import json
                    raw = resp.model_dump() if hasattr(resp, "model_dump") else {}
                    logger.info(f"[DEBUG] Raw response (first call):\n{json.dumps(raw, indent=2, default=str)[:3000]}")
                except Exception as de:
                    logger.info(f"[DEBUG] Could not model_dump: {de}")
                    logger.info(f"[DEBUG] choice.message.__dict__: {getattr(choice.message, '__dict__', {})}")

            if thinking_mode:
                # Try all known locations where vLLM may surface the thinking trace.
                # 1. Standard attribute (newer vLLM)
                reasoning = getattr(choice.message, "reasoning_content", None) or ""
                # 2. Nested under model_extra (some vLLM builds)
                if not reasoning:
                    reasoning = ((getattr(choice.message, "model_extra", None) or {})
                                 .get("reasoning_content", "")) or ""
                # 3. Raw __dict__ fallback
                if not reasoning:
                    reasoning = (getattr(choice.message, "__dict__", {})
                                 .get("reasoning_content", "")) or ""

                answer_part = choice.message.content or ""

                if reasoning:
                    assistant_text = (reasoning + "\n" + answer_part).strip()
                else:
                    logger.warning(
                        "[thinking_mode] reasoning_content is empty — falling back to content. "
                        "Check the DEBUG dump above for the actual response structure."
                    )
                    assistant_text = answer_part
            else:
                assistant_text = choice.message.content or ""

            token_logprobs = []
            if getattr(choice, "logprobs", None) is not None:
                token_logprobs = choice.logprobs.content or []
            finish_reason  = getattr(choice, "finish_reason", None)
            hit_max        = finish_reason == "length"
            return assistant_text, token_logprobs, hit_max, finish_reason
        except Exception as e:
            last_err = e
            logger.warning(
                f"vLLM call failed (attempt {attempt+1}/{retries+1}): {e}"
            )
            time.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"vLLM call failed after retries: {last_err}")


# ---------------------------------------------------------------------------
# Optional PR-AUC summary
# ---------------------------------------------------------------------------

def log_prauc_summary(rows: List[Dict], windows: List[int], logger: logging.Logger) -> None:
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not found — skipping PR-AUC summary.")
        return

    import numpy as np

    labels = [r["correct"] for r in rows]
    if len(set(labels)) < 2:
        logger.warning("Only one class present — skipping PR-AUC summary.")
        return

    y_true = np.array(labels, dtype=int)
    # Incorrect = positive class (1 = wrong); flip correct flag
    y_fail  = 1 - y_true

    feature_sets = ["entropy_mean", "nll_mean", "near_tie_mean", "entropy_max", "nll_max"]

    logger.info("=" * 60)
    logger.info("PR-AUC summary (predicting failure = incorrect)")
    logger.info(f"  N={len(rows)}, n_correct={y_true.sum()}, n_incorrect={y_fail.sum()}")
    logger.info(f"  {'Feature':<22} {'Full':>8} " + " ".join(f"w{T:>5}" for T in windows))

    for feat in feature_sets:
        scores = {}
        # Full trace
        col = feat
        vals = np.array([r.get(col, float("nan")) for r in rows], dtype=float)
        mask = np.isfinite(vals)
        if mask.sum() >= 10 and len(set(y_fail[mask])) == 2:
            scores["full"] = average_precision_score(y_fail[mask], vals[mask])
        else:
            scores["full"] = float("nan")

        # Window prefixes
        for T in windows:
            col = f"{feat}_w{T}"
            vals = np.array([r.get(col, float("nan")) for r in rows], dtype=float)
            mask = np.isfinite(vals)
            if mask.sum() >= 10 and len(set(y_fail[mask])) == 2:
                scores[T] = average_precision_score(y_fail[mask], vals[mask])
            else:
                scores[T] = float("nan")

        row_str = (
            f"  {feat:<22} {scores['full']:>8.4f} "
            + " ".join(f"{scores.get(T, float('nan')):>7.4f}" for T in windows)
        )
        logger.info(row_str)

    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="GSM8K single-trace window ablation pipeline (Llama / Qwen)"
    )
    ap.add_argument("--base_url",    required=True,  help="vLLM OpenAI-compatible endpoint")
    ap.add_argument("--model",       required=True,  help="Model name as registered in vLLM")
    ap.add_argument("--system",      default=DEFAULT_SYSTEM)
    ap.add_argument("--split",       default="test", help="GSM8K split (train / test)")
    ap.add_argument("--n",           type=int,   default=0,
                    help="Number of problems to run (0 = all)")
    ap.add_argument("--max_tokens",  type=int,   default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--vocab_size",  type=int,   default=DEFAULT_VOCAB_SIZE,
                    help="top_logprobs to request (set to model's vocab size)")
    ap.add_argument("--nucleus_p",   type=float, default=0.9)
    ap.add_argument("--tie_delta",   type=float, default=0.1)
    ap.add_argument("--windows",     type=int,   nargs="+", default=ABLATION_WINDOWS)
    ap.add_argument("--save_every",  type=int,   default=20,
                    help="Checkpoint every N problems")
    ap.add_argument("--out_prefix",  type=str,   default="gsm8k_vllm")
    ap.add_argument("--timeout_s",   type=float, default=180.0)
    ap.add_argument("--retries",     type=int,   default=2)
    ap.add_argument("--thinking_mode", action="store_true",
                    help="Enable Qwen3-style thinking mode (reasoning_content + content)")
    ap.add_argument("--resume",      action="store_true",
                    help="Resume from existing partial CSV")
    args = ap.parse_args()

    log_path  = f"{args.out_prefix}.run.log"
    csv_path  = f"{args.out_prefix}.partial.csv"
    final_csv = f"{args.out_prefix}.final.csv"

    logger = setup_logger(log_path)
    logger.info("=" * 70)
    logger.info("GSM8K vLLM pipeline — single-trace window ablation")
    logger.info(f"  model={args.model}  base_url={args.base_url}")
    logger.info(f"  split={args.split}  n={args.n}  max_tokens={args.max_tokens}")
    logger.info(f"  temperature={args.temperature}  vocab_size={args.vocab_size}")
    logger.info(f"  windows={args.windows}")
    logger.info(f"  nucleus_p={args.nucleus_p}  tie_delta={args.tie_delta}")
    logger.info(f"  save_every={args.save_every}  resume={args.resume}  thinking_mode={args.thinking_mode}")
    logger.info("=" * 70)

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    # Load dataset
    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    n  = min(args.n, len(ds)) if args.n > 0 else len(ds)
    logger.info(f"Loaded GSM8K {args.split}: {len(ds)} problems; running {n}")

    # Resume logic
    rows: List[Dict[str, Any]] = []
    completed_indices: set = set()
    if args.resume:
        try:
            existing = pd.read_csv(csv_path)
            if len(existing) > 0 and "problem_idx" in existing.columns:
                rows = existing.to_dict(orient="records")
                completed_indices = set(existing["problem_idx"].tolist())
                logger.info(
                    f"Resuming: {len(rows)} rows loaded, "
                    f"{len(completed_indices)} problems already done."
                )
        except Exception as e:
            logger.warning(f"Could not load checkpoint ({e}) — starting fresh.")

    start_time = time.time()

    for i in range(n):
        if i in completed_indices:
            continue

        ex  = ds[i]
        q   = ex["question"]
        gt  = extract_true_answer(ex["answer"])

        t0 = time.time()
        try:
            assistant_text, token_logprobs, hit_max, finish_reason = call_vllm_chat(
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
                thinking_mode=args.thinking_mode,
            )
        except Exception as e:
            logger.error(f"[{i+1}/{n}] Fatal error after retries: {e} — skipping.")
            continue

        # Answer extraction & correctness
        pred, extraction_method = extract_gsm8k_answer(assistant_text)
        correct = int(is_correct(pred, gt))

        # Full-trace signals + all window prefixes (same token sequence)
        full_signals = compute_signals_multiwindow(
            token_logprobs,
            windows=args.windows,
            nucleus_p=args.nucleus_p,
            tie_delta=args.tie_delta,
            suffix="",
        )

        # Prefinal-stripped signals (strip "Final: \boxed{}" line and beyond)
        cutoff = find_prefinal_char_cutoff(assistant_text, last_occurrence=args.thinking_mode)
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
                args.windows, suffix="_prefinal",
                nucleus_p=args.nucleus_p, tie_delta=args.tie_delta,
            )
            prefinal_len = 0

        generated_len = len(token_logprobs)
        deg = int(detect_degenerate(assistant_text))
        dt  = time.time() - t0
        elapsed = time.time() - start_time

        row: Dict[str, Any] = {
            # Identification
            "problem_idx":        i,
            "question":           q[:500],        # truncated to save space
            "true_answer":        gt,
            # Prediction
            "predicted_answer":   pred,
            "extraction_method":  extraction_method,
            "correct":            correct,
            # Output metadata
            "assistant_output":   assistant_text[:3000],
            "generated_len":      generated_len,
            "prefinal_len":       prefinal_len,
            "hit_max_tokens":     int(bool(hit_max)),
            "finish_reason":      str(finish_reason) if finish_reason else "",
            "degenerate":         deg,
            "latency_s":          float(dt),
            # Uncertainty signals
            **full_signals,
            **prefinal_signals,
        }

        rows.append(row)
        completed_indices.add(i)

        logger.info(
            f"[{i+1}/{n}] correct={correct} pred={pred} gt={gt} "
            f"gen={generated_len} prefinal={prefinal_len} "
            f"deg={deg} lat={dt:.1f}s elapsed={elapsed/3600:.2f}h"
        )

        if (i + 1) % args.save_every == 0 or (i + 1) == n:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            logger.info(
                f"Checkpoint: {csv_path} "
                f"(rows={len(rows)}, done={len(completed_indices)}/{n})"
            )

    # Final save
    df = pd.DataFrame(rows)
    df.to_csv(final_csv, index=False)
    df.to_csv(csv_path, index=False)
    logger.info(f"Final output saved: {final_csv}  ({len(df)} rows)")

    # Quick accuracy report
    if len(df) > 0 and "correct" in df.columns:
        acc = df["correct"].mean()
        logger.info(f"Overall accuracy: {acc:.3f}  ({df['correct'].sum()}/{len(df)})")
        logger.info(
            f"Prefinal coverage: "
            f"{(df['prefinal_len'] > 0).sum()}/{len(df)} rows have prefinal_len > 0"
        )

    # PR-AUC summary
    log_prauc_summary(rows, args.windows, logger)
    logger.info("Done.")


if __name__ == "__main__":
    main()
