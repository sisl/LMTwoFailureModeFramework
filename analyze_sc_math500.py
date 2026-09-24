"""
Adapter: run analyze_sc_replication.py analyses on the MATH-500 SC file.

The MATH-500 SC file has: idx, completion_idx, correct, question,
boxed_answer, true_answer, plus prefinal feature columns.

The GPQA SC file that analyze_sc_replication.py was written for has:
question_id, majority_correct, agreement_rate, p_correct, completion_idx,
predicted_answer, subject.

This adapter derives the missing columns from the raw correct column,
renames idx → question_id, and calls the existing analysis functions
directly so nothing in analyze_sc_replication.py needs to change.

Usage:
    python run_math500_sc_replication.py \
        --input results/math500_gemma4_sc.partial.csv \
        --early_window 400 \
        --output_dir results/self_consistency/math500_gemma4/
"""

import argparse, os, sys
import numpy as np
import pandas as pd

# ── point at the repo so we can import the SC analysis script ─────────────
REPO = "/Users/tinat/LMTwoFailureModeFramework"
sys.path.insert(0, REPO)

# Import the analysis functions directly — we won't call main()
import gpqa.analyze_sc_replication as SC

# ─────────────────────────────────────────────────────────────────────────────

def build_sc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive majority_correct, agreement_rate, p_correct from per-completion
    correct flags grouped by idx.  Also adds a question_id column (copy of
    idx) because get_question_level() groups on question_id.

    majority_correct = 1 if majority of completions for that question are
                       correct, 0 otherwise.  Ties go to incorrect (conservative).
    agreement_rate   = fraction of completions that agree with the majority
                       answer — approximated here as max(p_correct, 1-p_correct)
                       since we don't have per-answer string agreement.
    p_correct        = fraction of completions that are correct.
    """
    df = df.copy()

    # Add question_id as alias for idx — get_question_level() expects question_id
    df["question_id"] = df["idx"]

    # Derive per-question SC statistics (group by idx, merge back on idx)
    q_stats = (
        df.groupby("idx")["correct"]
        .agg(
            p_correct=lambda x: float(x.mean()),
            n_completions="count",
        )
        .reset_index()
    )
    q_stats["majority_correct"] = (q_stats["p_correct"] > 0.5).astype(int)
    # agreement_rate: fraction voting with the majority
    q_stats["agreement_rate"] = q_stats["p_correct"].apply(
        lambda p: max(p, 1 - p)
    )

    df = df.merge(
        q_stats[["idx", "majority_correct", "agreement_rate", "p_correct"]],
        on="idx",
        how="left",
    )

    # Columns the SC script expects but MATH-500 doesn't have
    if "predicted_answer" not in df.columns:
        df["predicted_answer"] = df["boxed_answer"]   # close enough for clean()
    if "subject" not in df.columns:
        df["subject"] = "math500"                     # single subject
    if "hit_max_tokens" not in df.columns:
        df["hit_max_tokens"] = 0
    if "degenerate" not in df.columns:
        df["degenerate"] = 0

    return df


def remap_feature_columns(df: pd.DataFrame, w: int) -> pd.DataFrame:
    """
    The GPQA SC script expects:
        entropy_mean_prefinal          (full-trace prefinal, no window suffix)
        margin_mean_prefinal
        near_tie_mean_prefinal
        entropy_mean_prefinal_w{w}     (early-window prefinal)
        margin_mean_prefinal_w{w}
        near_tie_mean_prefinal_w{w}

    The MATH-500 SC file stores these as:
        entropy_mean_prefinal_w{T}     for T in {128,256,400,512,1024,2048}

    Strategy: treat the largest available window (2048) as the "full"
    prefinal trace (matches what analyze_updated_dataset_agnostic.py does
    when it remaps _prefinal columns in prefinal mode).  The early window
    is whatever --early_window specifies.
    """
    df = df.copy()

    # Map full-trace prefinal: use the largest window present as proxy
    windows_available = []
    for T in [128, 256, 400, 512, 1024, 2048]:
        if f"entropy_mean_prefinal_w{T}" in df.columns:
            windows_available.append(T)

    if not windows_available:
        raise ValueError(
            "No prefinal window columns found. "
            "Expected entropy_mean_prefinal_w{T} for T in {128,...,2048}."
        )

    full_proxy = max(windows_available)
    print(f"  [adapter] using T={full_proxy} as full-trace prefinal proxy")
    print(f"  [adapter] early window: T={w}")
    print(f"  [adapter] windows available: {windows_available}")

    feats = ["entropy_mean", "margin_mean", "near_tie_mean"]
    for feat in feats:
        src_full  = f"{feat}_prefinal_w{full_proxy}"
        dst_full  = f"{feat}_prefinal"
        src_early = f"{feat}_prefinal_w{w}"
        dst_early = f"{feat}_prefinal_w{w}"   # same name — no rename needed

        if src_full in df.columns:
            df[dst_full] = df[src_full]
        else:
            print(f"  [adapter] WARNING: {src_full} not found — {dst_full} will be NaN")
            df[dst_full] = np.nan

        if src_early not in df.columns:
            print(f"  [adapter] WARNING: {src_early} not found — early features will be NaN")

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",        required=True)
    ap.add_argument("--early_window", type=int, default=400)
    ap.add_argument("--output_dir",   default="results/self_consistency/math500_gemma4/")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    w = args.early_window

    print(f"Loading: {args.input}")
    df_raw = SC.load_data(args.input)
    print(f"  {len(df_raw)} rows, {df_raw['idx'].nunique()} questions")

    # Step 1: derive SC columns
    print("\nDeriving SC columns (majority_correct, agreement_rate, p_correct)...")
    df_sc = build_sc_columns(df_raw)

    # Quick sanity check
    q_check = df_sc.groupby("question_id")["majority_correct"].first()
    n_sc_correct = int(q_check.sum())
    n_q = len(q_check)
    print(f"  {n_q} questions | SC correct: {n_sc_correct} ({100*n_sc_correct/n_q:.1f}%)")
    print(f"  SC failures:    {n_q - n_sc_correct} ({100*(n_q-n_sc_correct)/n_q:.1f}%)")

    # Step 2: remap feature column names
    print("\nRemapping feature columns...")
    df_remapped = remap_feature_columns(df_sc, w)

    # Step 3: clean (filter hit_max_tokens=0, degenerate=0, predicted_answer notna)
    comp_df = SC.clean(df_remapped)
    print(f"  After clean(): {len(comp_df)} rows")

    # Step 4: aggregate to question level (single completion per question)
    print(f"\nAggregating to question level (completion_idx=0 per question)...")
    q_df = SC.get_question_level(comp_df, w)
    print(f"  {len(q_df)} questions | early window T={w}")

    # Verify SC failures are present — needed for triage analysis
    n_sc_fail = (q_df["sc_correct"] == 0).sum()
    print(f"  SC failures in question-level df: {n_sc_fail} "
          f"({100*n_sc_fail/len(q_df):.1f}%)")
    if n_sc_fail < 10:
        print("  WARNING: very few SC failures — triage curves will be noisy")

    # Step 5: run the three analyses
    print(f"\n{'='*70}")
    print(f"Running SC replication analyses (early_window=T{w})")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*70}")

    agree_df, agree_best = SC.analysis_direct_agreement(q_df, args.output_dir, w)
    pred_results         = SC.analysis_sc_verdict_prediction(q_df, args.output_dir, w)
    select_df, curves    = SC.analysis_selective_sc(q_df, args.output_dir, w)
    SC.analysis_accuracy_proxy(comp_df, q_df, args.output_dir, w)
    SC.print_summary(agree_best, pred_results, curves["prefinal_full"], q_df)

    # Save tables
    agree_df.to_csv(
        os.path.join(args.output_dir, f"sc_agreement_sweep_w{w}.csv"), index=False)
    select_df.to_csv(
        os.path.join(args.output_dir, f"sc_selective_curve_all_signals_w{w}.csv"),
        index=False)

    # ── Rebuttal-ready table: Analysis 3 at key operating points ─────────
    print(f"\n{'='*70}")
    print("REBUTTAL TABLE — Analysis 3: Selective SC operating points")
    print(f"{'='*70}")
    print(f"{'Signal':<38} {'skip20':>8} {'skip30':>8} {'skip45':>8} {'skip50':>8}")
    print("-" * 70)
    for sig_name, curve in curves.items():
        label = SC.SIGNAL_LABELS[sig_name].format(w=w)
        def r_at(pct):
            rows = curve[curve["skip_pct"] == pct]
            return rows["recall_sc_fail"].values[0] if len(rows) else float("nan")
        def p_at(pct):
            rows = curve[curve["skip_pct"] == pct]
            return rows["skip_precision"].values[0] if len(rows) else float("nan")
        print(f"  {label:<36} "
              f"{r_at(20):.3f}/{p_at(20):.3f}  "
              f"{r_at(30):.3f}/{p_at(30):.3f}  "
              f"{r_at(45):.3f}/{p_at(45):.3f}  "
              f"{r_at(50):.3f}/{p_at(50):.3f}")
    print("  Format: recall/precision at each skip rate")
    print("\nDONE")


if __name__ == "__main__":
    main()