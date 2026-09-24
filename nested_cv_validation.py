"""
Nested-CV independent validation of the committed/persistent mode assignment.
Rung-3 answer to R1's circularity objection: the mode label and T* are chosen
using ONLY training folds; Delta is estimated on the held-out fold the label
never saw. Every question is in exactly one outer test fold (no data halving).

Design (per configuration):
  Outer GroupKFold (by idx), K_OUTER folds. For each outer fold:
    - TRAIN portion: compute the PR-AUC curve over windows + full trace using
      cv_oof_scores (inner CV), pick T* = argmax early-window PR-AUC, and read
      the verdict-rule mode from delta_ci on the train portion.
    - TEST fold: refit at the train-chosen T* and full, compute
      Delta_test = PR-AUC(T*) - PR-AUC(full) on the held-out fold only.
  Aggregate: per-config out-of-sample Delta = mean over outer folds; mode =
  majority train-fold label. POOLED TEST (where the power is): committed vs
  persistent out-of-sample Deltas, Mann-Whitney; pooled sign test.

Reuses the paper's own functions (analyze_updated_dataset_agnostic.py):
prefinal mode, same classifier, same window/verdict logic. Gemma/MATH-500 is
ONE config here (decision 2a: strata handled separately), classified by its
aggregate verdict.

Scope: runs on the pool configs whose raw per-completion files are on disk.
Configs without a raw file are reported as UNAVAILABLE, not silently dropped.

Usage:
  python nested_cv_validation.py
  python nested_cv_validation.py --k_outer 5 --n_boot 2000
"""
import sys, os, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

REPO = "/Users/tinat/LMTwoFailureModeFramework"
sys.path.insert(0, REPO)
import analyze_updated_dataset_agnostic as A

R = f"{REPO}/results"
WINDOWS = [128, 256, 400, 512, 1024, 2048]

# Pool configs → raw file + pool class (from pooling_analysis.py v5).
# in_pool True = counted; file=None = raw data not on disk (reported UNAVAILABLE).
CONFIGS = {
    # committed
    "gsm8k_qwen3.5-2b":     dict(f=f"{R}/gsm8k_qwen3.5-2b.final.csv",              cls="C"),
    "gsm8k_llama3.1-8b":    dict(f=f"{R}/gsm8k_llama3.1-8B.final.csv",             cls="C"),
    "math500_qwen3.5-2b":   dict(f=f"{R}/math500_qwen3.5-2b.final.xlsx",           cls="C"),
    "math500_qwen3.5-122b": dict(f=f"{R}/math500_qwen3.5-122b.final.xlsx",         cls="C"),
    "math500_gpt-oss-20b":  dict(f=f"{R}/gpt-oss-20b_math500_updated.final.xlsx",  cls="C"),
    "math500_gemini2.5":    dict(f=f"{R}/math500_gemini2.5.final.csv",             cls="C"),
    "gpqa_qwen3.5-2b":      dict(f=f"{R}/gpqa_qwen3.5-2b.final.xlsx",              cls="C"),
    "gpqa_gemma4-31b":      dict(f=f"{R}/gpqa_gemma4-31b.final.xlsx",              cls="C"),
    "gpqa_gpt4o":           dict(f=f"{R}/gpqa_gpt4o.final.csv",                    cls="C"),
    "lcb_gemma4-31b":       dict(f=f"{R}/lcb_gemma4-31B.final.xlsx",               cls="C"),
    "lcb_gpt-oss-20b":      dict(f=f"{R}/lcb_gpt-oss-20b_v2.final.xlsx",           cls="C"),
    # Gemma/MATH-500 as ONE config (decision 2a)
    "math500_gemma4-31b":   dict(f=f"{R}/math500_gemma4-31b.final.xlsx",           cls="C"),
    # persistent
    "gsm8k_gpt-oss-20b":    dict(f=f"{R}/gsm8k_gpt-oss-20b_final_prefinal_fixed.csv", cls="P"),
    "math500_llama3.1-8b":  dict(f=f"{R}/math500_llama3.1-8B.final.xlsx",          cls="P"),
    "math500_gpt4o":        dict(f=f"{R}/math500_gpt4o.final.csv",                 cls="P"),
    "gpqa_gpt-oss-20b":     dict(f=f"{R}/gpqa_gptoss20b.final.xlsx",               cls="P"),
    "lcb_qwen3.5-2b":       dict(f=f"{R}/lcb_qwen3.5-2b.final.xlsx",               cls="P"),
    "lcb_qwen3.5-122b":     dict(f=f"{R}/lcb_qwen3.5-122b.final.xlsx",             cls="P"),
    # UNAVAILABLE — raw single-completion GPQA files not in results/ (see notes)
    "gpqa_qwen3.5-9b_single":  dict(f=None, cls="P"),
    "gpqa_qwen122b_single":    dict(f=None, cls="P"),
}


def prep(path):
    df = A.load_file(path)
    if "y" not in df.columns:
        if "correct" in df.columns:
            df["y"] = 1 - df["correct"].astype(int)
        else:
            raise ValueError("no y/correct column")
    work, mode = A._prepare_df_for_analysis(df, prefinal=True)
    qcol = next((c for c in ("idx","question_id","qid") if c in work.columns), None)
    if qcol is None:
        work = work.reset_index(drop=True); work["_qid"] = work.index; qcol = "_qid"
    work[qcol] = work[qcol].astype(str)
    return work, mode, qcol


def allowed_windows(df):
    len_col = next((c for c in ("prefinal_len","preboxed_generated_len","generated_len")
                    if c in df.columns), None)
    mx = df[len_col].max() if len_col else max(WINDOWS)
    return [T for T in WINDOWS if T <= 512 or mx >= T]


def curve_pick_tstar(train, wins):
    """On TRAIN only: PR-AUC per window (inner CV), argmax = T*, verdict mode."""
    best = None
    for T in wins:
        wc = A.window_cols(train, T)
        e, ey, _ = A.cv_oof_scores(train, wc)
        if e is None: continue
        pa = average_precision_score(ey, e)
        if best is None or pa > best[1]:
            best = (T, pa)
    if best is None:
        return None, None
    T_star = best[0]
    # verdict mode from train-portion delta CI at T*
    wc = A.window_cols(train, T_star); fc = A.full_cols(train)
    e, ey, _ = A.cv_oof_scores(train, wc)
    f, fy, _ = A.cv_oof_scores(train, fc)
    if e is None or f is None or not np.array_equal(ey, fy):
        return T_star, "A"
    d, lo, hi, p = A.delta_ci(fy, e, f, n_boot=1000)
    v = A.verdict(d, lo, hi, p)
    mode = "C" if v in ("✓ Committed","~ Near-committed") else ("P" if v=="✗ Full better" else "A")
    return T_star, mode


def delta_on_test(test, T_star):
    """Held-out fold: refit at fixed T*, Delta = PR-AUC(T*) - PR-AUC(full)."""
    wc = A.window_cols(test, T_star); fc = A.full_cols(test)
    e, ey, _ = A.cv_oof_scores(test, wc)
    f, fy, _ = A.cv_oof_scores(test, fc)
    if e is None or f is None or not np.array_equal(ey, fy):
        return None
    return average_precision_score(ey, e) - average_precision_score(ey, f)


def run_config(name, work, qcol, k_outer):
    wins = allowed_windows(work)
    qids = work[qcol].values
    y_by_q = work.groupby(qcol)["y"].max()
    uq = y_by_q.index.values
    # need enough failing questions to make outer folds meaningful
    n_fail_q = int((y_by_q.values == 1).sum())
    if n_fail_q < k_outer:
        return dict(status="too_few_fail_q", n_fail_q=n_fail_q)
    gkf = GroupKFold(n_splits=k_outer)
    Xq = np.zeros(len(uq)); yq = y_by_q.reindex(uq).values
    fold_deltas, fold_modes, fold_tstars = [], [], []
    # split on questions, then map back to rows
    for tr_q, te_q in gkf.split(Xq, yq, groups=uq):
        tr_ids = set(uq[tr_q]); te_ids = set(uq[te_q])
        train = work[work[qcol].isin(tr_ids)]
        test  = work[work[qcol].isin(te_ids)]
        T_star, mode = curve_pick_tstar(train, wins)
        if T_star is None: continue
        dt = delta_on_test(test, T_star)
        if dt is None: continue
        fold_deltas.append(dt); fold_modes.append(mode); fold_tstars.append(T_star)
    if not fold_deltas:
        return dict(status="no_valid_folds")
    modes = [m for m in fold_modes if m in ("C","P")]
    maj = max(set(modes), key=modes.count) if modes else "A"
    return dict(status="ok",
                oos_delta=float(np.mean(fold_deltas)),
                oos_delta_folds=fold_deltas,
                mode_oos=maj,
                mode_stability=(modes.count(maj)/len(modes)) if modes else 0.0,
                modal_tstar=max(set(fold_tstars), key=fold_tstars.count),
                n_fail_q=n_fail_q, n_folds=len(fold_deltas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k_outer", type=int, default=5)
    args = ap.parse_args()

    rows, unavailable = [], []
    for name, spec in CONFIGS.items():
        if spec["f"] is None:
            unavailable.append((name, spec["cls"])); continue
        if not os.path.exists(spec["f"]):
            unavailable.append((name, spec["cls"] + " (file missing)")); continue
        try:
            work, mode, qcol = prep(spec["f"])
        except Exception as e:
            unavailable.append((name, f'{spec["cls"]} (load err: {e})')); continue
        res = run_config(name, work, qcol, args.k_outer)
        if res["status"] != "ok":
            unavailable.append((name, f'{spec["cls"]} ({res["status"]})')); continue
        rows.append(dict(config=name, pool_class=spec["cls"], **res))
        print(f"{name:<24} pool={spec['cls']} "
              f"mode_oos={res['mode_oos']}({res['mode_stability']:.0%}) "
              f"T*={res['modal_tstar']} "
              f"oos_Δ={res['oos_delta']:+.4f} "
              f"n_fail_q={res['n_fail_q']} folds={res['n_folds']}")

    df = pd.DataFrame(rows)
    print("\n" + "="*72)
    print(f"NESTED-CV VALIDATION — {len(df)}/{len(CONFIGS)} pool configs with raw data")
    if unavailable:
        print("Unavailable/failed:")
        for n, why in unavailable: print(f"    {n}: {why}")

    # PRIMARY pooled test: out-of-sample Delta, committed vs persistent (pool_class)
    from scipy.stats import mannwhitneyu, binomtest
    com = df[df["pool_class"]=="C"]["oos_delta"].values
    per = df[df["pool_class"]=="P"]["oos_delta"].values
    print(f"\nPer-pool-class out-of-sample Δ:")
    print(f"  committed  n={len(com)}  median={np.median(com):+.4f}  mean={np.mean(com):+.4f}")
    print(f"  persistent n={len(per)}  median={np.median(per):+.4f}  mean={np.mean(per):+.4f}")
    if len(com) and len(per):
        u = mannwhitneyu(com, per, alternative="greater")   # committed Δ > persistent Δ
        print(f"  Mann-Whitney (committed Δ > persistent Δ): U={u.statistic:.1f}, p={u.pvalue:.4g}")

    # Directional out-of-sample sign checks (pool class as ground truth)
    com_pos = int((com > 0).sum()); per_nonpos = int((per <= 0).sum())
    print(f"\nOut-of-sample directional agreement with pool class:")
    print(f"  committed with oos_Δ>0:  {com_pos}/{len(com)}")
    print(f"  persistent with oos_Δ≤0: {per_nonpos}/{len(per)}")
    alln = len(com)+len(per); match = com_pos+per_nonpos
    if alln:
        bt = binomtest(match, alln, 0.5, alternative="greater")
        print(f"  joint: {match}/{alln}, binomial p={bt.pvalue:.4g}")

    # Agreement between out-of-sample mode and pool class (label reproducibility)
    if len(df):
        agree = (df["mode_oos"]==df["pool_class"]).mean()
        print(f"\nOut-of-sample mode matches pool class in {100*agree:.0f}% of configs")

    df.to_csv("nested_cv_validation.csv", index=False)
    print("\n-> nested_cv_validation.csv written")
    print("\nInterpretation: this tests whether the mode ASSIGNMENT PROCEDURE")
    print("(label+T* from train folds) predicts held-out Δ. Pool membership was")
    print("set on full data; report as validation of assignment, not of the pool.")
    print("DONE")


if __name__ == "__main__":
    main()