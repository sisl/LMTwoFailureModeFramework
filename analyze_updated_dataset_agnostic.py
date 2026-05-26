#!/usr/bin/env python3
"""
analyze_gsm8k_singlefile.py
============================
Analysis script for single-file multi-window GSM8K output produced by
qwen_vllm.py or gpqa_vllm.py style pipelines where ALL window sizes are
computed from the SAME trace and stored as columns in one CSV/XLSX file.

Column naming convention (from gpqa_vllm.py compute_signals_multiwindow):
  Full-trace features (no suffix):
      entropy_mean, entropy_max, entropy_std,
      nll_mean, nll_max, nll_std,
      margin_mean, margin_max, margin_std,
      nucleus_size_mean, nucleus_size_max,
      near_tie_mean, near_tie_max,
      fork_rate, full_len

  Early-window features (_w{T} suffix):
      entropy_mean_w128, entropy_max_w128, ...  (same base names)
      window_len_w128, ...

  Prefinal features (_prefinal suffix, full trace):
      entropy_mean_prefinal, entropy_max_prefinal, ...
      prefinal_len

  Prefinal early-window (_prefinal_w{T} suffix):
      entropy_mean_prefinal_w128, ...

  Required metadata columns:
      correct   (1=correct, 0=wrong)
      generated_len  (or full_len)
      prefinal_len   (0 if no \\boxed{} / Final Answer found)

Sections produced
-----------------
1. Dataset overview
2. Window ablation — early vs full PR-AUC with 95% bootstrap CIs + delta CI
3. Prefinal ablation — length confound check
4. Per-feature breakdown — mean vs max, individual curves, inverted-U detection
5. Summary verdict + publication-quality plot

Methodological note on --prefinal flag
---------------------------------------
When --prefinal is passed, the ENTIRE analysis (Sections 2, 4, 5, plot) runs
on prefinal features only:
  - Early: uncertainty features over tokens 0→T of the prefinal trace
  - Full:  uncertainty features over all tokens up to "Final Answer:" cutoff

This is the methodologically correct primary analysis because both early and
full baselines operate on the same token pool (the prefinal trace), eliminating
the length confound. Without --prefinal, Section 3 still does this comparison
as a robustness check, but Sections 2/4/5 use raw full-trace features.
"""

import argparse
import sys
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Feature config
# ─────────────────────────────────────────────────────────────────────────────

BASE_FEATURES = [
    "entropy_mean", "entropy_max",
    "nll_mean",     "nll_max",
    "margin_mean",  "margin_max",
    "nucleus_size_mean", "nucleus_size_max",
    "near_tie_mean",     "near_tie_max",
    "fork_rate",
]

MAX_FEATURES  = [f for f in BASE_FEATURES if "max"  in f]
MEAN_FEATURES = [f for f in BASE_FEATURES if "mean" in f or f == "fork_rate"]

N_BOOT   = 10000
SEED     = 42
CV_K     = 5
MIN_FAIL = 5


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_file(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        for enc in ("utf-8", "latin-1", "cp1252", "utf-8-sig"):
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            raise ValueError(f"Cannot decode {path}")
    if "correct" not in df.columns:
        raise ValueError("'correct' column missing")
    df["y"] = 1 - df["correct"].astype(int)
    return df


def window_cols(df: pd.DataFrame, T: int, prefinal: bool = False) -> List[str]:
    """Return available feature columns for a given window T."""
    suffix = f"_prefinal_w{T}" if prefinal else f"_w{T}"
    cols = [f + suffix for f in BASE_FEATURES]
    return [c for c in cols if c in df.columns]


def full_cols(df: pd.DataFrame, prefinal: bool = False) -> List[str]:
    """Return available full-trace feature columns."""
    suffix = "_prefinal" if prefinal else ""
    cols = [f + suffix for f in BASE_FEATURES]
    return [c for c in cols if c in df.columns]


def detect_windows(df: pd.DataFrame) -> List[int]:
    """Auto-detect window sizes from column names."""
    import re
    windows = set()
    for col in df.columns:
        m = re.search(r"_w(\d+)$", col)
        if m:
            windows.add(int(m.group(1)))
    return sorted(windows)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _avail(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


# ── Cluster-bootstrap helpers ─────────────────────────────────────────────────
# When --cluster_col is set (e.g. question_id for multi-completion data),
# bootstrap resamples CLUSTER IDs rather than individual rows.  All rows
# belonging to a sampled cluster are included together, so within-cluster
# correlation is fully respected.  The effective sample size is the number
# of unique clusters, not the number of rows.
#
# CV fold assignment uses GroupKFold so that all completions of a question
# are always in the same fold (never split across train/val).
# ─────────────────────────────────────────────────────────────────────────────

# Global cluster array — set by main() when --cluster_col is provided.
# None means standard i.i.d. bootstrap / StratifiedKFold.
CLUSTER_IDS: Optional[np.ndarray] = None


def _cluster_resample(rng, cluster_ids: np.ndarray):
    """
    Resample cluster IDs with replacement, return row indices.
    Each unique cluster is treated as a single bootstrap unit.
    """
    unique_clusters = np.unique(cluster_ids)
    sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters),
                                  replace=True)
    idx = np.concatenate([np.where(cluster_ids == c)[0]
                          for c in sampled_clusters])
    return idx


def bootstrap_prauc(y_true, y_score, n_boot=N_BOOT, seed=SEED,
                    cluster_ids: Optional[np.ndarray] = None):
    rng    = np.random.default_rng(seed)
    n      = len(y_true)
    scores = []
    use_cluster = cluster_ids is not None and len(cluster_ids) == n

    for _ in range(n_boot):
        if use_cluster:
            idx = _cluster_resample(rng, cluster_ids)
        else:
            idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        scores.append(average_precision_score(yt, ys))
    if not scores:
        return float("nan"), float("nan"), float("nan")
    s = np.array(scores)
    return float(np.mean(s)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def cv_prauc(df, features, label="y", k=CV_K,
             cluster_ids: Optional[np.ndarray] = None):
    avail = _avail(df, features)
    if not avail:
        return float("nan"), float("nan"), float("nan"), float("nan")
    X = df[avail].fillna(df[avail].median()).values
    y = df[label].values
    if y.sum() < MIN_FAIL:
        return float("nan"), float("nan"), float("nan"), float("nan")
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced",
                                  solver="lbfgs", C=1.0)),
    ])

    use_cluster = cluster_ids is not None and len(cluster_ids) == len(df)
    if use_cluster:
        from sklearn.model_selection import GroupKFold
        # GroupKFold keeps all completions of a question in the same fold.
        # It doesn't stratify, so check there are failures in each fold.
        gkf = GroupKFold(n_splits=k)
        splits = list(gkf.split(X, y, groups=cluster_ids))
    else:
        from sklearn.model_selection import StratifiedKFold
        skf    = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
        splits = list(skf.split(X, y))

    all_probs, all_y = [], []
    for tr, va in splits:
        if y[tr].sum() == 0 or y[tr].sum() == len(tr):
            continue
        if y[va].sum() == 0:
            continue
        pipe.fit(X[tr], y[tr])
        all_probs.append(pipe.predict_proba(X[va])[:, 1])
        all_y.append(y[va])
    if not all_probs:
        return float("nan"), float("nan"), float("nan"), float("nan")
    yp = np.concatenate(all_probs)
    yt = np.concatenate(all_y)
    if yt.sum() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    pa = float(average_precision_score(yt, yp))
    au = float(roc_auc_score(yt, yp))

    # CIs: cluster-bootstrap over OOF scores using their original cluster ids
    # (only rows that appeared in validation folds)
    val_mask = np.zeros(len(df), dtype=bool)
    for _, va in splits:
        val_mask[va] = True
    ci_clusters = cluster_ids[val_mask] if use_cluster else None
    _, lo, hi = bootstrap_prauc(yt, yp, cluster_ids=ci_clusters)
    return pa, lo, hi, au


def cv_oof_scores(df, features, label="y", k=CV_K,
                  cluster_ids: Optional[np.ndarray] = None):
    """
    Return (OOF predicted probabilities, true labels, val_cluster_ids).
    val_cluster_ids is None when not clustering, otherwise the cluster id
    for each OOF row — needed for cluster-bootstrap delta CIs.
    """
    avail = _avail(df, features)
    if not avail:
        return None, None, None
    X = df[avail].fillna(df[avail].median()).values
    y = df[label].values
    if y.sum() < MIN_FAIL:
        return None, None, None
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced",
                                  solver="lbfgs", C=1.0)),
    ])

    use_cluster = cluster_ids is not None and len(cluster_ids) == len(df)
    if use_cluster:
        from sklearn.model_selection import GroupKFold
        gkf    = GroupKFold(n_splits=k)
        splits = list(gkf.split(X, y, groups=cluster_ids))
    else:
        from sklearn.model_selection import StratifiedKFold
        skf    = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
        splits = list(skf.split(X, y))

    probs    = np.full(len(y), float("nan"))
    for tr, va in splits:
        if y[tr].sum() == 0 or y[tr].sum() == len(tr):
            continue
        if y[va].sum() == 0:
            continue
        pipe.fit(X[tr], y[tr])
        probs[va] = pipe.predict_proba(X[va])[:, 1]

    valid = ~np.isnan(probs)
    val_clusters = cluster_ids[valid] if use_cluster else None
    return probs[valid], y[valid], val_clusters


def delta_ci(y_true, score_early, score_full, n_boot=N_BOOT, seed=SEED,
             cluster_ids: Optional[np.ndarray] = None):
    """
    Paired bootstrap CI on Delta = PR-AUC(early) - PR-AUC(full).
    When cluster_ids is provided, resamples cluster IDs so that all
    completions from the same question move together — correctly accounting
    for within-question correlation in multi-completion datasets.
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    use_cluster = cluster_ids is not None and len(cluster_ids) == n
    deltas = []
    for _ in range(n_boot):
        if use_cluster:
            idx = _cluster_resample(rng, cluster_ids)
        else:
            idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            pe = average_precision_score(yt, score_early[idx])
            pf = average_precision_score(yt, score_full[idx])
            deltas.append(pe - pf)
        except Exception:
            continue
    if not deltas:
        return float("nan"), float("nan"), float("nan"), float("nan")
    d = np.array(deltas)
    return (float(np.mean(d)), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)), float(np.mean(d > 0)))


def verdict(delta, lo, hi, p):
    if any(np.isnan(x) for x in [delta, lo, hi, p]):
        return "N/A"
    if lo > 0:
        return "✓ Committed"
    if hi < 0:
        return "✗ Full better"
    if p > 0.8 and delta > 0:
        return "~ Near-committed"
    return "? Ambiguous"


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def banner(title, width=76):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def section(title, width=76):
    print(f"\n{'─'*width}\n  {title}\n{'─'*width}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 – Overview
# ─────────────────────────────────────────────────────────────────────────────

def section_overview(df: pd.DataFrame, windows: List[int], model_name: str):
    banner(f"SECTION 1 — Dataset Overview  [{model_name}]")
    n   = len(df)
    nf  = int(df["y"].sum())
    acc = 1 - nf / n
    gl  = df["generated_len"].mean() if "generated_len" in df.columns else (
          df["full_len"].mean() if "full_len" in df.columns else float("nan"))

    pl_col = next((c for c in ["prefinal_len", "preboxed_generated_len"] if c in df.columns), None)
    if pl_col:
        pvalid = (df[pl_col] > 0).sum()
        print(f"\n  n={n}  failures={nf} ({100*nf/n:.1f}%)  accuracy={100*acc:.1f}%")
        print(f"  mean_gen_len={gl:.0f}  prefinal_valid={pvalid}/{n} ({100*pvalid/n:.1f}%)")
    else:
        print(f"\n  n={n}  failures={nf} ({100*nf/n:.1f}%)  accuracy={100*acc:.1f}%")
        print(f"  mean_gen_len={gl:.0f}  (no prefinal column found)")

    print(f"\n  Windows detected: {windows}")
    print(f"  Full-trace columns available: {len(full_cols(df))}")

    # window coverage
    print(f"\n  {'Window':<10} {'Cols available':>16} {'Mean window len':>18}")
    print("  " + "─" * 48)
    for T in windows:
        wc  = window_cols(df, T)
        lc  = f"window_len_w{T}"
        wl  = df[lc].mean() if lc in df.columns else float("nan")
        wl_str = f"{wl:.0f}" if not np.isnan(wl) else "N/A"
        print(f"  T={T:<8} {len(wc):>16} {wl_str:>18}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 – Window ablation
# ─────────────────────────────────────────────────────────────────────────────

def section_window_ablation(df: pd.DataFrame, windows: List[int],
                            cluster_ids: Optional[np.ndarray] = None) -> List[dict]:
    banner("SECTION 2 — Window Ablation: Early vs Full-Trace PR-AUC")

    n        = len(df)
    nf       = int(df["y"].sum())
    baseline = nf / n
    fc       = full_cols(df)

    # compute full-trace OOF scores once
    full_probs, full_y, full_cids = cv_oof_scores(df, fc, cluster_ids=cluster_ids)
    pa_f, lo_f, hi_f, au_f = cv_prauc(df, fc, cluster_ids=cluster_ids)

    ci_note = " (cluster-robust)" if cluster_ids is not None else ""
    print(f"\n  Baseline PR-AUC (random): {baseline:.4f}")
    print(f"  Full-trace PR-AUC: {pa_f:.4f}  95% CI [{lo_f:.4f}, {hi_f:.4f}]{ci_note}  AUROC={au_f:.4f}\n")

    # ── 2a. Main table ──
    section("2a. Early-Window PR-AUC Table")
    hdr = (f"  {'Window':<8} {'Early PR-AUC':>14} {'95% CI':>22} "
           f"{'Early AUROC':>13} {'Full PR-AUC':>13}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    rows = []
    for T in windows:
        wc = window_cols(df, T)
        pa_e, lo_e, hi_e, au_e = cv_prauc(df, wc, cluster_ids=cluster_ids)
        ep  = f"{pa_e:.4f}" if not np.isnan(pa_e) else " N/A  "
        ci  = f"[{lo_e:.4f}, {hi_e:.4f}]" if not np.isnan(lo_e) else "[  N/A  ]"
        au  = f"{au_e:.4f}" if not np.isnan(au_e) else " N/A  "
        fp  = f"{pa_f:.4f}" if not np.isnan(pa_f) else " N/A  "
        print(f"  T={T:<6} {ep:>14} {ci:>22} {au:>13} {fp:>13}")
        rows.append(dict(T=T, early_prauc=pa_e, early_lo=lo_e, early_hi=hi_e,
                         early_auroc=au_e, full_prauc=pa_f))

    # ── 2b. Delta CI table ──
    section("2b. Delta PR-AUC (Early − Full) with Bootstrap 95% CI")
    n_oof = len(full_probs) if full_probs is not None else 0
    print(f"\n  Full-trace OOF scores computed from {n_oof} samples{ci_note}\n")
    print(f"  {'Window':<8} {'ΔPR-AUC':>10} {'95% CI':>24} {'p(early>full)':>16} {'Verdict':>18}")
    print("  " + "─" * 82)

    for r in rows:
        T  = r["T"]
        wc = window_cols(df, T)
        e_probs, e_y, e_cids = cv_oof_scores(df, wc, cluster_ids=cluster_ids)

        if e_probs is not None and full_probs is not None and len(e_probs) == len(full_probs):
            d, dlo, dhi, p = delta_ci(full_y, e_probs, full_probs,
                                      cluster_ids=full_cids)
        else:
            d, dlo, dhi, p = float("nan"), float("nan"), float("nan"), float("nan")

        r.update(dict(delta=d, delta_lo=dlo, delta_hi=dhi, p_early_gt_full=p))

        if np.isnan(d):
            print(f"  T={T:<6}  N/A")
            continue
        ci = f"[{dlo:+.4f}, {dhi:+.4f}]"
        v  = verdict(d, dlo, dhi, p)
        print(f"  T={T:<6} {d:>+10.4f} {ci:>24} {p:>16.3f} {v:>18}")

    # commitment-point summary
    valid = [r for r in rows if not np.isnan(r["early_prauc"])]
    if valid:
        best = max(valid, key=lambda r: r["early_prauc"])
        print(f"\n  → Best early window: T={best['T']}  (Early PR-AUC={best['early_prauc']:.4f})")
        if any(r.get("delta_lo", float("nan")) > 0 for r in rows):
            print("  → Inverted-U committed failure signature PRESENT ✓")
        elif any(verdict(r.get("delta", float("nan")), r.get("delta_lo", float("nan")),
                         r.get("delta_hi", float("nan")), r.get("p_early_gt_full", float("nan")))
                 == "~ Near-committed" for r in rows):
            print("  → Near-committed — positive delta, p>0.8 but CI spans zero")
        else:
            print("  → Persistent uncertainty — full trace dominates across all windows")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 – Prefinal ablation
# ─────────────────────────────────────────────────────────────────────────────

def section_prefinal_ablation(df: pd.DataFrame, windows: List[int],
                              cluster_ids: Optional[np.ndarray] = None):
    banner("SECTION 3 — Prefinal Ablation  [Length Confound Check]")

    pl_col = next((c for c in ["prefinal_len", "preboxed_generated_len"] if c in df.columns), None)
    if pl_col is None:
        print("  ⚠  No prefinal length column found. Skipping.")
        return []

    valid_rows = df[df[pl_col] > 0].copy()
    n_valid    = len(valid_rows)
    nf_valid   = int(valid_rows["y"].sum())
    print(f"\n  Prefinal-valid rows: {n_valid}/{len(df)} ({100*n_valid/len(df):.1f}%)")
    print(f"  Failures in prefinal subset: {nf_valid} ({100*nf_valid/n_valid:.1f}%)\n")

    # Subset cluster_ids to prefinal-valid rows
    sub_cids = None
    if cluster_ids is not None:
        mask     = (df[pl_col] > 0).values
        sub_cids = cluster_ids[mask]

    # check prefinal feature columns exist
    pfc = full_cols(df, prefinal=True)
    if not pfc:
        pfc = [c for c in df.columns if c.endswith("_prefinal") and
               any(c.startswith(b) for b in BASE_FEATURES)]
    if not pfc:
        print("  ⚠  No prefinal full-trace feature columns found.")
        print("     Expected columns like entropy_mean_prefinal etc.")
        return []

    print(f"  Prefinal full-trace features: {len(pfc)}")

    # full prefinal trace
    pa_pf, lo_pf, hi_pf, au_pf    = cv_prauc(valid_rows, pfc, cluster_ids=sub_cids)
    full_probs_p, full_y_p, fc_ids = cv_oof_scores(valid_rows, pfc, cluster_ids=sub_cids)

    # ── 3a. Table ──
    section("3a. Prefinal Early-Window PR-AUC Table")
    print(f"  Prefinal full-trace PR-AUC: {pa_pf:.4f}  [{lo_pf:.4f}, {hi_pf:.4f}]\n")
    print(f"  {'Window':<8} {'Early PR-AUC':>14} {'95% CI':>22} "
          f"{'Early AUROC':>13} {'Beats full?':>13}")
    print("  " + "─" * 74)

    pre_rows = []
    for T in windows:
        pwc = [f + f"_prefinal_w{T}" for f in BASE_FEATURES]
        pwc = _avail(valid_rows, pwc)
        if not pwc:
            pwc = [f + f"_w{T}_prefinal" for f in BASE_FEATURES]
            pwc = _avail(valid_rows, pwc)
        if not pwc:
            print(f"  T={T:<6}  no prefinal window columns found")
            continue

        pa_e, lo_e, hi_e, au_e = cv_prauc(valid_rows, pwc, cluster_ids=sub_cids)
        beats = "Yes ✓" if (not np.isnan(pa_e) and not np.isnan(pa_pf) and pa_e > pa_pf) else "No"
        ep  = f"{pa_e:.4f}" if not np.isnan(pa_e) else " N/A  "
        ci  = f"[{lo_e:.4f}, {hi_e:.4f}]" if not np.isnan(lo_e) else "[  N/A  ]"
        au  = f"{au_e:.4f}" if not np.isnan(au_e) else " N/A  "
        print(f"  T={T:<6} {ep:>14} {ci:>22} {au:>13} {beats:>13}")
        pre_rows.append(dict(T=T, early_prauc=pa_e, full_prauc=pa_pf))

    # ── 3b. Delta CI for prefinal ──
    section("3b. Prefinal Delta PR-AUC Table")
    print(f"\n  {'Window':<8} {'ΔPR-AUC':>10} {'95% CI':>24} {'p(early>full)':>16} {'Verdict':>18}")
    print("  " + "─" * 82)

    for r in pre_rows:
        T   = r["T"]
        pwc = [f + f"_prefinal_w{T}" for f in BASE_FEATURES]
        pwc = _avail(valid_rows, pwc)
        e_probs_p, e_y_p, _ = cv_oof_scores(valid_rows, pwc, cluster_ids=sub_cids) if pwc else (None, None, None)

        if e_probs_p is not None and full_probs_p is not None and len(e_probs_p) == len(full_probs_p):
            d, dlo, dhi, p = delta_ci(full_y_p, e_probs_p, full_probs_p,
                                      cluster_ids=fc_ids)
        else:
            d, dlo, dhi, p = float("nan"), float("nan"), float("nan"), float("nan")

        if np.isnan(d):
            print(f"  T={T:<6}  N/A"); continue
        ci = f"[{dlo:+.4f}, {dhi:+.4f}]"
        v  = verdict(d, dlo, dhi, p)
        print(f"  T={T:<6} {d:>+10.4f} {ci:>24} {p:>16.3f} {v:>18}")

    # ── 3c. Commitment point comparison ──
    section("3c. Prefinal vs Regular — Commitment Point Comparison")
    valid_pre = [r for r in pre_rows if not np.isnan(r["early_prauc"])]
    if valid_pre:
        best_pre = max(valid_pre, key=lambda r: r["early_prauc"])
        print(f"  Prefinal best early window: T={best_pre['T']}  "
              f"(PR-AUC={best_pre['early_prauc']:.4f})")
        if any(r["early_prauc"] > r["full_prauc"] for r in valid_pre
               if not np.isnan(r["full_prauc"])):
            print("  → Inverted-U signature SURVIVES prefinal stripping ✓")
            print("    Length confound is NOT driving the signal.")
        else:
            print("  → Early never beats full in prefinal analysis.")
            print("    Inverted-U (if any in regular) may be length-confounded,")
            print("    OR model is in persistent uncertainty regime.")

    return pre_rows


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 – Per-feature breakdown
# ─────────────────────────────────────────────────────────────────────────────

def section_feature_breakdown(df: pd.DataFrame, windows: List[int],
                              cluster_ids: Optional[np.ndarray] = None):
    banner("SECTION 4 — Per-Feature Breakdown (Mean vs Max, Individual Features)")

    print("\n  Individual feature PR-AUC (CV-LR, single feature at a time)\n")
    hdr = f"  {'Feature':<24} {'Full':>8}" + "".join(f"{'T='+str(T):>10}" for T in windows)
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    results = {}
    for feat in sorted(BASE_FEATURES):
        # full trace
        fc = feat if feat in df.columns else None
        pa_full = cv_prauc(df, [fc], cluster_ids=cluster_ids)[0] if fc else float("nan")
        row_str = f"  {feat:<24} {pa_full:>8.4f}" if not np.isnan(pa_full) else f"  {feat:<24} {'N/A':>8}"
        vals = [pa_full]
        for T in windows:
            col = f"{feat}_w{T}"
            if col not in df.columns:
                row_str += f"{'N/A':>10}"; vals.append(float("nan")); continue
            pa, *_ = cv_prauc(df, [col], cluster_ids=cluster_ids)
            row_str += f"{pa:>10.4f}" if not np.isnan(pa) else f"{'N/A':>10}"
            vals.append(pa)
        print(row_str)
        results[feat] = vals  # [full, T1, T2, ...]

    # ── 4b. Inverted-U detection ──
    section("4b. Inverted-U Signature per Feature")
    print(f"\n  {'Feature':<24} {'Best T':>10} {'Best PA':>10} {'Full PA':>10} {'Shape':>16}")
    print("  " + "─" * 74)
    for feat, vals in sorted(results.items()):
        full_pa = vals[0]
        window_vals = list(zip(windows, vals[1:]))
        valid_w = [(T, v) for T, v in window_vals if not np.isnan(v)]
        if not valid_w:
            print(f"  {feat:<24} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>16}"); continue
        best_T, best_pa = max(valid_w, key=lambda x: x[1])
        fp_str = f"{full_pa:.4f}" if not np.isnan(full_pa) else "N/A"
        if best_pa > full_pa and not np.isnan(full_pa):
            shape = "Early>Full ✓"
        elif best_T == windows[-1]:
            shape = "Monotone ↑"
        elif best_T == windows[0]:
            shape = "First peak"
        else:
            shape = "Inverted-U ✓"
        print(f"  {feat:<24} {best_T:>10} {best_pa:>10.4f} {fp_str:>10} {shape:>16}")

    # ── 4c. Max vs mean aggregate ──
    section("4c. Aggregate: Max Features vs Mean Features")
    max_full  = _avail(df, MAX_FEATURES)
    mean_full = _avail(df, MEAN_FEATURES)
    print(f"\n  {'':6} {'Max (full)':>12} {'Mean (full)':>13}" +
          "".join(f"{'T='+str(T):>22}" for T in windows))
    print(f"  {'':6} {'':>12} {'':>13}" +
          "".join(f"{'max / mean':>22}" for _ in windows))
    print("  " + "─" * (34 + 22 * len(windows)))

    pa_mxf = cv_prauc(df, max_full, cluster_ids=cluster_ids)[0]  if max_full  else float("nan")
    pa_mnf = cv_prauc(df, mean_full, cluster_ids=cluster_ids)[0] if mean_full else float("nan")
    row = f"  {'':6} {pa_mxf:>12.4f} {pa_mnf:>13.4f}"
    for T in windows:
        mxc = _avail(df, [f + f"_w{T}" for f in MAX_FEATURES])
        mnc = _avail(df, [f + f"_w{T}" for f in MEAN_FEATURES])
        pa_mx = cv_prauc(df, mxc, cluster_ids=cluster_ids)[0] if mxc else float("nan")
        pa_mn = cv_prauc(df, mnc, cluster_ids=cluster_ids)[0] if mnc else float("nan")
        cell = f"{pa_mx:.4f}/{pa_mn:.4f}" if not (np.isnan(pa_mx) or np.isnan(pa_mn)) else "N/A"
        row += f"{cell:>22}"
    print(row)


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 – Summary + plot
# ─────────────────────────────────────────────────────────────────────────────

def section_summary(df, windows, rows, model_name):
    banner("SECTION 5 — Summary and Commitment-Point Verdict")
    n  = len(df)
    nf = int(df["y"].sum())
    fc = full_cols(df)
    pa_f, *_ = cv_prauc(df, fc)

    print(f"\n  Model: {model_name}")
    print(f"  N={n}  failures={nf} ({100*nf/n:.1f}%)  accuracy={100*(1-nf/n):.1f}%")
    print(f"  Full-trace PR-AUC: {pa_f:.4f}\n")

    print(f"  {'Window':<8} {'Early PA':>10} {'Full PA':>10} {'Delta':>8} "
          f"{'95% CI':>22} {'p(E>F)':>8} {'Verdict':>18}")
    print("  " + "─" * 90)

    for r in rows:
        if np.isnan(r.get("delta", float("nan"))):
            ep = f"{r['early_prauc']:.4f}" if not np.isnan(r['early_prauc']) else "N/A"
            print(f"  T={r['T']:<6} {ep:>10}  (delta N/A)"); continue
        ci = f"[{r['delta_lo']:+.4f}, {r['delta_hi']:+.4f}]"
        v  = verdict(r["delta"], r["delta_lo"], r["delta_hi"], r["p_early_gt_full"])
        ep = f"{r['early_prauc']:.4f}" if not np.isnan(r["early_prauc"]) else "N/A"
        fp = f"{r['full_prauc']:.4f}"  if not np.isnan(r["full_prauc"])  else "N/A"
        print(f"  T={r['T']:<6} {ep:>10} {fp:>10} {r['delta']:>+8.4f} "
              f"{ci:>22} {r['p_early_gt_full']:>8.3f} {v:>18}")

    committed = [r for r in rows if r.get("delta_lo", float("nan")) > 0]
    near = [r for r in rows
            if verdict(r.get("delta", float("nan")), r.get("delta_lo", float("nan")),
                       r.get("delta_hi", float("nan")), r.get("p_early_gt_full", float("nan")))
            == "~ Near-committed"]
    print()
    if committed:
        best = max(committed, key=lambda r: r["early_prauc"])
        print(f"  ┌─ VERDICT: TYPE 1 — COMMITTED FAILURE")
        print(f"  │  Commitment point: T={best['T']}  "
              f"(Early={best['early_prauc']:.4f} > Full={best['full_prauc']:.4f})")
        print(f"  └─ Early window provides strictly better failure prediction.")
    elif near:
        best = max(rows, key=lambda r: r["early_prauc"] if not np.isnan(r["early_prauc"]) else -1)
        print(f"  ┌─ VERDICT: NEAR-COMMITTED / AMBIGUOUS")
        print(f"  │  Best window T={best['T']} (Early={best['early_prauc']:.4f}), "
              f"positive delta but CI spans zero.")
        print(f"  └─ Consistent with committed failure; more data would resolve.")
    else:
        print(f"  ┌─ VERDICT: TYPE 2 — PERSISTENT UNCERTAINTY")
        print(f"  │  Full-trace always statistically better than early window.")
        print(f"  └─ Uncertainty accumulates throughout the reasoning trace.")


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation(df, windows, rows, model_name, out_prefix, force_mode=None):
    """
    Plots the PREFINAL window ablation — early prefinal PR-AUC vs prefinal
    full-trace PR-AUC. This is the primary plot because it strips post-answer
    tokens, eliminating the length confound and revealing the true failure mode.
    Falls back to the regular ablation if no prefinal columns are found.
    """
    EARLY_COLOR  = "#2166AC"   # deep blue — early prefinal curve
    FULL_COLOR   = "#D6604D"   # coral   — prefinal full-trace reference
    SHADE_COLOR  = "#AEC7E8"   # light blue CI band
    BASE_COLOR   = "#999999"   # grey baseline

    plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    # ── Determine whether prefinal data exists ────────────────────────────────
    # If force_mode="prefinal", the df passed in has already been remapped so
    # raw features ARE the prefinal features. Plot them directly as primary.
    # If force_mode="regular" or None, auto-detect as before.
    pl_col = next((c for c in ["prefinal_len", "preboxed_generated_len"]
                   if c in df.columns), None)
    pfc    = full_cols(df, prefinal=True)

    if force_mode == "prefinal":
        # df is already the prefinal-remapped subset; raw cols ARE prefinal cols
        sub      = df
        fc       = full_cols(df, prefinal=False)  # remapped to prefinal by _prepare
        pa_f, lo_f, hi_f, _ = cv_prauc(sub, fc)
        baseline = sub["y"].mean()
        plot_rows = [dict(T=r["T"], pa=r["early_prauc"],
                          lo=r["early_lo"], hi=r["early_hi"])
                     for r in rows if not np.isnan(r["early_prauc"])]
        label_early  = "Early-window"
        label_full   = f"Full-trace ({pa_f:.3f})"
        title_suffix = "Prefinal Window Ablation"
        suffix_note  = "Post-answer tokens stripped (prefinal). Shaded: 95% bootstrap CI."
        file_suffix  = "_prefinal_ablation"
        use_prefinal = True
    elif (pl_col is not None and pfc and (df[pl_col] > 0).sum() >= 10):
        sub      = df[df[pl_col] > 0].copy()
        pa_f, lo_f, hi_f, _ = cv_prauc(sub, pfc)
        baseline = sub["y"].mean()
        plot_rows = []
        for T in windows:
            pwc = _avail(sub, [f + f"_prefinal_w{T}" for f in BASE_FEATURES])
            if not pwc:
                continue
            pa_e, lo_e, hi_e, _ = cv_prauc(sub, pwc)
            plot_rows.append(dict(T=T, pa=pa_e, lo=lo_e, hi=hi_e))
        label_early    = "Early-window"
        label_full     = f"Full-trace ({pa_f:.3f})"
        title_suffix   = "Prefinal Window Ablation"
        suffix_note    = "Post-answer tokens stripped. Shaded: 95% bootstrap CI."
        file_suffix    = "_prefinal_ablation"
        use_prefinal   = True
    else:
        # fallback to regular ablation
        sub      = df
        fc       = full_cols(df)
        pa_f, lo_f, hi_f, _ = cv_prauc(sub, fc)
        baseline = sub["y"].mean()
        plot_rows = [dict(T=r["T"], pa=r["early_prauc"],
                          lo=r["early_lo"], hi=r["early_hi"])
                     for r in rows if not np.isnan(r["early_prauc"])]
        label_early    = "Early-window"
        label_full     = f"Full-trace ({pa_f:.3f})"
        title_suffix   = "Window Ablation"
        suffix_note    = "Shaded: 95% bootstrap CI."
        file_suffix    = "_window_ablation"
        use_prefinal   = False

    if not plot_rows:
        print("  ⚠  No data to plot.")
        return

    Ts       = np.array([r["T"]  for r in plot_rows if not np.isnan(r["pa"])])
    early_pa = np.array([r["pa"] for r in plot_rows if not np.isnan(r["pa"])])
    lo_ci    = np.array([r["lo"] for r in plot_rows if not np.isnan(r["pa"])])
    hi_ci    = np.array([r["hi"] for r in plot_rows if not np.isnan(r["pa"])])

    # ── Paper figure constants — keep consistent across all plots ────────────
    FONT_SIZE   = 9    # axis labels, tick labels, legend, annotations
    FIG_W, FIG_H = 3.5, 2.6   # single-column figure (inches)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax  = fig.add_subplot(111)

    # CI shading
    ax.fill_between(Ts, lo_ci, hi_ci, color=SHADE_COLOR, alpha=0.45, linewidth=0)

    # Early prefinal curve
    ax.plot(Ts, early_pa, color=EARLY_COLOR, linewidth=1.8, marker="o",
            markersize=4.5, markerfacecolor="white", markeredgewidth=1.5,
            markeredgecolor=EARLY_COLOR, label=label_early, zorder=4)

    # Prefinal full-trace reference (flat line — single trace, single value)
    ax.axhline(pa_f, color=FULL_COLOR, linewidth=1.4, linestyle="--",
               label=label_full, zorder=3)

    # Baseline
    ax.axhline(baseline, color=BASE_COLOR, linewidth=1.0, linestyle=":",
               label=f"Chance ({baseline:.3f})", zorder=2)

    # Commitment point — only annotate if peak T is strictly less than the
    # median trace length, meaning the model locked in before the trace ended.
    best_idx    = int(np.argmax(early_pa))
    best_T      = int(Ts[best_idx])
    best_pa_val = float(early_pa[best_idx])

    len_col = next((c for c in ["prefinal_len", "preboxed_generated_len",
                                 "generated_len", "full_len"]
                    if c in sub.columns), None)
    median_len = float(sub[len_col].median()) if len_col is not None else np.inf
    is_commitment_point = best_T < median_len

    if is_commitment_point:
        ax.axvline(best_T, color=EARLY_COLOR, linewidth=0.9,
                   linestyle="-.", alpha=0.55, zorder=1)
        span = float(hi_ci.max() - lo_ci.min()) if len(hi_ci) > 0 else 0.05
        # On log scale, offset in log-space so it looks natural
        log_range = np.log10(Ts[-1]) - np.log10(Ts[0])
        log_offset = log_range * 0.06
        log_best = np.log10(best_T)
        x_text = 10 ** (log_best + log_offset) if log_best + log_offset < np.log10(Ts[-1]) * 0.95 \
                 else 10 ** (log_best - log_offset * 4)
        ax.annotate(
            f"  T={best_T}",
            xy=(best_T, best_pa_val),
            xytext=(x_text, best_pa_val - span * 0.25),
            fontsize=FONT_SIZE, color=EARLY_COLOR,
            arrowprops=dict(arrowstyle="-", color=EARLY_COLOR, lw=0.7, alpha=0.7),
        )

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xlabel("Early-window size $T$ (tokens)", fontsize=FONT_SIZE)
    ax.set_ylabel("PR-AUC", fontsize=FONT_SIZE)

    # Log x-axis — spreads dense low-T ticks naturally, no rotation needed
    ax.set_xscale("log")
    ax.set_xticks(Ts)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(axis="x", labelsize=FONT_SIZE, which="both")
    ax.tick_params(axis="y", labelsize=FONT_SIZE, which="both")

    # Dynamic y-axis: zoom strictly to curve + CI + full-trace line.
    # Baseline is shown as a line but excluded from ylim so it doesn't
    # pull the view down into empty space.
    y_data = np.concatenate([early_pa, lo_ci, hi_ci, [pa_f]])
    y_min_data = float(np.nanmin(y_data))
    y_max_data = float(np.nanmax(y_data))
    y_pad = max((y_max_data - y_min_data) * 0.12, 0.02)
    y_lo = max(0.0, y_min_data - y_pad)
    y_hi = min(1.0, y_max_data + y_pad)
    ax.set_ylim(y_lo, y_hi)

    # Y-axis: major ticks only (no minor ticks, no minor grid lines)
    y_span = y_hi - y_lo
    major_step = 0.05 if y_span > 0.15 else 0.02
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.MultipleLocator(major_step))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", which="minor", left=False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)

    # CI shading legend entry
    ci_patch = matplotlib.patches.Patch(facecolor=SHADE_COLOR, alpha=0.45,
                                        linewidth=0, label="95% CI")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(ci_patch)
    labels.append("95% CI")
    ax.legend(handles, labels, loc="lower right", fontsize=FONT_SIZE - 2,
              frameon=True, framealpha=0.85, edgecolor="#cccccc", borderpad=0.2,
              handlelength=1.0, handletextpad=0.2, labelspacing=0.1,
              borderaxespad=0.2)

    fig.tight_layout()
    pdf_path = f"{out_prefix}{file_suffix}.pdf"
    png_path = f"{out_prefix}{file_suffix}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    mode = "prefinal" if use_prefinal else "regular (prefinal not available)"
    print(f"\n  Plot saved ({mode}):")
    print(f"    PDF → {pdf_path}")
    print(f"    PNG → {png_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GSM8K/MATH/GPQA single-file multi-window analysis")
    p.add_argument("--file",        required=True,
                   help="Single CSV/XLSX file with all window features as columns")
    p.add_argument("--windows",     nargs="*", type=int, default=None,
                   help="Window sizes to analyse (auto-detected if omitted)")
    p.add_argument("--model_name",  default="Model",
                   help="Human-readable model name for display and plot title")
    p.add_argument("--out_prefix",  default=None,
                   help="Output prefix for saved plots (default: model_name_gsm8k)")
    p.add_argument("--n_boot",      type=int, default=N_BOOT,
                   help="Bootstrap iterations (default 1000)")
    p.add_argument("--no_plot",     action="store_true",
                   help="Skip plot generation")
    p.add_argument("--prefinal",    action="store_true",
                   help="[RECOMMENDED] Run entire analysis on prefinal features. "
                        "Early = tokens 0→T of prefinal trace; "
                        "Full = all tokens up to Final Answer cutoff. "
                        "Both sides operate on same token pool, eliminating "
                        "the length confound in ALL sections, not just Section 3.")
    p.add_argument("--cluster_col", default=None,
                   help="Column name containing cluster/group IDs (e.g. 'question_id'). "
                        "When set, bootstrap resamples CLUSTERS rather than rows and "
                        "CV uses GroupKFold so all completions of a question stay in "
                        "the same fold. Use for multi-completion datasets to get honest "
                        "CIs that account for within-question correlation.")
    return p.parse_args()


def _prepare_df_for_analysis(df: pd.DataFrame, prefinal: bool):
    """
    When --prefinal is set, restrict the dataframe to prefinal-valid rows
    and remap column names so that the rest of the analysis functions
    transparently use prefinal features as if they were the primary features.

    Mapping applied:
      {feat}_prefinal       →  {feat}            (full-trace baseline)
      {feat}_prefinal_w{T}  →  {feat}_w{T}       (early window features)
      prefinal_len          →  used as validity filter (rows where > 0)

    The original columns are preserved under their original names so that
    Section 3 (robustness check) can still compare against them if needed.
    Returns (working_df, mode_label).
    """
    import re as _re
    if not prefinal:
        return df, "regular"

    pl_col = next((c for c in ["prefinal_len", "preboxed_generated_len"]
                   if c in df.columns), None)
    if pl_col is None:
        print("  ⚠  --prefinal requested but no prefinal_len column found.")
        print("     Falling back to regular (raw) features.")
        return df, "regular"

    sub = df[df[pl_col] > 0].copy()
    if sub["y"].sum() < MIN_FAIL:
        print(f"  ⚠  Only {int(sub['y'].sum())} failures in prefinal subset. "
              "Falling back to regular features.")
        return df, "regular"

    # Remap _prefinal full-trace columns → primary full-trace columns
    for feat in BASE_FEATURES:
        src_full = f"{feat}_prefinal"
        if src_full in sub.columns:
            sub[feat] = sub[src_full]

    # Remap _prefinal_w{T} columns → _w{T} columns
    for col in list(sub.columns):
        m = _re.match(r"^(.+)_prefinal_w(\d+)$", col)
        if m:
            base, T = m.group(1), m.group(2)
            new_col = f"{base}_w{T}"
            sub[new_col] = sub[col]

    n_valid = len(sub)
    nf      = int(sub["y"].sum())
    print(f"\n  [Prefinal mode] Restricted to {n_valid} prefinal-valid rows "
          f"({nf} failures, {100*nf/n_valid:.1f}% failure rate)")
    print(f"  Early features = tokens 0→T of prefinal trace")
    print(f"  Full  features = all tokens up to Final Answer cutoff\n")

    return sub, "prefinal"


def main():
    args = parse_args()
    global N_BOOT
    N_BOOT = args.n_boot

    print(f"Loading: {args.file}", flush=True)
    df = load_file(args.file)

    windows = args.windows if args.windows else detect_windows(df)
    if not windows:
        print("ERROR: no window columns found and --windows not specified.")
        sys.exit(1)

    out_prefix = args.out_prefix or args.model_name.replace(" ", "_").lower() + "_gsm8k"

    # ── Build cluster ID array ────────────────────────────────────────────────
    cluster_ids = None
    if args.cluster_col:
        if args.cluster_col not in df.columns:
            print(f"ERROR: --cluster_col '{args.cluster_col}' not found in file.")
            print(f"  Available columns: {list(df.columns[:20])}")
            sys.exit(1)
        cluster_ids = pd.Categorical(df[args.cluster_col]).codes.astype(np.int32)
        n_clusters  = int(np.unique(cluster_ids).shape[0])
        print(f"  Clustering on '{args.cluster_col}': {n_clusters} unique clusters, "
              f"{len(df)} rows ({len(df)/n_clusters:.1f} rows/cluster avg)")

    banner(f"Single-File Multi-Window Analysis — {args.model_name}", width=76)
    print(f"  File: {args.file}")
    print(f"  Windows: {windows}")
    print(f"  Bootstrap iterations: {N_BOOT}")
    print(f"  Mode: {'PREFINAL (recommended)' if args.prefinal else 'regular (use --prefinal for primary analysis)'}")
    if cluster_ids is not None:
        print(f"  CIs: cluster-robust (resampling by '{args.cluster_col}')")
    else:
        print(f"  CIs: standard i.i.d. bootstrap")

    # Prepare working dataframe — remap prefinal cols to primary if --prefinal
    working_df, mode = _prepare_df_for_analysis(df, args.prefinal)

    # Re-derive cluster_ids for the working_df (may be a filtered subset)
    working_cids = None
    if cluster_ids is not None:
        if args.cluster_col in working_df.columns:
            working_cids = pd.Categorical(
                working_df[args.cluster_col]).codes.astype(np.int32)
        else:
            # Fall back to positional alignment if column was dropped
            working_cids = cluster_ids[working_df.index]

    section_overview(working_df, windows, args.model_name)
    rows = section_window_ablation(working_df, windows, cluster_ids=working_cids)

    if args.prefinal:
        banner("SECTION 3 — Length Confound Reference  [Raw full-trace vs Prefinal-early]")
        print("\n  NOTE: In --prefinal mode, Sections 2/4/5 already use the correct")
        print("  prefinal-vs-prefinal comparison. This section shows the raw full-trace")
        print("  PR-AUC for reference only (demonstrates the length confound).")
        raw_fc = full_cols(df)
        pa_raw, lo_raw, hi_raw, _ = cv_prauc(df, raw_fc, cluster_ids=cluster_ids)
        print(f"\n  Raw full-trace PR-AUC (all rows): {pa_raw:.4f}  [{lo_raw:.4f}, {hi_raw:.4f}]")
        if "generated_len" in df.columns:
            gl_prauc, gl_lo, gl_hi, _ = cv_prauc(df, ["generated_len"],
                                                  cluster_ids=cluster_ids)
            print(f"  generated_len PR-AUC (length confound proxy): "
                  f"{gl_prauc:.4f}  [{gl_lo:.4f}, {gl_hi:.4f}]")
    else:
        section_prefinal_ablation(working_df, windows, cluster_ids=working_cids)

    section_feature_breakdown(working_df, windows, cluster_ids=working_cids)
    section_summary(working_df, windows, rows, args.model_name)

    if not args.no_plot:
        plot_ablation(working_df, windows, rows, args.model_name, out_prefix,
                      force_mode=mode)

    print("\n" + "=" * 76)
    print("  Analysis complete.")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()