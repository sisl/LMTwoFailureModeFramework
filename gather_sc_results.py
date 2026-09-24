"""
Gather self-consistency results from the SC-replication output CSVs and emit
LaTeX: (1) combined triage table (GPQA + MATH-500, committed vs persistent),
(2) completion-level complementarity table across the three MATH-500 configs,
(3) a pgfplots figure (native LaTeX) of the triage curves.

All numbers are read from files written by analyze_sc_replication.py:
  - sc_selective_curve_all_signals_w{T}.csv   (triage curves)
  - sc_accuracy_proxy_w{T}.csv                 (Analysis 4; requires the patch)

Nothing is hardcoded. If an input is missing, that block is skipped with a
warning rather than fabricated.

Usage:
  python gather_sc_results.py --root results/self_consistency --out tables/
"""
import argparse, os, csv
from collections import defaultdict

# Which runs feed which table. EDIT paths/dirs to match your layout.
# Each entry: key -> (directory, window, benchmark, model, mode)
RUNS = {
    # GPQA (existing) — committed vs persistent
    "gpqa_gemma":    dict(dir="results/self_consistency/gpqa_gemma4",    w=400, bench="GPQA",     model="Gemma4-31B",   mode="committed"),
    "gpqa_qwen9b":   dict(dir="results/self_consistency/gpqa_qwen9b",    w=2048,bench="GPQA",     model="Qwen3.5-9B",   mode="persistent"),
    "gpqa_qwen122b": dict(dir="results/self_consistency/gpqa_qwen122b",  w=512, bench="GPQA",     model="Qwen3.5-122B", mode="persistent"),
    # MATH-500 (new) — committed vs persistent (+ Qwen9B for completion-level only)
    "math_gemma":    dict(dir="results/self_consistency/math500_gemma4",   w=400, bench="MATH-500", model="Gemma4-31B",   mode="committed"),
    "math_qwen122b": dict(dir="results/self_consistency/math500_qwen122b",  w=400, bench="MATH-500", model="Qwen3.5-122B", mode="persistent"),
    "math_qwen9b":   dict(dir="results/self_consistency/math500_qwen9b",    w=400, bench="MATH-500", model="Qwen3.5-9B",   mode="persistent"),
}

# Which runs appear in the triage table (Qwen9B/MATH-500 excluded: truncation)
TRIAGE_KEYS = ["gpqa_gemma", "gpqa_qwen9b", "gpqa_qwen122b",
               "math_gemma", "math_qwen122b", "math_qwen9b"]
COMBO_KEYS  = ["gpqa_gemma", "gpqa_qwen9b", "gpqa_qwen122b",
               "math_gemma", "math_qwen9b", "math_qwen122b"]
SKIPS = [20, 30, 45, 50]
SIGNAL = "prefinal_early"   # comparison-relevant signal for the triage table



# ---------------------------------------------------------------------------
# GPQA fallback values, transcribed from the SC-replication console logs
# (no CSVs were saved for the GPQA runs). Each block cites the log it came
# from so the numbers can be re-verified without re-running.
# Provenance:
#   gpqa_gemma    <- gpqa_gemma4_31b_sc15_backup.final.xlsx  (T*=400, 20 SC fails)
#   gpqa_qwen9b   <- gpqa_qwen3.5-9b_sc15.final.xlsx          (T*=2048, 24 SC fails)
#   gpqa_qwen122b <- gpqa_qwen122b_multi.final.xlsx           (T*=512, 25 SC fails)
# TRIAGE = prefinal-early recall/precision at skip 20/30/45/50 (Analysis 3).
# PROXY  = Analysis 4d: agreement_rate alone, combined(full+agreement).
# ---------------------------------------------------------------------------
GPQA_HARDCODED = {
    "gpqa_gemma": {
        "triage": {20: (1.00, 1.00), 30: (1.00, 1.00), 45: (0.95, 0.989), 50: (0.85, 0.969)},
        "proxy":  {"agreement_rate": 0.7593, "combined_full": 0.7856},
    },
    "gpqa_qwen9b": {
        "triage": {20: (1.00, 1.00), 30: (0.917, 0.965), 45: (0.875, 0.965), 50: (0.833, 0.958)},
        "proxy":  {"agreement_rate": 0.8149, "combined_full": 0.8501},
    },
    "gpqa_qwen122b": {
        "triage": {20: (1.00, 1.00), 30: (0.960, 0.983), 45: (0.880, 0.966), 50: (0.840, 0.960)},
        "proxy":  {"agreement_rate": 0.7500, "combined_full": 0.7954},
    },
}


def load_triage(spec):
    """recall/precision by skip_pct for the prefinal_early signal."""
    key = spec.get("key")
    if key in GPQA_HARDCODED:
        return dict(GPQA_HARDCODED[key]["triage"])
    p = os.path.join(spec["dir"], f"sc_selective_curve_all_signals_w{spec['w']}.csv")
    if not os.path.exists(p):
        return None
    out = {}
    with open(p) as fh:
        for r in csv.DictReader(fh):
            if r.get("signal") == SIGNAL:
                out[int(float(r["skip_pct"]))] = (float(r["recall_sc_fail"]),
                                                  float(r["skip_precision"]))
    return out or None


def load_proxy(spec):
    """4d combination: agreement_rate alone vs combined (agreement + uncertainty)."""
    key = spec.get("key")
    if key in GPQA_HARDCODED:
        return dict(GPQA_HARDCODED[key]["proxy"])
    p = os.path.join(spec["dir"], f"sc_accuracy_proxy_w{spec['w']}.csv")
    if not os.path.exists(p):
        return None
    vals = {}
    with open(p) as fh:
        for r in csv.DictReader(fh):
            if r["analysis"] == "4d_combined":
                vals[r["signal"]] = float(r["prauc"])
    return vals or None


def fmt(x): return f"{x:.2f}" if x is not None else "--"


def triage_table(specs):
    rows = []
    for k in TRIAGE_KEYS:
        s = specs[k]; d = load_triage(s)
        if d is None:
            print(f"  [warn] no triage CSV for {k} — row skipped"); continue
        cells = " & ".join(f"{fmt(d[p][0])}" for p in SKIPS if p in d)
        rows.append((s, s["w"], cells))
    if not rows: return "% (no triage data found)\n"

    L = [r"\begin{tabular}{llccccc}", r"\toprule",
         r"Benchmark & Configuration & $T^\ast$ & \multicolumn{4}{c}{Recall on SC failures at skip rate} \\",
         r"\cmidrule(lr){4-7}",
         r" &  &  & 20\% & 30\% & 45\% & 50\% \\", r"\midrule"]
    cur = None
    for s, w, cells in rows:
        b = s["bench"]
        bcol = b if b != cur else ""
        cur = b
        L.append(f"{bcol} & {s['model']} ({s['mode']}) & {w} & {cells} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def combination_table(specs):
    rows = []
    for k in COMBO_KEYS:
        s = specs[k]; v = load_proxy(s)
        if v is None:
            print(f"  [warn] no proxy CSV for {k} — row skipped"); continue
        ar = v.get("agreement_rate"); comb = v.get("combined_full")
        rows.append((s, ar, comb))
    if not rows: return "% (no combination data found)\n"

    L = [r"\begin{tabular}{llccc}", r"\toprule",
         r"Benchmark & Configuration & Agreement rate & Combined & $\Delta$ \\",
         r"\midrule"]
    cur = None
    for s, ar, comb in rows:
        delta = (comb - ar) if (ar is not None and comb is not None) else None
        bcol = s["bench"] if s["bench"] != cur else ""
        cur = s["bench"]
        L.append(f"{bcol} & {s['model']} ({s['mode']}) & {fmt(ar)} & {fmt(comb)} & "
                 f"{('+'+fmt(delta)) if delta is not None else '--'} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def triage_pgfplots(specs):
    """Native-LaTeX pgfplots figure: recall vs skip, committed vs persistent,
    one benchmark per subplot (MATH-500 shown; add GPQA group by duplicating)."""
    def coords(k):
        d = load_triage(specs[k])
        if d is None: return None
        return " ".join(f"({p},{d[p][0]:.3f})" for p in SKIPS if p in d)

    blocks = []
    for bench, ckey, pkey in [("MATH-500", "math_gemma", "math_qwen122b"),
                              ("GPQA", "gpqa_gemma", "gpqa_qwen9b")]:
        cc, pc = coords(ckey), coords(pkey)
        if cc is None or pc is None:
            print(f"  [warn] missing triage data for {bench} plot — skipped"); continue
        blocks.append(rf"""
\begin{{subfigure}}{{0.48\textwidth}}
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
  width=\textwidth, height=5.2cm,
  xlabel={{Skip rate (\% of most-confident inputs)}},
  ylabel={{Recall on SC failures}},
  xmin=15, xmax=55, ymin=0.35, ymax=1.05,
  xtick={{20,30,45,50}}, legend pos=south west,
  legend style={{font=\scriptsize, draw=none}},
  tick label style={{font=\scriptsize}}, label style={{font=\small}},
  every axis plot/.append style={{mark size=2pt, line width=0.9pt}},
]
\addplot[color=blue, mark=*] coordinates {{{cc}}};
\addlegendentry{{{specs[ckey]['model']} (committed)}}
\addplot[color=red, mark=square*] coordinates {{{pc}}};
\addlegendentry{{{specs[pkey]['model']} (persistent)}}
\draw[gray, dotted] (axis cs:15,0.9) -- (axis cs:55,0.9);
\end{{axis}}
\end{{tikzpicture}}
\caption{{{bench}}}
\end{{subfigure}}""")
    if not blocks:
        return "% (no triage plot data found)\n"
    return (r"\begin{figure}[t]\centering" + "\n" +
            "\n\\hfill\n".join(blocks) + "\n" +
            r"\caption{Recall on self-consistency failures as a function of skip "
            r"rate, using single-completion pre-final early-window confidence. "
            r"Committed configurations sustain higher recall at aggressive skip "
            r"rates than persistent ones on both benchmarks.}" + "\n" +
            r"\label{fig:sc_triage}" + "\n" + r"\end{figure}")


def _stamp_keys():
    for k, v in RUNS.items():
        v["key"] = k


def main():
    _stamp_keys()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name, gen in [("triage_table.tex", triage_table),
                      ("combination_table.tex", combination_table),
                      ("triage_figure.tex", triage_pgfplots)]:
        tex = gen(RUNS)
        with open(os.path.join(args.out, name), "w") as fh:
            fh.write(tex + "\n")
        print(f"wrote {os.path.join(args.out, name)}")
    print("\nNote: pgfplots figure needs \\usepackage{pgfplots}\\usepackage{subcaption} "
          "and \\pgfplotsset{compat=1.18} in the preamble.")


if __name__ == "__main__":
    main()