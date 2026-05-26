"""
Our GRU-based classifier over a 4-step uncertainty trajectory where each step is one of our window ablations
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


BASE_SIGNALS = [
    "entropy_mean",
    "entropy_max",
    "entropy_std",
    "margin_mean",
    "margin_max",
    "margin_std",
    "nll_mean",
    "nll_max",
    "nll_std",
    "fork_rate",
    "nucleus_size_mean",
    "nucleus_size_max",
    "near_tie_mean",
    "near_tie_max",
]

EARLY_COLS = [f"{s}_early" for s in BASE_SIGNALS]
N_FEATURES = len(BASE_SIGNALS)  # 14
N_STEPS    = 4
STEP_LABELS = ["T=128", "T=256", "T=400", "T=512"]



class GRUClassifier(nn.Module):
    """
    Single-layer GRU over a 4-step uncertainty trajectory.
    """
    def __init__(self, input_size: int, hidden_size: int = 32, dropout: float = 0.3):
        super().__init__()
        self.gru  = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)      
        h = h.squeeze(0)
        h = self.drop(h)
        return torch.sigmoid(self.head(h)).squeeze(-1)


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    mask = (
        (df["hit_max_tokens"] == 0) &
        (df["degenerate"] == 0) &
        (df["boxed_answer"].notna())
    )
    out = df[mask].copy()
    out["label"] = (out["correct"] == 0).astype(int)
    return out.set_index("idx")


def build_4step_sequences(dfs: list, shared_idx: list = None):
    
    # Union of all clean idx across runs
    all_idx = set()
    for df in dfs:
        all_idx.update(df.index.tolist())

    if shared_idx is not None:
        all_idx = all_idx & set(shared_idx)

    all_idx = sorted(all_idx)

    # Build step arrays; fill NaN where run doesn't have this idx so that we don't have to drop the whole thing
    steps = []
    for df in dfs:
        feat = np.full((len(all_idx), len(EARLY_COLS)), np.nan)
        present_pos  = [i for i, idx in enumerate(all_idx) if idx in df.index]
        present_idx  = [all_idx[i] for i in present_pos]
        if present_idx:
            feat[present_pos] = df.loc[present_idx, EARLY_COLS].astype(float).values
        steps.append(feat)

    X = np.stack(steps, axis=1) 


    y = np.full(len(all_idx), np.nan)
    for df in dfs:
        for i, idx in enumerate(all_idx):
            if idx in df.index and np.isnan(y[i]):
                y[i] = df.loc[idx, "label"]
    df256 = dfs[1]
    for i, idx in enumerate(all_idx):
        if idx in df256.index:
            y[i] = df256.loc[idx, "label"]

    # keep missing steps rather than dropping entire examples.
    for i in range(len(all_idx)):
        row = X[i] 
        step_valid = np.isfinite(row).all(axis=1)
        if step_valid.all():
            continue
        if not step_valid.any():
            continue 


        last_valid = None
        for t in range(4):
            if step_valid[t]:
                last_valid = row[t].copy()
            elif last_valid is not None:
                row[t] = last_valid


        first_valid = None
        for t in range(3, -1, -1):
            if step_valid[t]:
                first_valid = row[t].copy()
            elif first_valid is not None and not np.isfinite(row[t]).all():
                row[t] = first_valid

        X[i] = row


    still_nan = ~np.isfinite(X).all(axis=(1, 2))
    label_nan = ~np.isfinite(y)
    valid     = ~still_nan & ~label_nan

    n_imputed = int((~np.isfinite(np.stack(steps, axis=1))).any(axis=(1,2)).sum()
                    - still_nan.sum())
    print(f"  Imputed missing steps for {n_imputed} examples  "
          f"(dropped {still_nan.sum()} with all steps missing)")

    X        = X[valid]
    y        = y[valid].astype(int)
    idx_used = [all_idx[i] for i in range(len(all_idx)) if valid[i]]

    X_delta = np.diff(X, axis=1)

    return X, X_delta, y, idx_used

def train_gru_fold(X_tr, y_tr, X_te, scaler, hidden_size, epochs,
                   lr, batch_size, pos_weight, device):
    N_tr, T, F = X_tr.shape


    X_tr_2d = scaler.transform(X_tr.reshape(N_tr * T, F))
    X_tr    = X_tr_2d.reshape(N_tr, T, F)

    N_te    = X_te.shape[0]
    X_te_2d = scaler.transform(X_te.reshape(N_te * T, F))
    X_te    = X_te_2d.reshape(N_te, T, F)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)

    ds     = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model     = GRUClassifier(input_size=F, hidden_size=hidden_size).to(device)
    pw        = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred    = model(xb)
            weights = torch.where(yb == 1, pw.expand_as(yb), torch.ones_like(yb))
            loss    = (criterion(pred, yb) * weights).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        probs = model(X_te_t).cpu().numpy()
    return probs


def run_cv(X, y, model_fn, n_splits=5):

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    aurocs, praucs, briers = [], [], []
    for tr, te in cv.split(X.reshape(len(X), -1), y):
        probs = model_fn(X[tr], y[tr], X[te])
        if len(np.unique(y[te])) < 2:
            continue
        aurocs.append(roc_auc_score(y[te], probs))
        praucs.append(average_precision_score(y[te], probs))
        briers.append(brier_score_loss(y[te], probs))
    return {
        "auroc_mean":  float(np.mean(aurocs)),
        "auroc_std":   float(np.std(aurocs)),
        "prauc_mean":  float(np.mean(praucs)),
        "prauc_std":   float(np.std(praucs)),
        "brier_mean":  float(np.mean(briers)),
        "brier_std":   float(np.std(briers)),
        "n_folds":     len(aurocs),
    }


def make_gru_fn(hidden_size, epochs, lr, batch_size, pos_weight, device):
    def fn(X_tr, y_tr, X_te):
        N_tr, T, F = X_tr.shape
        scaler = StandardScaler()
        scaler.fit(X_tr.reshape(N_tr * T, F))
        return train_gru_fold(X_tr, y_tr, X_te, scaler,
                               hidden_size, epochs, lr, batch_size,
                               pos_weight, device)
    return fn


def make_lr_fn(X_flat, class_weight="balanced"):

    def fn(X_tr_seq, y_tr, X_te_seq):

        N_tr = X_tr_seq.shape[0]
        N_te = X_te_seq.shape[0]

        Xf_tr = X_tr_seq.reshape(N_tr, -1)
        Xf_te = X_te_seq.reshape(N_te, -1)
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight=class_weight))
        ])
        pipe.fit(Xf_tr, y_tr)
        return pipe.predict_proba(Xf_te)[:, 1]
    return fn


def make_lr_single_step_fn(step_idx, class_weight="balanced"):

    def fn(X_tr_seq, y_tr, X_te_seq):
        Xf_tr = X_tr_seq[:, step_idx, :]
        Xf_te = X_te_seq[:, step_idx, :]
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight=class_weight))
        ])
        pipe.fit(Xf_tr, y_tr)
        return pipe.predict_proba(Xf_te)[:, 1]
    return fn


def print_result(label, r, baseline_prauc):
    print(
        f"  {label:<40} "
        f"AUROC={r['auroc_mean']:.4f}±{r['auroc_std']:.4f}  "
        f"PR-AUC={r['prauc_mean']:.4f}±{r['prauc_std']:.4f}  "
        f"Brier={r['brier_mean']:.4f}±{r['brier_std']:.4f}  "
        f"Lift={r['prauc_mean']-baseline_prauc:+.4f}"
    )



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t128",       required=True, help="T=128 results xlsx/csv")
    ap.add_argument("--t256",       required=True, help="T=256 results xlsx/csv")
    ap.add_argument("--t400",       required=True, help="T=400 results xlsx/csv")
    ap.add_argument("--t512",       required=True, help="T=512 results xlsx/csv")
    ap.add_argument("--idx_filter", default=None,  help="Shared clean idx CSV")
    ap.add_argument("--n_splits",   type=int,   default=5)
    ap.add_argument("--hidden_size",type=int,   default=32)
    ap.add_argument("--epochs",     type=int,   default=100)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int,   default=32)
    ap.add_argument("--seed",       type=int,   default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")


    paths = [args.t128, args.t256, args.t400, args.t512]
    print("Loading runs...")
    dfs = []
    for path, label in zip(paths, STEP_LABELS):
        df = load_and_clean(path)
        print(f"  {label}: {len(df)} clean rows  "
              f"(incorrect={df['label'].sum()}, "
              f"{100*df['label'].mean():.1f}%)")
        dfs.append(df)


    shared_idx = None
    if args.idx_filter:
        shared_idx = pd.read_csv(args.idx_filter)["idx"].tolist()
        print(f"Shared idx filter: {len(shared_idx)} entries")


    X, X_delta, y, idx_used = build_4step_sequences(dfs, shared_idx)
    N, T, F = X.shape
    pos_rate = y.mean()

    print(f"\nDataset: {N} examples  "
          f"(incorrect={y.sum()}, {100*pos_rate:.1f}%)")
    print(f"Sequence shape: {X.shape}  (N x steps x features)")
    print(f"Delta shape:    {X_delta.shape}  (N x transitions x features)")
    print(f"Steps: {STEP_LABELS}")
    print(f"Baseline PR-AUC (random): {pos_rate:.4f}")
    print(f"Baseline Brier:           {pos_rate*(1-pos_rate):.4f}")

    if len(np.unique(y)) < 2 or y.sum() < args.n_splits:
        print("ERROR: insufficient positive examples.")
        return

    pos_weight = float((y == 0).sum()) / max(float(y.sum()), 1)
    print(f"pos_weight (for GRU): {pos_weight:.2f}")

    print("\nRunning cross-validation...")

    #T=256 early only which is our best single window
    lr_t256 = run_cv(X, y, make_lr_single_step_fn(step_idx=1), args.n_splits)


    lr_flat_bal = run_cv(X, y, make_lr_fn(X, class_weight="balanced"), args.n_splits)

    X_delta_flat = X_delta.reshape(N, -1)
    def make_lr_delta_fn(cw="balanced"):
        def fn(X_tr_seq, y_tr, X_te_seq):

            tr_idx = [i for i in range(N) if any(
                np.array_equal(X_tr_seq[j], X[i]) for j in range(len(X_tr_seq))
            )]


            pipe = Pipeline([
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, class_weight=cw))
            ])


            tr_mask = np.array([
                np.any(np.all(X == x_row, axis=(1,2))) for x_row in X_tr_seq
            ])

            dtr = np.diff(X_tr_seq, axis=1).reshape(len(X_tr_seq), -1)
            dte = np.diff(X_te_seq, axis=1).reshape(len(X_te_seq), -1)
            pipe.fit(dtr, y_tr)
            return pipe.predict_proba(dte)[:, 1]
        return fn

    lr_delta = run_cv(X, y, make_lr_delta_fn("balanced"), args.n_splits)


    def make_lr_mag_delta_fn(cw="balanced"):
        def fn(X_tr_seq, y_tr, X_te_seq):
            mag_tr  = X_tr_seq.reshape(len(X_tr_seq), -1)
            mag_te  = X_te_seq.reshape(len(X_te_seq), -1)
            delt_tr = np.diff(X_tr_seq, axis=1).reshape(len(X_tr_seq), -1)
            delt_te = np.diff(X_te_seq, axis=1).reshape(len(X_te_seq), -1)
            Xc_tr   = np.concatenate([mag_tr, delt_tr], axis=1)
            Xc_te   = np.concatenate([mag_te, delt_te], axis=1)
            pipe = Pipeline([
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, class_weight=cw))
            ])
            pipe.fit(Xc_tr, y_tr)
            return pipe.predict_proba(Xc_te)[:, 1]
        return fn

    lr_mag_delta = run_cv(X, y, make_lr_mag_delta_fn("balanced"), args.n_splits)


    gru_fn  = make_gru_fn(args.hidden_size, args.epochs, args.lr,
                           args.batch_size, pos_weight, device)
    gru_res = run_cv(X, y, gru_fn, args.n_splits)


    print(f"\n{'='*95}")
    print(f"  4-STEP TRAJECTORY CLASSIFIER  (steps: {', '.join(STEP_LABELS)})")
    print(f"{'='*95}")
    print(f"  {'Model':<45} {'AUROC':>12} {'PR-AUC':>14} {'Brier':>12} {'Lift':>8}")
    print(f"  {'-'*91}")

    print_result("LR — T=256 only (best single window)",       lr_t256,      pos_rate)
    print_result("LR — 4 steps magnitude, flattened",          lr_flat_bal,  pos_rate)
    print_result("LR — delta features only (shape only)",      lr_delta,     pos_rate)
    print_result("LR — magnitude + delta (full picture)",      lr_mag_delta, pos_rate)
    print_result(f"GRU — 4-step raw sequence, hidden={args.hidden_size}", gru_res, pos_rate)

    print(f"  {'-'*91}")
    print(f"  {'Baseline (random)':<45}  PR-AUC={pos_rate:.4f}  "
          f"Brier={pos_rate*(1-pos_rate):.4f}")
    print(f"{'='*95}")


    delta_adds        = lr_mag_delta["prauc_mean"] - lr_flat_bal["prauc_mean"]
    delta_alone       = lr_delta["prauc_mean"]     - pos_rate
    gru_vs_flat       = gru_res["prauc_mean"]      - lr_flat_bal["prauc_mean"]
    gru_vs_mag_delta  = gru_res["prauc_mean"]      - lr_mag_delta["prauc_mean"]
    multiwindow_gain  = lr_flat_bal["prauc_mean"]  - lr_t256["prauc_mean"]

    print("\nINTERPRETATION")
    print("-" * 65)
    print(f"  Multi-window vs T=256 only (magnitude gain):   {multiwindow_gain:+.4f}")
    print(f"  Delta features added to magnitude (shape gain):{delta_adds:+.4f}")
    print(f"  Delta features alone vs random (shape signal): {delta_alone:+.4f}")
    print(f"  GRU vs LR-flat (ordering beyond magnitude):    {gru_vs_flat:+.4f}")
    print(f"  GRU vs LR-mag+delta (GRU beyond explicit):     {gru_vs_mag_delta:+.4f}")
    print()

    if delta_adds > 0.01:
        print("  TRAJECTORY SHAPE FINDING: Delta features add meaningful signal")
        print("  (+{:.4f} PR-AUC) — how uncertainty evolves matters beyond magnitude.".format(delta_adds))
        print("  Your intuition about rising entropy being predictive is supported.")
    elif abs(delta_adds) <= 0.01:
        print("  TRAJECTORY SHAPE FINDING: Delta features add negligible signal")
        print("  ({:+.4f} PR-AUC). Magnitude at each window is sufficient.".format(delta_adds))
        print("  Failure-predictive uncertainty is a sustained property, not a trend.")
    else:
        print("  TRAJECTORY SHAPE FINDING: Delta features HURT performance")
        print("  ({:+.4f} PR-AUC) — trajectory deltas add noise.".format(delta_adds))

    if gru_vs_flat > 0.01:
        print("\n  GRU FINDING: GRU outperforms LR-flat — implicit temporal ordering helps.")
    elif abs(gru_vs_flat) <= 0.01:
        print("\n  GRU FINDING: GRU ≈ LR-flat — temporal ordering adds no signal.")
        print("  Justifies logistic regression as primary classifier.")
    else:
        print(f"\n  GRU FINDING: GRU underperforms LR-flat ({gru_vs_flat:+.4f}).")
        print("  Likely overfitting on limited positive examples.")

    print("=" * 95)


if __name__ == "__main__":
    main()