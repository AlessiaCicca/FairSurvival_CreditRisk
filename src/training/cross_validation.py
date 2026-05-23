"""
src/training/cross_validation.py

GroupKFold cross-validation loop for M_STATIC, M_DYNAMIC, and M_PP.
Returns OOF predictions, per-fold metrics, and the last-fold model/scaler.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, precision_recall_curve

from src.training.train_mlp import train_mlp


# ── Utility ───────────────────────────────────────────────────────────────────

def find_best_threshold(y_true: np.ndarray, p: np.ndarray,
                        max_th_quantile: float = 0.90) -> float:
    """F1-optimal threshold computed on the training fold."""
    p = np.clip(p, 0, 1)
    prec, rec, thresholds = precision_recall_curve(y_true, p)
    max_th    = np.quantile(p, max_th_quantile)
    f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
    f1_scores[thresholds > max_th] = 0
    return float(thresholds[np.argmax(f1_scores)]) if len(thresholds) > 0 else 0.5


def metrics_all(y_true: np.ndarray, p: np.ndarray,
                threshold: float = 0.5) -> dict:
    """AUC, Brier score, F1 for a single fold."""
    p   = np.clip(p, 0, 1)
    auc = roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan
    return dict(
        AUC   = auc,
        Brier = brier_score_loss(y_true, p),
        F1    = f1_score(y_true, (p >= threshold).astype(int), zero_division=0),
        Th    = threshold,
    )


def agg_mean_sd(list_of_dicts: list) -> dict:
    """Aggregate a list of metric dicts into mean ± SD."""
    out = {}
    for k in list_of_dicts[0].keys():
        vals = [d[k] for d in list_of_dicts]
        out[f"{k}_Mean"] = float(np.nanmean(vals))
        out[f"{k}_SD"]   = float(np.nanstd(vals))
    return out


# ── Main CV function ──────────────────────────────────────────────────────────

def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sensitive: np.ndarray,
    time_arr: np.ndarray = None,
    subj_ids: np.ndarray = None,
    model_name: str = "",
    n_splits: int = 5,
    landmarks: list = None,
    **train_kwargs,
) -> dict:
    """
    Run GroupKFold CV for one model.

    Parameters
    ----------
    X, y        : features and labels
    groups      : subject IDs for GroupKFold splitting
    sensitive   : sensitive attribute array (parallel to X)
    time_arr    : landmark / loan_age array (parallel to X); None for static
    subj_ids    : subject IDs parallel to X (used by hazard penalty)
    model_name  : "static" | "dynamic" | "person_period"
    n_splits    : number of CV folds
    landmarks   : list of landmark values — if provided, computes per-landmark
                  metrics on the last fold (dynamic model only)
    **train_kwargs : forwarded verbatim to train_mlp
                     (beta, alpha, gamma, eo_mode_d, eo_mode_p, …)

    Returns
    -------
    dict with keys:
        oof_preds       : OOF predicted probabilities  (shape N,)
        metrics         : list of per-fold metric dicts
        times           : list of per-fold wall-clock seconds
        summary         : aggregated mean/SD metrics dict
        model_last      : trained MLP from last fold
        scaler_last     : fitted StandardScaler from last fold
        metrics_by_lmk  : dict {landmark: [metric_dicts]} (dynamic only)
    """
    gkf            = GroupKFold(n_splits=n_splits)
    oof_preds      = np.zeros(len(y), dtype=np.float64)
    metrics_list   = []
    times_list     = []
    model_last     = None
    scaler_last    = None
    metrics_by_lmk = {L: [] for L in (landmarks or [])}

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        t0 = time.perf_counter()

        p_te, p_tr, model, scaler = train_mlp(
            X[tr], y[tr], X[te], y[te],
            sensitive_tr = sensitive[tr] if sensitive is not None else None,
            time_tr      = time_arr[tr]  if time_arr  is not None else None,
            subj_ids_tr  = subj_ids[tr]  if subj_ids  is not None else None,
            model_name   = model_name,
            verbose      = (fold == 0),
            **train_kwargs,
        )

        oof_preds[te] = p_te
        best_th       = find_best_threshold(y[tr], p_tr)
        metrics_list.append(metrics_all(y[te].astype(int), p_te, threshold=best_th))
        times_list.append(time.perf_counter() - t0)

        print(f"  Fold {fold + 1} — AUC: {metrics_list[-1]['AUC']:.4f}  "
              f"th={best_th:.5f}")

        # last fold: save model + per-landmark metrics
        if fold == n_splits - 1:
            model_last  = model
            scaler_last = scaler

            if landmarks is not None and time_arr is not None:
                for L in landmarks:
                    mask = time_arr[te] == L
                    if mask.sum() > 10 and len(np.unique(y[te][mask])) > 1:
                        metrics_by_lmk[L].append(
                            metrics_all(
                                y[te][mask].astype(int),
                                p_te[mask],
                                threshold=best_th,
                            )
                        )

    summary = agg_mean_sd(metrics_list)
    summary["Model"]         = model_name.upper()
    summary["Time_Mean_sec"] = float(np.mean(times_list))
    summary["Time_SD_sec"]   = float(np.std(times_list))

    return dict(
        oof_preds      = oof_preds,
        metrics        = metrics_list,
        times          = times_list,
        summary        = summary,
        model_last     = model_last,
        scaler_last    = scaler_last,
        metrics_by_lmk = metrics_by_lmk,
    )


# ── Summary helpers ───────────────────────────────────────────────────────────

def build_summary_table(cv_results: dict) -> pd.DataFrame:
    """
    Build a summary DataFrame from a dict of {model_name: cv_result}.

    Parameters
    ----------
    cv_results : dict
        Keys are model names (e.g. "M_STATIC"), values are dicts
        returned by run_cv().

    Returns
    -------
    pd.DataFrame with one row per model.
    """
    rows = []
    for name, res in cv_results.items():
        row = res["summary"].copy()
        row["Model"] = name
        rows.append(row)

    cols = [
        "Model", "AUC_Mean", "AUC_SD",
        "Brier_Mean", "Brier_SD",
        "F1_Mean", "F1_SD",
        "Time_Mean_sec", "Time_SD_sec",
    ]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


def build_landmark_summary(cv_result: dict, landmarks: list) -> pd.DataFrame:
    """
    Build a per-landmark AUC summary from the last fold of a dynamic model.

    Parameters
    ----------
    cv_result : dict returned by run_cv()
    landmarks : list of landmark values

    Returns
    -------
    pd.DataFrame with columns [Landmark, AUC_Mean, Brier_Mean, F1_Mean]
    """
    rows = []
    for L in landmarks:
        fold_metrics = cv_result["metrics_by_lmk"].get(L, [])
        if fold_metrics:
            r = agg_mean_sd(fold_metrics)
            r["Landmark"] = L
            rows.append(r)

    if not rows:
        return pd.DataFrame(columns=["Landmark", "AUC_Mean", "Brier_Mean", "F1_Mean"])

    df = pd.DataFrame(rows)
    return df[["Landmark", "AUC_Mean", "Brier_Mean", "F1_Mean"]]
