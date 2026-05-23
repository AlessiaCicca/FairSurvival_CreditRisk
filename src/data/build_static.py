"""
src/data/build_static.py

Builds the static dataset (t=0) from the raw longitudinal panel.
One row per subject, features from first observation only.
Target: default within HORIZON months from origination.
"""

import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def build_static(
    df: pd.DataFrame,
    static_cols: list,
    cat_cols: list,
    horizon: int,
    id_col: str = "ID",
    time_col: str = "Time",
    first_event_col: str = "FirstEventTime",
    sens_col: str = "sens_loan",
    enc_cat: OneHotEncoder = None,
) -> dict:
    """
    Build the static dataset from the raw panel.

    Parameters
    ----------
    df              : raw longitudinal panel (one row per subject × time)
    static_cols     : list of static numeric feature column names
    cat_cols        : list of categorical feature column names
    horizon         : prediction horizon in periods (e.g. 3 or 12)
    id_col          : subject ID column name
    time_col        : time column name
    first_event_col : column with the first event time per subject (or NaN)
    sens_col        : sensitive attribute column (already merged onto df)
    enc_cat         : fitted OneHotEncoder; if None, a new one is fit on this data

    Returns
    -------
    dict with keys:
        X           : feature matrix  (np.float32, shape N × p)
        y           : binary labels   (np.int8,    shape N)
        groups      : subject IDs     (shape N)
        sensitive   : sensitive attr  (shape N)
        enc_cat     : fitted OneHotEncoder (reuse for dynamic/PP)
        medians     : pd.Series of column medians (for imputation at inference)
        feature_names : list of feature names matching X columns
    """

    # ── Take first observation per subject ────────────────────────────────────
    static_df = (
        df.sort_values(time_col)
          .groupby(id_col)
          .first()
          .reset_index()
    )

    # ── Target: default within horizon ───────────────────────────────────────
    static_df["target_static"] = (
        static_df[first_event_col].notna() &
        (static_df[first_event_col] <= horizon)
    ).astype(np.int8)

    # ── Categorical encoding ──────────────────────────────────────────────────
    if enc_cat is None:
        enc_cat = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )
        enc_cat.fit(static_df[cat_cols])

    cats = enc_cat.transform(static_df[cat_cols])
    cat_feature_names = list(enc_cat.get_feature_names_out(cat_cols))

    # ── Numeric imputation ────────────────────────────────────────────────────
    medians = static_df[static_cols].median()
    num     = static_df[static_cols].fillna(medians).to_numpy(dtype=np.float32)

    # ── Assemble X ────────────────────────────────────────────────────────────
    X = np.hstack([num, cats])

    y         = static_df["target_static"].to_numpy(dtype=np.int8)
    groups    = static_df[id_col].to_numpy()
    sensitive = static_df[sens_col].to_numpy()

    feature_names = static_cols + cat_feature_names

    n_pos = y.sum()
    print(f"  [static] Rows: {len(X):,} | Positives: {n_pos} ({y.mean():.2%})")
    print(f"  [static] X shape: {X.shape}  NaN={np.isnan(X).sum()}  Inf={np.isinf(X).sum()}")

    del cats
    gc.collect()

    return dict(
        X             = X,
        y             = y,
        groups        = groups,
        sensitive     = sensitive,
        enc_cat       = enc_cat,
        medians       = medians,
        feature_names = feature_names,
    )
