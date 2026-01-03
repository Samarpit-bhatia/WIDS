\
import numpy as np
import pandas as pd

def basic_audit(df: pd.DataFrame) -> dict:
    """Lightweight audit used for logging/reporting."""
    audit = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_duplicates": int(df.duplicated().sum()),
        "missing_by_col": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    # Constant columns often appear as "Unnamed: 32" in some dataset copies
    nunique = df.nunique(dropna=False)
    audit["constant_cols"] = nunique[nunique <= 1].index.tolist()
    return audit

def drop_known_junk_columns(df: pd.DataFrame) -> pd.DataFrame:
    junk = [c for c in df.columns if c.lower().startswith("unnamed")]
    # Kaggle version often has id column and an "Unnamed: 32"
    for c in ["id", "ID", "Id", "unnamed: 32", "Unnamed: 32"]:
        if c in df.columns and c not in junk:
            junk.append(c)
    return df.drop(columns=junk, errors="ignore")

def encode_target(df: pd.DataFrame, target_col: str = "diagnosis") -> pd.DataFrame:
    """
    Encodes common {M,B} labels to {1,0}. Keeps any other labels as-is if present.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    if df[target_col].dtype == object:
        mapping = {"M": 1, "B": 0, "malignant": 1, "benign": 0}
        df[target_col] = df[target_col].map(lambda v: mapping.get(v, v))
    return df

def split_features_target(df: pd.DataFrame, target_col: str = "diagnosis"):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y

def impute_missing(X: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Imputes missing values.
    - median: robust to outliers
    - mean: standard
    - zero: sometimes used for sparse-like features
    """
    X = X.copy()
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in X.columns if c not in num_cols]

    if strategy == "median":
        fill = X[num_cols].median()
    elif strategy == "mean":
        fill = X[num_cols].mean()
    elif strategy == "zero":
        fill = pd.Series(0.0, index=num_cols)
    else:
        raise ValueError("strategy must be one of: median, mean, zero")

    X[num_cols] = X[num_cols].fillna(fill)

    # categorical: fill missing with explicit category
    for c in cat_cols:
        X[c] = X[c].fillna("missing")
    return X

def one_hot_encode(X: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode any non-numeric columns (safe even if none exist).
    """
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) == 0:
        return X
    return pd.get_dummies(X, columns=list(cat_cols), drop_first=True)

def outlier_mask_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)

def outlier_mask_robust_z(series: pd.Series, z_thresh: float = 3.5) -> pd.Series:
    """
    Robust z-score using MAD.
    z = 0.6745 * (x - median) / MAD
    """
    x = series.to_numpy()
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return pd.Series(False, index=series.index)
    z = 0.6745 * (x - med) / mad
    return pd.Series(np.abs(z) > z_thresh, index=series.index)

def winsorize_outliers(X: pd.DataFrame, method: str = "iqr", strength: float = 1.5) -> (pd.DataFrame, dict):
    """
    Handles outliers by winsorization (clipping) to a computed range.
    Returns (X_new, stats).
    """
    X = X.copy()
    num_cols = X.select_dtypes(include=[np.number]).columns
    stats = {"method": method, "n_outliers_by_col": {}}

    for c in num_cols:
        s = X[c]
        if method == "iqr":
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - strength * iqr
            upper = q3 + strength * iqr
            mask = outlier_mask_iqr(s, k=strength)
        elif method == "robust_z":
            # for robust_z, define clipping bounds via percentiles (stable)
            lower = s.quantile(0.01)
            upper = s.quantile(0.99)
            mask = outlier_mask_robust_z(s, z_thresh=strength)
        else:
            raise ValueError("method must be one of: iqr, robust_z")

        stats["n_outliers_by_col"][c] = int(mask.sum())
        X[c] = s.clip(lower, upper)

    return X, stats

def standard_scale(X: pd.DataFrame) -> (pd.DataFrame, pd.Series, pd.Series):
    """
    Standard scaling: (x - mean) / std
    """
    Xn = X.copy()
    num_cols = Xn.select_dtypes(include=[np.number]).columns
    mean = Xn[num_cols].mean()
    std = Xn[num_cols].std(ddof=0).replace(0, 1.0)
    Xn[num_cols] = (Xn[num_cols] - mean) / std
    return Xn, mean, std

def robust_scale(X: pd.DataFrame) -> (pd.DataFrame, pd.Series, pd.Series):
    """
    Robust scaling: (x - median) / IQR
    """
    Xn = X.copy()
    num_cols = Xn.select_dtypes(include=[np.number]).columns
    med = Xn[num_cols].median()
    q1 = Xn[num_cols].quantile(0.25)
    q3 = Xn[num_cols].quantile(0.75)
    iqr = (q3 - q1).replace(0, 1.0)
    Xn[num_cols] = (Xn[num_cols] - med) / iqr
    return Xn, med, iqr
