\
import argparse
import json
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .preprocessing import (
    basic_audit,
    drop_known_junk_columns,
    encode_target,
    split_features_target,
    impute_missing,
    one_hot_encode,
    winsorize_outliers,
    standard_scale,
    robust_scale,
)
from .pca_from_scratch import PCAFromScratch
from .metrics import reconstruction_rmse, variance_retained

def save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", default="diagnosis")
    parser.add_argument("--impute", default="median", choices=["median", "mean", "zero"])
    parser.add_argument("--outliers", default="iqr", choices=["iqr", "robust_z"])
    parser.add_argument("--outlier_strength", type=float, default=1.5, help="IQR k or robust z threshold")
    parser.add_argument("--scaler", default="standard", choices=["standard", "robust"])
    parser.add_argument("--variance", type=float, default=0.95, help="Target variance retained for PCA")
    parser.add_argument("--whiten", action="store_true")
    parser.add_argument("--out_dir", default="reports")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.input)
    audit_raw = basic_audit(df_raw)

    df = drop_known_junk_columns(df_raw)
    df = encode_target(df, target_col=args.target)

    # Duplicates
    n_dup = int(df.duplicated().sum())
    if n_dup > 0:
        df = df.drop_duplicates()

    X, y = split_features_target(df, target_col=args.target)

    # Missing values
    X = impute_missing(X, strategy=args.impute)

    # Encoding (in case dataset includes any categorical features)
    X = one_hot_encode(X)

    # Outliers (winsorization)
    X, outlier_stats = winsorize_outliers(X, method=args.outliers, strength=args.outlier_strength)

    # Scaling
    if args.scaler == "standard":
        Xs, loc, scale = standard_scale(X)
        scaler_params = {"type": "standard", "mean": loc.to_dict(), "std": scale.to_dict()}
    else:
        Xs, loc, scale = robust_scale(X)
        scaler_params = {"type": "robust", "median": loc.to_dict(), "iqr": scale.to_dict()}

    # PCA
    pca = PCAFromScratch(variance_ratio=args.variance, whiten=args.whiten)
    Z = pca.fit_transform(Xs.to_numpy())
    X_rec = pca.inverse_transform(Z)

    # Metrics
    rmse = reconstruction_rmse(Xs.to_numpy(), X_rec)
    retained = variance_retained(pca.explained_variance_ratio_)

    # Save artifacts
    cleaned = pd.concat([y.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    cleaned.to_csv(out_dir / "cleaned.csv", index=False)

    embed = pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(Z.shape[1])])
    embed[args.target] = y.values
    embed.to_csv(out_dir / "pca_embedding.csv", index=False)

    # Plots
    # 1) Missingness bar plot (raw)
    miss = pd.Series(audit_raw["missing_by_col"]).sort_values(ascending=False)
    plt.figure(figsize=(10, 4))
    miss[miss > 0].plot(kind="bar")
    plt.title("Missing Values per Column (raw)")
    plt.ylabel("count")
    save_fig(fig_dir / "missing_values.png")

    # 2) Correlation heatmap on a subset (first 15 cols) to keep readable
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    subset = num_cols[:15]
    if len(subset) >= 2:
        plt.figure(figsize=(9, 7))
        sns.heatmap(X[subset].corr(), annot=False)
        plt.title("Correlation Heatmap (subset of features)")
        save_fig(fig_dir / "correlation_heatmap_subset.png")

    # 3) Explained variance curve
    cum = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(cum) + 1), cum, marker="o")
    plt.axhline(args.variance, linestyle="--")
    plt.title("Cumulative Explained Variance (PCA)")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative variance")
    save_fig(fig_dir / "pca_cumulative_variance.png")

    # 4) 2D scatter of first 2 PCs (if available)
    if Z.shape[1] >= 2:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=y, palette="coolwarm", s=35)
        plt.title("PCA Projection (PC1 vs PC2)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        save_fig(fig_dir / "pca_scatter_pc1_pc2.png")

    # Summary JSON
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input": args.input,
        "target": args.target,
        "audit_raw": audit_raw,
        "n_rows_after_dedup": int(df.shape[0]),
        "preprocess": {
            "impute": args.impute,
            "outliers": outlier_stats,
            "scaler": scaler_params["type"],
        },
        "pca": {
            "variance_target": args.variance,
            "n_components": int(pca.n_components_),
            "variance_retained": float(retained),
            "whiten": bool(args.whiten),
        },
        "reconstruction": {"rmse": float(rmse)},
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("Done.")
    print(f"Components selected: {pca.n_components_}")
    print(f"Variance retained: {retained:.4f}")
    print(f"Reconstruction RMSE (scaled space): {rmse:.6f}")
    print(f"Saved outputs to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
