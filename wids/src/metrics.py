\
import numpy as np

def reconstruction_mse(X: np.ndarray, X_rec: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    X_rec = np.asarray(X_rec, dtype=float)
    return float(np.mean((X - X_rec) ** 2))

def reconstruction_rmse(X: np.ndarray, X_rec: np.ndarray) -> float:
    return float(np.sqrt(reconstruction_mse(X, X_rec)))

def variance_retained(explained_variance_ratio) -> float:
    return float(np.sum(explained_variance_ratio))
