\
import numpy as np

class PCAFromScratch:
    """
    PCA implemented from scratch using NumPy.

    Implementation notes:
    - We center X (zero-mean).
    - Use SVD of centered matrix for numerical stability:
        X_centered = U S V^T
      Principal directions are rows of V^T (columns of V).
      Eigenvalues of covariance are (S^2)/(n-1).
    """
    def __init__(self, n_components=None, variance_ratio=None, whiten=False):
        if n_components is None and variance_ratio is None:
            raise ValueError("Provide n_components or variance_ratio.")
        self.n_components = n_components
        self.variance_ratio = variance_ratio
        self.whiten = whiten

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        # explained variance
        eigenvalues = (S ** 2) / (n - 1)
        total_var = eigenvalues.sum()
        explained_ratio = eigenvalues / total_var
        cumulative = np.cumsum(explained_ratio)

        if self.variance_ratio is not None:
            k = int(np.argmax(cumulative >= self.variance_ratio) + 1)
            self.n_components_ = k
        else:
            self.n_components_ = int(self.n_components)

        self.components_ = Vt[: self.n_components_, :]           # shape (k, d)
        self.singular_values_ = S[: self.n_components_]
        self.eigenvalues_ = eigenvalues[: self.n_components_]
        self.explained_variance_ratio_ = explained_ratio[: self.n_components_]
        self.cumulative_variance_ratio_ = cumulative[: self.n_components_]
        self.total_variance_ = float(total_var)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Xc = X - self.mean_
        Z = Xc @ self.components_.T
        if self.whiten:
            # divide by sqrt(eigenvalues) to decorrelate and unit variance
            Z = Z / np.sqrt(self.eigenvalues_)
        return Z

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=float)
        if self.whiten:
            Z = Z * np.sqrt(self.eigenvalues_)
        X_rec = Z @ self.components_ + self.mean_
        return X_rec

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
