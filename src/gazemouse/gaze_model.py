"""Per-user mapping from eye features to a screen point (physical pixels)."""
import numpy as np

# iris_x, iris_y_corner, iris_y_lid, openness. Head columns are left out: in the
# 2026 spike, raw head pose as regression inputs made accuracy worse.
USED = (0, 1, 2, 3)


class GazeModel:
    """Ridge regression on standardised eye features (best model in the spike)."""

    def fit(self, feats: np.ndarray, targets: np.ndarray, lam: float = 1e-3) -> None:
        feats, targets = np.asarray(feats, float), np.asarray(targets, float)
        if len(feats) < 10:
            raise ValueError(f"need at least 10 samples to fit, got {len(feats)}")
        X = feats[:, USED]
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-9
        A = self._design(X)
        reg = lam * np.eye(A.shape[1])
        reg[0, 0] = 0.0  # never shrink the intercept
        self.weights = np.linalg.solve(A.T @ A + reg, A.T @ targets)

    def predict(self, feat: np.ndarray) -> np.ndarray:
        x = np.asarray(feat, float)[list(USED)][None, :]
        return (self._design(x) @ self.weights)[0]

    def _design(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean) / self.std
        return np.column_stack([np.ones(len(Z)), Z])

    def to_dict(self) -> dict:
        return {"used": list(USED), "mean": self.mean.tolist(), "std": self.std.tolist(),
                "weights": self.weights.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "GazeModel":
        if tuple(d["used"]) != USED:
            raise ValueError(f"profile was made with features {d['used']}, expected {list(USED)}")
        m = cls()
        m.mean, m.std, m.weights = (np.array(d[k], float) for k in ("mean", "std", "weights"))
        return m
