"""Rank sessions with Isolation Forest, max|z|, residual rate, and Mahalanobis at the same k."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

MODEL_FEATURES = [
    "log_bytes",
    "log_duration",
    "failed_login_attempts",
    "hour_sin",
    "hour_cos",
]

RANK_METHODS = (
    "isolation_forest",
    "max_abs_z",
    "rate_residual",
    "mahalanobis",
)


def add_circular_hour(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "hour_of_day" not in out.columns:
        ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        if ts.isna().any():
            ts = pd.to_datetime(out["timestamp"], errors="coerce")
        out["hour_of_day"] = ts.dt.hour
    hour = out["hour_of_day"].to_numpy(dtype=float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return out


def design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    framed = add_circular_hour(df)
    bytes_ = framed["bytes_sent"].to_numpy(dtype=float)
    dur = framed["duration_seconds"].to_numpy(dtype=float)
    framed["log_bytes"] = np.log1p(np.clip(bytes_, 0, None))
    framed["log_duration"] = np.log1p(np.clip(dur, 0, None))
    framed["log_rate"] = framed["log_bytes"] - framed["log_duration"]
    missing = [c for c in MODEL_FEATURES if c not in framed.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return framed[MODEL_FEATURES].to_numpy(dtype=float), framed


def _fit_slice(df: pd.DataFrame, fit_on: str) -> pd.DataFrame:
    if fit_on == "all":
        return df
    if fit_on == "benign":
        if "is_attack" not in df.columns:
            raise ValueError("fit_on='benign' requires an is_attack column.")
        clean = df.loc[~df["is_attack"].astype(bool)]
        if clean.empty:
            raise ValueError("fit_on='benign' but the train batch has no benign rows.")
        return clean
    raise ValueError("fit_on must be 'all' or 'benign'.")


class SessionRanker:
    """Fit once (train), score any batch.

    Isolation Forest ranking ignores contamination.
    rate_residual is the first-principle control: |log(1+bytes)-log(1+t) - μ|.
    """

    def __init__(
        self,
        seed: int = 42,
        n_estimators: int = 300,
        cov: str = "empirical",
    ):
        self.seed = seed
        self.n_estimators = n_estimators
        self.cov_name = cov
        self.scaler = StandardScaler()
        self.forest = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",
            random_state=seed,
        )
        self.cov = LedoitWolf() if cov == "ledoit" else EmpiricalCovariance()
        self.mu_: np.ndarray | None = None
        self.sd_: np.ndarray | None = None
        self.rate_mu_: float | None = None
        self.rate_sd_: float | None = None
        self.fit_on_: str | None = None
        self.n_fit_: int = 0
        self.fitted_ = False

    def fit(self, df: pd.DataFrame, fit_on: str = "all") -> "SessionRanker":
        train = _fit_slice(df, fit_on)
        x, framed = design_matrix(train)
        xs = self.scaler.fit_transform(x)
        self.forest.fit(xs)
        self.cov.fit(xs)
        self.mu_ = np.median(x, axis=0)
        mad = 1.4826 * np.median(np.abs(x - self.mu_), axis=0)
        std = x.std(axis=0, ddof=1)
        self.sd_ = np.where(mad > 1e-6, mad, std)
        self.sd_ = np.clip(self.sd_, 1e-6, None)
        rate = framed["log_rate"].to_numpy(dtype=float)
        self.rate_mu_ = float(np.median(rate))
        rate_mad = 1.4826 * float(np.median(np.abs(rate - self.rate_mu_)))
        rate_std = float(np.std(rate, ddof=1)) if len(rate) > 1 else 1.0
        self.rate_sd_ = max(rate_mad if rate_mad > 1e-6 else rate_std, 1e-6)
        self.fit_on_ = fit_on
        self.n_fit_ = int(len(train))
        self.fitted_ = True
        return self

    def scores(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        if not self.fitted_:
            raise RuntimeError("Call fit() before scores().")
        x, framed = design_matrix(df)
        xs = self.scaler.transform(x)
        rate_z = np.abs((framed["log_rate"].to_numpy(dtype=float) - self.rate_mu_) / self.rate_sd_)
        raw = {
            "isolation_forest": -self.forest.decision_function(xs),
            "max_abs_z": np.max(np.abs((x - self.mu_) / self.sd_), axis=1),
            "rate_residual": rate_z,
            "mahalanobis": self.cov.mahalanobis(xs),
        }
        return framed, raw

    def score_table(self, df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        framed, raw = self.scores(df)
        k = min(top_k, len(framed))
        out = framed
        out["anomaly_score"] = raw["isolation_forest"]
        out["anomaly_rank"] = _rank_desc(raw["isolation_forest"])
        out["is_anomaly"] = out["anomaly_rank"] <= k
        out["maxz_score"] = raw["max_abs_z"]
        out["maxz_rank"] = _rank_desc(raw["max_abs_z"])
        out["is_maxz"] = out["maxz_rank"] <= k
        out["rate_score"] = raw["rate_residual"]
        out["rate_rank"] = _rank_desc(raw["rate_residual"])
        out["is_rate"] = out["rate_rank"] <= k
        out["mahal_score"] = raw["mahalanobis"]
        out["mahal_rank"] = _rank_desc(raw["mahalanobis"])
        out["is_mahal"] = out["mahal_rank"] <= k
        return out

    def save(self, path: Path) -> None:
        if not self.fitted_:
            raise RuntimeError("Nothing to save; call fit() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=4)

    @staticmethod
    def load(path: Path) -> "SessionRanker":
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, SessionRanker):
            raise TypeError(f"{path} is not a SessionRanker pickle.")
        return obj


def _rank_desc(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return series.rank(ascending=False, method="first").astype(int).to_numpy()


def detect_anomalies(
    df: pd.DataFrame,
    top_k: int = 30,
    seed: int = 42,
    fit_on: str = "auto",
) -> pd.DataFrame:
    """Transductive convenience: fit and rank the same batch."""
    if df.empty:
        raise ValueError("No sessions to analyze.")
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")
    resolved = _resolve_fit_on(df, fit_on)
    return SessionRanker(seed=seed).fit(df, fit_on=resolved).score_table(df, top_k=top_k)


def _resolve_fit_on(df: pd.DataFrame, fit_on: str) -> str:
    if fit_on == "auto":
        if "is_attack" in df.columns and (~df["is_attack"].astype(bool)).any():
            return "benign"
        return "all"
    return fit_on


def rule_baseline(df: pd.DataFrame) -> pd.Series:
    """Naive high thresholds. Not a ranker — unbounded alert volume."""
    hour = df["hour_of_day"] if "hour_of_day" in df.columns else pd.to_datetime(df["timestamp"]).dt.hour
    return (
        (df["failed_login_attempts"] >= 5)
        | (df["bytes_sent"] > 100000)
        | ((hour <= 5) & (df["duration_seconds"] > 400))
    )


def evaluate_binary(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "alerts": int(np.sum(y_pred)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def ranking_metrics(y_true, scores, top_k: int) -> dict:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    ranks = _rank_desc(s)
    pred = ranks <= min(top_k, len(s))
    out = evaluate_binary(y, pred)
    if y.any() and not y.all():
        auroc = float(roc_auc_score(y, s))
        ap = float(average_precision_score(y, s))
        att = s[y]
        nor = s[~y]
        inversions = int(np.sum(att[:, None] < nor[None, :]))
        pairs = int(att.size * nor.size)
        out["auroc"] = round(auroc, 6)
        out["ap"] = round(ap, 6)
        out["margin"] = round(float(att.min() - nor.max()), 4)
        out["inversions"] = inversions
        out["pair_count"] = pairs
        out["worst_attack_rank"] = int(ranks[y].max())
    else:
        out["auroc"] = None
        out["ap"] = None
        out["margin"] = None
        out["inversions"] = None
        out["pair_count"] = None
        out["worst_attack_rank"] = None
    return out


def recall_by_label(df: pd.DataFrame, pred_col: str = "is_anomaly") -> dict:
    if "label" not in df.columns:
        return {}
    out = {}
    for label, group in df.groupby("label"):
        if label == "normal":
            continue
        out[str(label)] = round(float(group[pred_col].mean()), 3)
    return out


def compare_rankers(df: pd.DataFrame, top_k: int) -> dict:
    if "is_attack" not in df.columns:
        return {}
    y = df["is_attack"]
    methods = {
        "isolation_forest": ("anomaly_score", "is_anomaly"),
        "max_abs_z": ("maxz_score", "is_maxz"),
        "rate_residual": ("rate_score", "is_rate"),
        "mahalanobis": ("mahal_score", "is_mahal"),
    }
    out = {}
    for name, (score_col, pred_col) in methods.items():
        if score_col not in df.columns:
            continue
        metrics = ranking_metrics(y, df[score_col], top_k)
        metrics["recall_by_attack_type"] = recall_by_label(df, pred_col)
        out[name] = metrics
    out["rule_baseline_unbounded"] = evaluate_binary(y, rule_baseline(df))
    return out
