"""Rank sessions with Isolation Forest, max|z|, and Mahalanobis at the same k."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
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


def add_circular_hour(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "hour_of_day" not in out.columns:
        out["hour_of_day"] = pd.to_datetime(out["timestamp"]).dt.hour
    hour = out["hour_of_day"].to_numpy(dtype=float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return out


def design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    framed = add_circular_hour(df)
    framed["log_bytes"] = np.log1p(framed["bytes_sent"].to_numpy(dtype=float))
    framed["log_duration"] = np.log1p(framed["duration_seconds"].to_numpy(dtype=float))
    missing = [c for c in MODEL_FEATURES if c not in framed.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return framed[MODEL_FEATURES].to_numpy(dtype=float), framed


class SessionRanker:
    """Fit once (train), score any batch. Isolation Forest ranking ignores contamination."""

    def __init__(self, seed: int = 42, n_estimators: int = 300):
        self.seed = seed
        self.n_estimators = n_estimators
        self.scaler = StandardScaler()
        self.forest = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",
            random_state=seed,
        )
        self.cov = EmpiricalCovariance()
        self.mu_: np.ndarray | None = None
        self.sd_: np.ndarray | None = None
        self.fitted_ = False

    def fit(self, df: pd.DataFrame) -> "SessionRanker":
        x, _ = design_matrix(df)
        xs = self.scaler.fit_transform(x)
        self.forest.fit(xs)
        self.cov.fit(xs)
        # Robust 1D location/scale so max|z| is not pulled by the planted tails.
        self.mu_ = np.median(x, axis=0)
        mad = 1.4826 * np.median(np.abs(x - self.mu_), axis=0)
        std = x.std(axis=0, ddof=1)
        self.sd_ = np.where(mad > 1e-6, mad, std)
        self.sd_ = np.clip(self.sd_, 1e-6, None)
        self.fitted_ = True
        return self

    def scores(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        if not self.fitted_:
            raise RuntimeError("Call fit() before scores().")
        x, framed = design_matrix(df)
        xs = self.scaler.transform(x)
        raw = {
            "isolation_forest": -self.forest.decision_function(xs),
            "max_abs_z": np.max(np.abs((x - self.mu_) / self.sd_), axis=1),
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
        out["mahal_score"] = raw["mahalanobis"]
        out["mahal_rank"] = _rank_desc(raw["mahalanobis"])
        out["is_mahal"] = out["mahal_rank"] <= k
        return out


def _rank_desc(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return series.rank(ascending=False, method="first").astype(int).to_numpy()


def detect_anomalies(df: pd.DataFrame, top_k: int = 30, seed: int = 42) -> pd.DataFrame:
    """Transductive convenience: fit and rank the same batch."""
    if df.empty:
        raise ValueError("No sessions to analyze.")
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")
    return SessionRanker(seed=seed).fit(df).score_table(df, top_k=top_k)


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
        out["auroc"] = round(float(roc_auc_score(y, s)), 4)
        out["ap"] = round(float(average_precision_score(y, s)), 4)
        margin = float(s[y].min() - s[~y].max())
        out["margin"] = round(margin, 4)
        out["worst_attack_rank"] = int(ranks[y].max())
    else:
        out["auroc"] = None
        out["ap"] = None
        out["margin"] = None
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
        "mahalanobis": ("mahal_score", "is_mahal"),
    }
    out = {}
    for name, (score_col, pred_col) in methods.items():
        metrics = ranking_metrics(y, df[score_col], top_k)
        metrics["recall_by_attack_type"] = recall_by_label(df, pred_col)
        out[name] = metrics
    out["rule_baseline_unbounded"] = evaluate_binary(y, rule_baseline(df))
    return out
