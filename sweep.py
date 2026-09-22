"""Seed sweeps. A single 42→99 table is a screenshot, not a result."""

from __future__ import annotations

from statistics import median

from detector import SessionRanker, compare_rankers
from simulator import generate_sessions

DEFAULT_SEEDS = (1, 7, 13, 21, 42, 99, 123, 999)


def run_one(
    train_seed: int,
    test_seed: int,
    top_k: int = 30,
    n_records: int = 2000,
    n_per_scenario: int = 8,
    fit_on: str = "benign",
    manifold: str = "single",
    slow_hours: str = "mixed",
    cov: str = "empirical",
) -> dict:
    train = generate_sessions(
        n_records=n_records,
        n_per_scenario=n_per_scenario,
        seed=train_seed,
        manifold=manifold,
        slow_hours=slow_hours,
    )
    test = generate_sessions(
        n_records=n_records,
        n_per_scenario=n_per_scenario,
        seed=test_seed,
        manifold=manifold,
        slow_hours=slow_hours,
    )
    scored = (
        SessionRanker(seed=train_seed, cov=cov)
        .fit(train, fit_on=fit_on)
        .score_table(test, top_k=top_k)
    )
    cmp_ = compare_rankers(scored, top_k)
    row = {
        "train_seed": train_seed,
        "test_seed": test_seed,
        "fit_on": fit_on,
        "manifold": manifold,
        "slow_hours": slow_hours,
        "cov": cov,
        "k": top_k,
        "n": n_records,
        "planted": int(test["is_attack"].sum()),
    }
    for name in ("isolation_forest", "max_abs_z", "rate_residual", "mahalanobis"):
        m = cmp_[name]
        row[f"{name}_p"] = m["precision"]
        row[f"{name}_r"] = m["recall"]
        row[f"{name}_ap"] = m["ap"]
        row[f"{name}_auroc"] = m["auroc"]
        row[f"{name}_margin"] = m["margin"]
        row[f"{name}_inversions"] = m["inversions"]
        row[f"{name}_worst"] = m["worst_attack_rank"]
        row[f"{name}_by_type"] = m["recall_by_attack_type"]
    return row


def sweep(
    test_seeds: tuple[int, ...] = DEFAULT_SEEDS,
    train_seed: int = 42,
    **kwargs,
) -> list[dict]:
    return [run_one(train_seed, seed, **kwargs) for seed in test_seeds]


def summarize(rows: list[dict], method: str) -> dict:
    rec = [r[f"{method}_r"] for r in rows]
    worst = [r[f"{method}_worst"] for r in rows]
    auroc = [r[f"{method}_auroc"] for r in rows]
    inv = [r[f"{method}_inversions"] for r in rows]
    return {
        "n": len(rows),
        "recall_median": median(rec),
        "recall_min": min(rec),
        "worst_median": median(worst),
        "worst_max": max(worst),
        "auroc_median": median(auroc),
        "auroc_min": min(auroc),
        "inversions_median": median(inv),
        "inversions_max": max(inv),
    }


def format_summary(rows: list[dict]) -> str:
    lines = [
        f"sweep n={len(rows)} train_seed={rows[0]['train_seed']} "
        f"fit_on={rows[0]['fit_on']} manifold={rows[0]['manifold']} "
        f"slow_hours={rows[0]['slow_hours']}"
    ]
    header = f"{'method':18} {'Rmed':>6} {'Rmin':>6} {'AUCmed':>8} {'AUCmin':>8} {'wstMed':>7} {'wstMax':>7} {'invMed':>7}"
    lines.append(header)
    for method in ("isolation_forest", "max_abs_z", "rate_residual", "mahalanobis"):
        s = summarize(rows, method)
        lines.append(
            f"{method:18} {s['recall_median']:6.3f} {s['recall_min']:6.3f} "
            f"{s['auroc_median']:8.5f} {s['auroc_min']:8.5f} "
            f"{s['worst_median']:7.1f} {s['worst_max']:7d} {s['inversions_median']:7.1f}"
        )
    return "\n".join(lines)
