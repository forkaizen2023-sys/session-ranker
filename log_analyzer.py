"""CLI: rank unusual sessions and measure the ranking against planted labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from detector import (
    MODEL_FEATURES,
    RANK_METHODS,
    SessionRanker,
    compare_rankers,
    detect_anomalies,
)
from simulator import SCENARIO_HINTS, generate_sessions
from sweep import format_summary, sweep

REQUIRED_COLUMNS = ["timestamp", "bytes_sent", "duration_seconds", "failed_login_attempts"]


def load_sessions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raw = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    bad_ts = int(df["timestamp"].isna().sum())
    if bad_ts:
        raise ValueError(f"{path} has {bad_ts} unparseable timestamps.")
    for col in ("bytes_sent", "duration_seconds", "failed_login_attempts"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = df[REQUIRED_COLUMNS[1:]].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"{path} has {int(bad.sum())} rows with non-numeric required fields.")
    if (df["bytes_sent"] < 0).any() or (df["duration_seconds"] < 0).any():
        raise ValueError(f"{path} has negative bytes_sent or duration_seconds.")
    if "hour_of_day" not in df.columns:
        df["hour_of_day"] = df["timestamp"].dt.hour
    if "is_attack" in df.columns:
        df["is_attack"] = df["is_attack"].astype(bool)
    return df


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def build_report(df: pd.DataFrame, top_k: int, protocol: str = "transductive") -> dict:
    has_labels = "is_attack" in df.columns
    comparison = compare_rankers(df, top_k) if has_labels else {}
    rank_col = "mahal_rank" if "mahal_rank" in df.columns else "anomaly_rank"
    top = df.nsmallest(top_k, rank_col)
    alert_cols = [
        c
        for c in [
            "anomaly_rank",
            "anomaly_score",
            "maxz_rank",
            "rate_rank",
            "mahal_rank",
            "timestamp",
            "src_ip",
            "bytes_sent",
            "duration_seconds",
            "failed_login_attempts",
            "hour_of_day",
            "label",
        ]
        if c in top.columns
    ]
    alerts = json.loads(top[alert_cols].to_json(orient="records", date_format="iso"))
    return {
        "problem": (
            "Rank the k sessions an analyst can review. Compare Isolation Forest, "
            "max|z|, the rate residual, and Mahalanobis at the same k."
        ),
        "protocol": protocol,
        "n_records": int(len(df)),
        "analyst_budget_k": top_k,
        "features": MODEL_FEATURES,
        "has_ground_truth": has_labels,
        "comparison_at_k": _json_safe(comparison),
        "scenario_hints": SCENARIO_HINTS if has_labels else None,
        "top_alerts": alerts,
    }


def print_report(report: dict) -> None:
    print("\n=== Session ranker ===")
    print(f"Protocol: {report['protocol']}")
    print(f"Records: {report['n_records']}")
    print(f"Analyst budget k: {report['analyst_budget_k']}")
    print(f"Features: {', '.join(report['features'])}")

    cmp_ = report.get("comparison_at_k") or {}
    if report["has_ground_truth"] and cmp_:
        print("\nSame-k comparison (higher score = more anomalous)")
        print(
            f"{'method':20} {'P@k':>7} {'R@k':>7} {'F1':>7} {'AP':>8} "
            f"{'AUROC':>10} {'margin':>9} {'inv':>6} {'worst':>7}"
        )
        for name in RANK_METHODS:
            if name not in cmp_:
                continue
            m = cmp_[name]
            print(
                f"{name:20} {m['precision']:7.3f} {m['recall']:7.3f} {m['f1']:7.3f} "
                f"{(m['ap'] or 0):8.3f} {(m['auroc'] or 0):10.6f} "
                f"{(m['margin'] if m['margin'] is not None else 0):9.4f} "
                f"{(m['inversions'] or 0):6d} "
                f"{m['worst_attack_rank'] or 0:7d}"
            )
        rules = cmp_["rule_baseline_unbounded"]
        print(
            f"\nUnbounded rules: alerts={rules['alerts']} P={rules['precision']:.3f} "
            f"R={rules['recall']:.3f} (not a ranker; volume is not k)"
        )
        print("\nRecall by planted type @ k")
        print(f"{'type':16} {'IF':>8} {'max|z|':>8} {'resid':>8} {'Mahal':>8}")
        labels = sorted(cmp_["isolation_forest"]["recall_by_attack_type"].keys())
        for label in labels:
            print(
                f"{label:16} "
                f"{cmp_['isolation_forest']['recall_by_attack_type'].get(label, 0):8.3f} "
                f"{cmp_['max_abs_z']['recall_by_attack_type'].get(label, 0):8.3f} "
                f"{cmp_.get('rate_residual', {}).get('recall_by_attack_type', {}).get(label, 0):8.3f} "
                f"{cmp_['mahalanobis']['recall_by_attack_type'].get(label, 0):8.3f}"
            )
        print(
            "\nHow to read: AP/AUROC score the full ranking. P@k/R@k are the "
            "analyst budget. margin < 0 means attack and normal scores overlap. "
            "inversions = pairs where a normal scored above an attack. "
            "rate_residual is |log(1+bytes)-log(1+t)-μ| — the physics control. "
            "A single seed is a screenshot; run --sweep."
        )
    else:
        print("\nNo is_attack column — ops mode. Ranking only.")

    print(f"\nTop {report['analyst_budget_k']} by Mahalanobis:")
    rows = report["top_alerts"]
    if not rows:
        print("  (none)")
        return
    preview = pd.DataFrame(rows)
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(preview.head(12).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank unusual sessions. Default lab: fit benign seed 42, score seed 99."
    )
    parser.add_argument("--input", type=Path, help="CSV of sessions to score.")
    parser.add_argument(
        "--fit-csv",
        type=Path,
        help="CSV to fit on (inductive). Default lab uses a generated train batch.",
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--n-records", type=int, default=2000)
    parser.add_argument("--n-per-scenario", type=int, default=8)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--test-seed", type=int, default=99)
    parser.add_argument(
        "--transductive",
        action="store_true",
        help="Fit and score the same batch (no holdout).",
    )
    parser.add_argument(
        "--fit-on",
        choices=("auto", "benign", "all"),
        default="auto",
        help="auto: benign if is_attack exists, else all. Unsupervised claim uses benign.",
    )
    parser.add_argument(
        "--manifold",
        choices=("single", "bimodal"),
        default="single",
        help="bimodal = two legitimate rates; breaks a single Σ.",
    )
    parser.add_argument(
        "--slow-hours",
        choices=("mixed", "night"),
        default="mixed",
        help="mixed removes the night-only crutch from low_and_slow.",
    )
    parser.add_argument("--cov", choices=("empirical", "ledoit"), default="empirical")
    parser.add_argument("--save-model", type=Path)
    parser.add_argument("--load-model", type=Path)
    parser.add_argument("--sweep", action="store_true", help="8-seed holdout sweep, then exit.")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--save-csv", type=Path)
    return parser.parse_args()


def _resolve(train: pd.DataFrame, fit_on: str) -> str:
    if fit_on != "auto":
        return fit_on
    if "is_attack" in train.columns and (~train["is_attack"].astype(bool)).any():
        return "benign"
    return "all"


def main() -> None:
    args = parse_args()

    if args.sweep:
        rows = sweep(
            train_seed=args.train_seed,
            top_k=args.top_k,
            n_records=args.n_records,
            n_per_scenario=args.n_per_scenario,
            fit_on="benign" if args.fit_on == "auto" else args.fit_on,
            manifold=args.manifold,
            slow_hours=args.slow_hours,
            cov=args.cov,
        )
        print(format_summary(rows))
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"[*] Sweep written to {args.json}")
        return

    if args.load_model and not args.input:
        raise SystemExit("--load-model requires --input")

    if args.load_model:
        ranker = SessionRanker.load(args.load_model)
        print(f"[*] Loaded model {args.load_model} (fit_on={ranker.fit_on_}, n={ranker.n_fit_})")
        df = load_sessions(args.input)
        scored = ranker.score_table(df, top_k=args.top_k)
        protocol = f"inductive:loaded:{ranker.fit_on_}"
    else:
        ranker = SessionRanker(seed=args.train_seed, cov=args.cov)
        gen_kw = dict(
            n_records=args.n_records,
            n_per_scenario=args.n_per_scenario,
            manifold=args.manifold,
            slow_hours=args.slow_hours,
        )
        if args.input and args.transductive:
            print(f"[*] Transductive: fit+score {args.input}")
            df = load_sessions(args.input)
            fit_on = _resolve(df, args.fit_on)
            scored = detect_anomalies(df, top_k=args.top_k, seed=args.train_seed, fit_on=fit_on)
            protocol = f"transductive:{fit_on}"
        elif args.input:
            train_src = args.fit_csv
            if train_src:
                print(f"[*] Inductive: fit {train_src}, score {args.input}")
                train = load_sessions(train_src)
            else:
                print(f"[*] Inductive: fit generated seed={args.train_seed}, score {args.input}")
                train = generate_sessions(seed=args.train_seed, **gen_kw)
            fit_on = _resolve(train, args.fit_on)
            ranker.fit(train, fit_on=fit_on)
            df = load_sessions(args.input)
            scored = ranker.score_table(df, top_k=args.top_k)
            protocol = f"inductive:{fit_on}"
        elif args.transductive:
            print(f"[*] Transductive lab seed={args.test_seed}")
            df = generate_sessions(seed=args.test_seed, **gen_kw)
            fit_on = _resolve(df, args.fit_on)
            scored = detect_anomalies(df, top_k=args.top_k, seed=args.train_seed, fit_on=fit_on)
            protocol = f"transductive:{fit_on}"
            print(f"    {len(df)} rows, {int(df['is_attack'].sum())} planted, fit_on={fit_on}")
        else:
            print(
                f"[*] Inductive lab: fit seed={args.train_seed}, "
                f"score seed={args.test_seed}, k={args.top_k}"
            )
            train = generate_sessions(seed=args.train_seed, **gen_kw)
            test = generate_sessions(seed=args.test_seed, **gen_kw)
            fit_on = _resolve(train, args.fit_on)
            print(
                f"    train {len(train)} / test {len(test)}, "
                f"planted in test={int(test['is_attack'].sum())}, fit_on={fit_on}"
            )
            ranker.fit(train, fit_on=fit_on)
            scored = ranker.score_table(test, top_k=args.top_k)
            protocol = f"inductive:{fit_on}"
            df = test

    report = build_report(scored, top_k=args.top_k, protocol=protocol)
    print_report(report)

    if args.save_model:
        if args.load_model:
            SessionRanker.load(args.load_model).save(args.save_model)
        else:
            ranker.save(args.save_model)
        print(f"[*] Model written to {args.save_model}")

    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.save_csv, index=False)
        print(f"\n[*] Scored table written to {args.save_csv}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[*] Report written to {args.json}")


if __name__ == "__main__":
    main()
