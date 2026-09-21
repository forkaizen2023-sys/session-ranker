"""CLI: rank unusual sessions and measure the ranking against planted labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from detector import (
    MODEL_FEATURES,
    SessionRanker,
    compare_rankers,
    detect_anomalies,
)
from simulator import SCENARIO_HINTS, generate_sessions

REQUIRED_COLUMNS = ["timestamp", "bytes_sent", "duration_seconds", "failed_login_attempts"]


def load_sessions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
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
            "Rank the k sessions an analyst can review. Compare Isolation Forest "
            "with max|z| and Mahalanobis at the same k, on a holdout batch when possible."
        ),
        "protocol": protocol,
        "n_records": int(len(df)),
        "analyst_budget_k": top_k,
        "features": MODEL_FEATURES,
        "has_ground_truth": has_labels,
        "comparison_at_k": _json_safe(comparison),
        "mitre_hints": SCENARIO_HINTS if has_labels else None,
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
        print(f"{'method':20} {'P@k':>7} {'R@k':>7} {'F1':>7} {'AP':>8} {'AUROC':>8} {'margin':>9} {'worst':>7}")
        for name in ("isolation_forest", "max_abs_z", "mahalanobis"):
            m = cmp_[name]
            print(
                f"{name:20} {m['precision']:7.3f} {m['recall']:7.3f} {m['f1']:7.3f} "
                f"{(m['ap'] or 0):8.3f} {(m['auroc'] or 0):8.3f} "
                f"{(m['margin'] if m['margin'] is not None else 0):9.4f} "
                f"{m['worst_attack_rank'] or 0:7d}"
            )
        rules = cmp_["rule_baseline_unbounded"]
        print(
            f"\nUnbounded rules: alerts={rules['alerts']} P={rules['precision']:.3f} "
            f"R={rules['recall']:.3f} (not a ranker; volume is not k)"
        )
        print("\nRecall by planted type @ k")
        print(f"{'type':16} {'IF':>8} {'max|z|':>8} {'Mahal':>8}")
        labels = sorted(cmp_["isolation_forest"]["recall_by_attack_type"].keys())
        for label in labels:
            print(
                f"{label:16} "
                f"{cmp_['isolation_forest']['recall_by_attack_type'].get(label, 0):8.3f} "
                f"{cmp_['max_abs_z']['recall_by_attack_type'].get(label, 0):8.3f} "
                f"{cmp_['mahalanobis']['recall_by_attack_type'].get(label, 0):8.3f}"
            )
        print(
            "\nHow to read: AP/AUROC score the full ranking. P@k/R@k are the "
            "analyst budget. margin < 0 means attack and normal scores overlap. "
            "rate_exfil / low_and_slow are the joint cases; brute_force is the 1D control."
        )
    else:
        print("\nNo is_attack column — ops mode. Ranking only.")

    print(f"\nTop {report['analyst_budget_k']} by Mahalanobis (geometry-matched ranker):")
    rows = report["top_alerts"]
    if not rows:
        print("  (none)")
        return
    preview = pd.DataFrame(rows)
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(preview.head(12).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank unusual sessions. Default lab: fit on seed 42, score seed 99."
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
    parser.add_argument("--json", type=Path)
    parser.add_argument("--save-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranker = SessionRanker(seed=args.train_seed)

    if args.input and args.transductive:
        print(f"[*] Transductive: fit+score {args.input}")
        df = load_sessions(args.input)
        scored = detect_anomalies(df, top_k=args.top_k, seed=args.train_seed)
        protocol = "transductive"
    elif args.input:
        train_src = args.fit_csv
        if train_src:
            print(f"[*] Inductive: fit {train_src}, score {args.input}")
            train = load_sessions(train_src)
        else:
            print(f"[*] Inductive: fit generated seed={args.train_seed}, score {args.input}")
            train = generate_sessions(
                n_records=args.n_records,
                n_per_scenario=args.n_per_scenario,
                seed=args.train_seed,
            )
        ranker.fit(train)
        df = load_sessions(args.input)
        scored = ranker.score_table(df, top_k=args.top_k)
        protocol = "inductive"
    elif args.transductive:
        print(f"[*] Transductive lab seed={args.test_seed}")
        df = generate_sessions(
            n_records=args.n_records,
            n_per_scenario=args.n_per_scenario,
            seed=args.test_seed,
        )
        scored = detect_anomalies(df, top_k=args.top_k, seed=args.train_seed)
        protocol = "transductive"
        print(f"    {len(df)} rows, {int(df['is_attack'].sum())} planted")
    else:
        print(
            f"[*] Inductive lab: fit seed={args.train_seed}, "
            f"score seed={args.test_seed}, k={args.top_k}"
        )
        train = generate_sessions(
            n_records=args.n_records,
            n_per_scenario=args.n_per_scenario,
            seed=args.train_seed,
        )
        test = generate_sessions(
            n_records=args.n_records,
            n_per_scenario=args.n_per_scenario,
            seed=args.test_seed,
        )
        print(f"    train {len(train)} / test {len(test)}, planted in test={int(test['is_attack'].sum())}")
        ranker.fit(train)
        scored = ranker.score_table(test, top_k=args.top_k)
        protocol = "inductive"
        df = test

    report = build_report(scored, top_k=args.top_k, protocol=protocol)
    print_report(report)

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
