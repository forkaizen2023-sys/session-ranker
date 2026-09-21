from datetime import datetime

import pandas as pd
import pytest

from detector import (
    MODEL_FEATURES,
    SessionRanker,
    compare_rankers,
    detect_anomalies,
    evaluate_binary,
)
from log_analyzer import build_report, load_sessions
from simulator import ATTACK_LABELS, BUSINESS_HOURS, generate_sessions


def test_timestamps_are_clock_hours_on_one_lab_day():
    df = generate_sessions(n_records=400, n_per_scenario=4, seed=42)
    assert df["timestamp"].dt.date.nunique() == 1
    assert df["hour_of_day"].between(0, 23).all()
    assert set(df.loc[df["label"] == "normal", "hour_of_day"]) & set(BUSINESS_HOURS)


def test_attack_families_are_not_stacked_on_one_row():
    df = generate_sessions(n_records=600, n_per_scenario=6, seed=42)
    brute = df[df["label"] == "brute_force"]
    exfil = df[df["label"] == "rate_exfil"]
    slow = df[df["label"] == "low_and_slow"]

    assert (brute["failed_login_attempts"] >= 9).all()
    assert (exfil["failed_login_attempts"] == 0).all()
    assert (slow["failed_login_attempts"] == 0).all()
    assert exfil["duration_seconds"].mean() < 50
    assert slow["duration_seconds"].mean() > 180
    assert slow["bytes_sent"].mean() < exfil["bytes_sent"].mean()


def test_rate_exfil_overlaps_normal_margins():
    df = generate_sessions(n_records=2000, n_per_scenario=8, seed=42)
    normal = df[df["label"] == "normal"]
    exfil = df[df["label"] == "rate_exfil"]
    nmin, nmax = normal["bytes_sent"].min(), normal["bytes_sent"].max()
    inside_bytes = ((exfil["bytes_sent"] >= nmin) & (exfil["bytes_sent"] <= nmax)).mean()
    inside_dur = (
        (exfil["duration_seconds"] >= normal["duration_seconds"].min())
        & (exfil["duration_seconds"] <= normal["duration_seconds"].max())
    ).mean()
    assert inside_bytes >= 0.8
    assert inside_dur == 1.0


def test_low_and_slow_overlaps_duration_and_bytes_margins():
    df = generate_sessions(n_records=2000, n_per_scenario=8, seed=42)
    normal = df[df["label"] == "normal"]
    slow = df[df["label"] == "low_and_slow"]
    inside_dur = (
        (slow["duration_seconds"] >= normal["duration_seconds"].min())
        & (slow["duration_seconds"] <= normal["duration_seconds"].max())
    ).mean()
    inside_bytes = (
        (slow["bytes_sent"] >= normal["bytes_sent"].min())
        & (slow["bytes_sent"] <= normal["bytes_sent"].max())
    ).mean()
    assert inside_dur == 1.0
    assert inside_bytes >= 0.8


def test_evaluate_binary_perfect_and_empty():
    y = [True, True, False, False]
    assert evaluate_binary(y, y)["f1"] == 1.0
    zeros = evaluate_binary([False, False], [False, False])
    assert zeros["precision"] == 0.0
    assert zeros["alerts"] == 0


def test_inductive_holdout_ranks_exactly_k():
    train = generate_sessions(n_records=2000, n_per_scenario=8, seed=42)
    test = generate_sessions(n_records=2000, n_per_scenario=8, seed=99)
    top_k = 30
    scored = SessionRanker(seed=42).fit(train).score_table(test, top_k=top_k)
    assert scored["is_anomaly"].sum() == top_k
    cmp_ = compare_rankers(scored, top_k)
    assert cmp_["isolation_forest"]["recall"] >= 0.25
    assert cmp_["mahalanobis"]["recall"] >= cmp_["isolation_forest"]["recall"]
    assert set(ATTACK_LABELS) <= set(cmp_["isolation_forest"]["recall_by_attack_type"])


def test_joint_case_is_harder_for_max_z_or_rules_than_a_1d_control():
    train = generate_sessions(n_records=2000, n_per_scenario=8, seed=42)
    test = generate_sessions(n_records=2000, n_per_scenario=8, seed=99)
    scored = SessionRanker(seed=42).fit(train).score_table(test, top_k=30)
    cmp_ = compare_rankers(scored, 30)
    brute_if = cmp_["isolation_forest"]["recall_by_attack_type"]["brute_force"]
    brute_z = cmp_["max_abs_z"]["recall_by_attack_type"]["brute_force"]
    joint_z = min(
        cmp_["max_abs_z"]["recall_by_attack_type"]["rate_exfil"],
        cmp_["max_abs_z"]["recall_by_attack_type"]["low_and_slow"],
    )
    assert brute_if >= 0.75
    assert brute_z >= 0.75
    assert joint_z <= brute_z


def test_ops_csv_without_labels_ranks_only(tmp_path):
    df = generate_sessions(n_records=160, n_per_scenario=2, seed=1)
    ops = df.drop(columns=["label", "is_attack"])
    path = tmp_path / "sessions.csv"
    ops.to_csv(path, index=False)
    loaded = load_sessions(path)
    scored = detect_anomalies(loaded, top_k=10)
    report = build_report(scored, top_k=10, protocol="transductive")
    assert not report["has_ground_truth"]
    assert report["comparison_at_k"] == {}
    assert len(report["top_alerts"]) == 10
    assert set(MODEL_FEATURES).issubset(scored.columns)


def test_missing_columns_fail_fast(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": [datetime(2026, 3, 12, 10, 0)], "bytes_sent": [1]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="missing columns"):
        load_sessions(path)
