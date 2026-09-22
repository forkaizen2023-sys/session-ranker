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
from sweep import summarize, sweep


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


def test_low_and_slow_is_not_night_only():
    df = generate_sessions(n_records=2000, n_per_scenario=8, seed=42)
    slow = df[df["label"] == "low_and_slow"]
    assert slow["hour_of_day"].nunique() >= 3
    assert not set(slow["hour_of_day"]).issubset({0, 1, 2, 3, 4, 5})


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
    scored = SessionRanker(seed=42).fit(train, fit_on="benign").score_table(test, top_k=top_k)
    assert scored["is_anomaly"].sum() == top_k
    cmp_ = compare_rankers(scored, top_k)
    assert cmp_["isolation_forest"]["recall"] >= 0.25
    assert cmp_["mahalanobis"]["recall"] >= cmp_["isolation_forest"]["recall"]
    assert cmp_["mahalanobis"]["recall"] >= 0.9
    assert cmp_["mahalanobis"]["auroc"] < 1.0 or cmp_["mahalanobis"]["margin"] >= 0
    assert set(ATTACK_LABELS) <= set(cmp_["isolation_forest"]["recall_by_attack_type"])


def test_joint_case_is_harder_for_max_z_than_the_1d_control():
    train = generate_sessions(n_records=2000, n_per_scenario=8, seed=42)
    test = generate_sessions(n_records=2000, n_per_scenario=8, seed=99)
    scored = SessionRanker(seed=42).fit(train, fit_on="benign").score_table(test, top_k=30)
    cmp_ = compare_rankers(scored, 30)
    brute_z = cmp_["max_abs_z"]["recall_by_attack_type"]["brute_force"]
    joint_z = min(
        cmp_["max_abs_z"]["recall_by_attack_type"]["rate_exfil"],
        cmp_["max_abs_z"]["recall_by_attack_type"]["low_and_slow"],
    )
    assert brute_z >= 0.75
    assert joint_z <= brute_z


def test_rate_residual_beats_if_on_joint_families():
    train = generate_sessions(n_records=2000, n_per_scenario=8, seed=42)
    test = generate_sessions(n_records=2000, n_per_scenario=8, seed=99)
    scored = SessionRanker(seed=42).fit(train, fit_on="benign").score_table(test, top_k=30)
    cmp_ = compare_rankers(scored, 30)
    joint_if = min(
        cmp_["isolation_forest"]["recall_by_attack_type"]["rate_exfil"],
        cmp_["isolation_forest"]["recall_by_attack_type"]["low_and_slow"],
    )
    joint_res = min(
        cmp_["rate_residual"]["recall_by_attack_type"]["rate_exfil"],
        cmp_["rate_residual"]["recall_by_attack_type"]["low_and_slow"],
    )
    assert joint_res >= joint_if
    assert joint_res >= 0.75


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


def test_negative_bytes_fail_fast(tmp_path):
    path = tmp_path / "neg.csv"
    pd.DataFrame(
        {
            "timestamp": [datetime(2026, 3, 12, 10, 0)],
            "bytes_sent": [-4],
            "duration_seconds": [3],
            "failed_login_attempts": [0],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="negative"):
        load_sessions(path)


def test_model_roundtrip(tmp_path):
    train = generate_sessions(n_records=400, n_per_scenario=4, seed=2)
    test = generate_sessions(n_records=400, n_per_scenario=4, seed=3)
    ranker = SessionRanker(seed=2).fit(train, fit_on="benign")
    path = tmp_path / "model.pkl"
    ranker.save(path)
    loaded = SessionRanker.load(path)
    a = ranker.score_table(test, 10)["mahal_score"].to_numpy()
    b = loaded.score_table(test, 10)["mahal_score"].to_numpy()
    assert (abs(a - b) < 1e-9).all()


def test_bimodal_is_harder_than_single_for_mahalanobis():
    single = generate_sessions(n_records=2000, n_per_scenario=8, seed=42, manifold="single")
    test_s = generate_sessions(n_records=2000, n_per_scenario=8, seed=99, manifold="single")
    bimodal = generate_sessions(n_records=2000, n_per_scenario=8, seed=42, manifold="bimodal")
    test_b = generate_sessions(n_records=2000, n_per_scenario=8, seed=99, manifold="bimodal")
    cmp_s = compare_rankers(
        SessionRanker(seed=42).fit(single, fit_on="benign").score_table(test_s, 30), 30
    )
    cmp_b = compare_rankers(
        SessionRanker(seed=42).fit(bimodal, fit_on="benign").score_table(test_b, 30), 30
    )
    assert cmp_b["mahalanobis"]["ap"] <= cmp_s["mahalanobis"]["ap"] + 1e-9


def test_seed_sweep_mahalanobis_floor():
    rows = sweep(test_seeds=(21, 99, 123), train_seed=42, fit_on="benign")
    stats = summarize(rows, "mahalanobis")
    assert stats["recall_min"] >= 0.85
    assert stats["auroc_min"] >= 0.99
    assert any(r["mahalanobis_inversions"] > 0 for r in rows)
