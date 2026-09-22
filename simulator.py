"""Session telemetry on a correlated manifold, plus three labeled attack families.

Normal traffic lives on bytes ≈ rate × duration. Two attack families are off that
strip while remaining inside the *marginal* ranges — the case a joint estimator
is supposed to catch. Hour is generated on a 24h clock (later encoded as sin/cos).

low_and_slow uses the same hour mix as normal traffic. Night-only planting made
hour a cheap extra tell; that is no longer the default.
"""

from datetime import datetime

import numpy as np
import pandas as pd

LAB_DAY = datetime(2026, 3, 12)
BUSINESS_HOURS = [8, 9, 10, 11, 12, 14, 15, 16, 17]
ATTACK_LABELS = ("brute_force", "rate_exfil", "low_and_slow")
SCENARIO_HINTS = {
    "brute_force": "analogue of credential stuffing — 1D, failed logins (not ATT&CK coverage)",
    "rate_exfil": "analogue of high-rate transfer — joint, bytes/duration off-manifold",
    "low_and_slow": "analogue of drip transfer — joint, long + low rate",
}


def _timestamp(hour: int, minute: int) -> datetime:
    return LAB_DAY.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)


def _internal_ip(rng: np.random.Generator) -> str:
    return f"10.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 254)}"


def _hours(rng: np.random.Generator, n: int) -> np.ndarray:
    """Global-ish mix so night is not a 1D rarity flag."""
    roll = rng.random(n)
    hours = np.empty(n, dtype=int)
    day = roll < 0.62
    evening = (roll >= 0.62) & (roll < 0.84)
    night = roll >= 0.84
    hours[day] = rng.choice(BUSINESS_HOURS, size=int(day.sum()))
    hours[evening] = rng.choice([6, 7, 18, 19, 20, 21], size=int(evening.sum()))
    hours[night] = rng.choice([0, 1, 2, 3, 4, 5, 22, 23], size=int(night.sum()))
    return hours


def _on_manifold(
    rng: np.random.Generator,
    n: int,
    manifold: str = "single",
) -> tuple[np.ndarray, np.ndarray]:
    duration = np.clip(rng.exponential(scale=16, size=n), 1.0, 400.0)
    backup = rng.random(n) < 0.04
    duration[backup] = rng.uniform(90, 170, size=int(backup.sum()))
    vpn = rng.random(n) < 0.03
    duration[vpn] = rng.uniform(185, 255, size=int(vpn.sum()))
    if manifold == "bimodal":
        pick = rng.random(n) < 0.55
        rate = np.empty(n, dtype=float)
        rate[pick] = np.clip(rng.normal(420, 55, size=int(pick.sum())), 90, 800)
        rate[~pick] = np.clip(rng.normal(70, 12, size=int((~pick).sum())), 20, 140)
    elif manifold == "single":
        rate = np.clip(rng.normal(420, 55, size=n), 90, 800)
    else:
        raise ValueError("manifold must be 'single' or 'bimodal'.")
    bytes_sent = np.clip(rate * duration + rng.normal(0, 250, size=n), 80, None)
    return bytes_sent, duration


def generate_sessions(
    n_records: int = 2000,
    n_per_scenario: int = 8,
    seed: int = 42,
    manifold: str = "single",
    slow_hours: str = "mixed",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_attacks = 3 * n_per_scenario
    if n_attacks >= n_records:
        raise ValueError("Need more records than planted attacks.")
    n_normal = n_records - n_attacks

    hours = _hours(rng, n_normal)
    minutes = rng.integers(0, 60, size=n_normal)
    bytes_sent, duration = _on_manifold(rng, n_normal, manifold=manifold)
    failed = np.zeros(n_normal, dtype=int)
    failed[rng.random(n_normal) < 0.05] = 1
    failed[rng.random(n_normal) < 0.01] = 2

    normal = pd.DataFrame(
        {
            "timestamp": [_timestamp(h, m) for h, m in zip(hours, minutes)],
            "src_ip": [_internal_ip(rng) for _ in range(n_normal)],
            "bytes_sent": bytes_sent,
            "duration_seconds": duration,
            "failed_login_attempts": failed,
            "hour_of_day": hours,
            "label": "normal",
            "is_attack": False,
        }
    )
    attacks = [
        _brute_force(rng, n_per_scenario),
        _rate_exfil(rng, n_per_scenario),
        _low_and_slow(rng, n_per_scenario, slow_hours=slow_hours),
    ]
    df = pd.concat([normal, *attacks], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _brute_force(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Extreme on failed logins only; volume/duration stay on the manifold."""
    hours = _hours(rng, n)
    minutes = rng.integers(0, 60, size=n)
    duration = np.clip(rng.normal(8, 3, size=n), 2, 20)
    rate = np.clip(rng.normal(420, 40, size=n), 90, 800)
    bytes_sent = np.clip(rate * duration + rng.normal(0, 80, size=n), 80, None)
    return pd.DataFrame(
        {
            "timestamp": [_timestamp(h, m) for h, m in zip(hours, minutes)],
            "src_ip": [_internal_ip(rng) for _ in range(n)],
            "bytes_sent": bytes_sent,
            "duration_seconds": duration,
            "failed_login_attempts": rng.integers(9, 18, size=n),
            "hour_of_day": hours,
            "label": "brute_force",
            "is_attack": True,
        }
    )


def _rate_exfil(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Typical duration, bytes too high *for that duration* — still inside the
    global bytes range of long on-manifold sessions. Failed logins stay 0.
    Business-hour heavy so hour is not the tell.
    """
    hours = rng.choice(BUSINESS_HOURS, size=n)
    minutes = rng.integers(0, 60, size=n)
    duration = rng.uniform(10, 40, size=n)
    rate = np.clip(rng.normal(3600, 250, size=n), 2600, 4800)
    bytes_sent = np.clip(rate * duration + rng.normal(0, 400, size=n), 80, None)
    return pd.DataFrame(
        {
            "timestamp": [_timestamp(h, m) for h, m in zip(hours, minutes)],
            "src_ip": [_internal_ip(rng) for _ in range(n)],
            "bytes_sent": bytes_sent,
            "duration_seconds": duration,
            "failed_login_attempts": 0,
            "hour_of_day": hours,
            "label": "rate_exfil",
            "is_attack": True,
        }
    )


def _low_and_slow(rng: np.random.Generator, n: int, slow_hours: str = "mixed") -> pd.DataFrame:
    """Duration overlaps forgotten-VPN, bytes overlap short sessions (low rate)."""
    if slow_hours == "night":
        hours = rng.choice([0, 1, 2, 3, 4, 5], size=n)
    elif slow_hours == "mixed":
        hours = _hours(rng, n)
    else:
        raise ValueError("slow_hours must be 'mixed' or 'night'.")
    minutes = rng.integers(0, 60, size=n)
    duration = rng.uniform(185, 255, size=n)
    rate = np.clip(rng.normal(22, 4, size=n), 8, 40)
    bytes_sent = np.clip(rate * duration + rng.normal(0, 80, size=n), 80, None)
    return pd.DataFrame(
        {
            "timestamp": [_timestamp(h, m) for h, m in zip(hours, minutes)],
            "src_ip": [_internal_ip(rng) for _ in range(n)],
            "bytes_sent": bytes_sent,
            "duration_seconds": duration,
            "failed_login_attempts": 0,
            "hour_of_day": hours,
            "label": "low_and_slow",
            "is_attack": True,
        }
    )
