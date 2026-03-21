from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KellyResult:
    fraction: float
    raw_kelly: float
    edge: float
    expected_log_growth: float
    break_even_prob: float
    effective_share_price: float
    net_odds: float


def kelly_fraction_binary(
    p_win: float,
    share_price: float,
    fee_rate: float = 0.0,
    fractional_kelly: float = 0.5,
    max_fraction: float = 0.20,
) -> KellyResult:
    if not 0 < share_price < 1:
        raise ValueError("share_price must be between 0 and 1")
    if not 0 <= p_win <= 1:
        raise ValueError("p_win must be between 0 and 1")

    # Fees are folded into the share price so the break-even probability is
    # computed on the all-in cost rather than the headline market price.
    effective_price = float(np.clip(share_price * (1.0 + fee_rate), 1e-6, 0.999999))
    net_odds = (1.0 - effective_price) / effective_price

    q = 1.0 - p_win
    # Binary Kelly fraction:
    # f* = (b p - q) / b, where b is net odds and q = 1 - p.
    raw_kelly = ((net_odds * p_win) - q) / net_odds
    raw_kelly = float(max(0.0, raw_kelly))

    # fractional_kelly and max_fraction are risk-policy controls, not
    # statistical calibration. They intentionally shrink the theoretical bet.
    fraction = raw_kelly * fractional_kelly
    fraction = float(np.clip(fraction, 0.0, max_fraction))

    growth = 0.0
    if fraction > 0:
        # Expected log growth under Kelly:
        # E[log(1 + f b)] * p + E[log(1 - f)] * (1 - p)
        growth = float(
            p_win * np.log1p(fraction * net_odds)
            + (1.0 - p_win) * np.log1p(-fraction)
        )

    return KellyResult(
        fraction=fraction,
        raw_kelly=raw_kelly,
        edge=float(p_win - effective_price),
        expected_log_growth=growth,
        break_even_prob=effective_price,
        effective_share_price=effective_price,
        net_odds=float(net_odds),
    )
