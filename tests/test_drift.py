"""漂移指标测试：验证 PSI 的数学性质是对的。

为什么值得测？因为一个算错的监控指标比没有监控更危险 ——
它会让团队误以为「一切正常」。
"""

from __future__ import annotations

import numpy as np
import pytest

from monitor.metrics import bin_shares, psi, status


def test_identical_distribution_is_zero() -> None:
    dist = [0.1] * 10
    assert psi(dist, dist) == pytest.approx(0.0, abs=1e-9)


def test_psi_is_non_negative() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.dirichlet(np.ones(10))
        b = rng.dirichlet(np.ones(10))
        assert psi(a.tolist(), b.tolist()) >= 0


def test_completely_disjoint_is_large() -> None:
    expected = [0.25, 0.25, 0.25, 0.25]
    actual = [0.0, 0.0, 0.0, 1.0]
    value = psi(expected, actual)
    assert value > 1.0
    assert status(value) == "DRIFT"


def test_small_change_is_stable() -> None:
    expected = [0.1] * 10
    actual = [0.105] * 9 + [0.055]
    assert status(psi(expected, actual)) in {"STABLE", "WATCH"}


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        psi([0.5, 0.5], [0.3, 0.3, 0.4])


def test_bin_shares_sums_to_one() -> None:
    values = np.random.default_rng(0).normal(size=1000)
    edges = np.quantile(values, np.linspace(0, 1, 11))
    shares = bin_shares(values, edges)
    assert len(shares) == 10
    assert sum(shares) == pytest.approx(1.0)


def test_bin_shares_handles_empty() -> None:
    assert bin_shares(np.array([]), [0, 1, 2]) == [0.0, 0.0]


def test_bin_shares_uses_given_edges_not_recomputed() -> None:
    """关键性质：边界必须来自训练集。整批数据跑到了最后一箱，就该显示 100%。"""
    edges = [0, 1, 2, 3, 4]
    shares = bin_shares(np.array([3.5, 3.6, 3.9]), edges)
    assert shares[-1] == pytest.approx(1.0)


def test_status_thresholds() -> None:
    assert status(0.05) == "STABLE"
    assert status(0.15) == "WATCH"
    assert status(0.26) == "DRIFT"
