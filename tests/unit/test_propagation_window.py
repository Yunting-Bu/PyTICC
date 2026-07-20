import numpy as np

from pyticc.propagation.grid import build_radial_sectors, iter_radial_windows


def test_memory_planner_falls_back_to_one_sector_and_caps_large_windows() -> None:
    sectors = build_radial_sectors([3.0, 23.0], [0.1])
    small = list(iter_radial_windows(sectors, n_grid=10_000, n_channel=20, n_energy=4, memory_limit_mb=1.0e-6))
    large = list(iter_radial_windows(sectors, n_grid=1, n_channel=1, n_energy=1, memory_limit_mb=1.0e6))

    assert len(small[0][0]) == 1
    assert len(large[0][0]) == 64


def test_radial_windows_share_only_their_endpoint() -> None:
    sectors = build_radial_sectors([3.0, 3.8], [0.1])
    windows = list(iter_radial_windows(sectors, n_grid=0, n_channel=1, n_energy=1, memory_limit_mb=1.0e-6))

    assert [len(window) for window, _ in windows] == [1, 1, 1, 1]
    first = windows[0][1]
    second = windows[1][1]
    assert first.shape == (3,)
    assert second.shape == (3,)
    np.testing.assert_allclose(first[-1], second[0])
