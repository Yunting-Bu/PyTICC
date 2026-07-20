from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pyticc.pes.fortran.executor as executor_module


class _FakePool:
    def __init__(self, coordinates: np.ndarray) -> None:
        self.coordinates = coordinates
        self.close_calls = 0
        self.terminate_calls = 0
        self.join_calls = 0

    def map(self, function: Any, radial_points: np.ndarray, chunksize: int) -> list[np.ndarray]:
        return [np.full(self.coordinates.shape[1], RR) for RR in radial_points]

    def close(self) -> None:
        self.close_calls += 1

    def terminate(self) -> None:
        self.terminate_calls += 1

    def join(self) -> None:
        self.join_calls += 1


class _FakeContext:
    def __init__(self) -> None:
        self.pools: list[_FakePool] = []

    def Pool(self, *, processes: int, initializer: Any, initargs: tuple[Any, ...]) -> _FakePool:
        pool = _FakePool(initargs[1])
        self.pools.append(pool)
        return pool


def test_fortran_executor_reuses_workers_until_coordinates_change(monkeypatch: pytest.MonkeyPatch) -> None:
    def interaction(RR: float, coordinates: np.ndarray) -> np.ndarray:
        return np.full(coordinates.shape[1], RR)

    context = _FakeContext()
    monkeypatch.setattr(executor_module.mp, "get_context", lambda method: context)
    spec = executor_module._FortranPESSpec("fake", Path("fake.so"), None)
    executor = executor_module._FortranPESExecutor(spec, processes=2, interaction=interaction)
    coordinates = np.arange(6.0).reshape(2, 3)

    first = executor.evaluate(np.array([3.0, 4.0]), coordinates)
    second = executor.evaluate(np.array([5.0]), coordinates.copy())

    assert len(context.pools) == 1
    np.testing.assert_allclose(first, [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    np.testing.assert_allclose(second, [[5.0, 5.0, 5.0]])

    executor.evaluate(np.array([6.0]), coordinates + 1.0)
    assert len(context.pools) == 2
    assert context.pools[0].close_calls == 1
    assert context.pools[0].join_calls == 1

    executor.close()
    assert context.pools[1].close_calls == 1
    assert context.pools[1].join_calls == 1
