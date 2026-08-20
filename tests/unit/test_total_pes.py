import numpy as np
import pytest

from pyticc.pes.total import TotalPES


def test_total_pes_validates_and_preserves_the_three_bond_contract() -> None:
    received: list[np.ndarray] = []

    def potential(bonds: np.ndarray) -> np.ndarray:
        assert bonds.flags.f_contiguous
        received.append(bonds.copy())
        return bonds[0] + 2.0 * bonds[1] + 3.0 * bonds[2]

    pes = TotalPES(potential)
    bonds = np.array([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
    values = pes(bonds)

    np.testing.assert_allclose(received[0], bonds)
    np.testing.assert_allclose(values, bonds[0] + 2.0 * bonds[1] + 3.0 * bonds[2])


@pytest.mark.parametrize(
    ("bonds", "message"),
    [
        (np.ones(3), "shape"),
        (np.ones((2, 4)), "shape"),
        (np.array([[1.0], [0.0], [2.0]]), "positive"),
        (np.array([[1.0], [np.nan], [2.0]]), "positive"),
    ],
)
def test_total_pes_rejects_invalid_bond_arrays(bonds: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TotalPES(lambda coordinates: np.zeros(coordinates.shape[1]))(bonds)


def test_total_pes_rejects_invalid_results() -> None:
    bonds = np.ones((3, 2))

    with pytest.raises(ValueError, match="returned shape"):
        TotalPES(lambda coordinates: np.zeros((1, coordinates.shape[1])))(bonds)
    with pytest.raises(ValueError, match="non-finite"):
        TotalPES(lambda coordinates: np.full(coordinates.shape[1], np.nan))(bonds)
    with pytest.raises(ValueError, match="real energies"):
        TotalPES(lambda coordinates: np.full(coordinates.shape[1], 1.0j))(bonds)


def test_total_pes_close_delegates_to_the_owned_resource() -> None:
    closed: list[bool] = []
    pes = TotalPES(lambda bonds: np.zeros(bonds.shape[1]), _close=lambda: closed.append(True))

    pes.close()

    assert closed == [True]
