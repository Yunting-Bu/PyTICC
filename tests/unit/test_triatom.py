import numpy as np
import pytest

from pyticc.basis.dvr import build_SineDVR
from pyticc.basis.triatom import TriatomBasis, _match_K0_states, build_TriatomBasis
from pyticc.system import MolInnerState, MonomerType


def test_triatom_basis_iterates_finite_jt_states_below_cutoff() -> None:
    energies = np.array(
        [
            [0.0, np.inf],
            [0.01, 0.03],
            [0.02, 0.04],
        ]
    )
    triatom = TriatomBasis(Eint=energies, jmax=2, tmax=1)

    states = list(triatom.mis_iter(0.025))

    assert triatom.type is MonomerType.TRIATOM
    assert [(state.j, state.t, state.Eint) for state in states] == [
        (0, 0, 0.0),
        (1, 0, 0.01),
        (2, 0, 0.02),
    ]


def test_triatom_basis_returns_state_energy() -> None:
    triatom = TriatomBasis(Eint=np.array([[0.0], [0.02]]), jmax=1, tmax=0)

    assert triatom.energy(MolInnerState(j=1, t=0), K=1) == pytest.approx(0.02)


def test_build_triatom_basis_solves_and_matches_K_blocks() -> None:
    masses = (1000.0, 1800.0, 1200.0)
    equilibrium = (2.0, 2.2, 1.8)

    def potential(coordinates: np.ndarray) -> np.ndarray:
        r1, r2, theta = coordinates
        return 0.01 * (r1 - equilibrium[0]) ** 2 + 0.012 * (r2 - equilibrium[1]) ** 2 + 0.005 * (np.cos(theta) - np.cos(equilibrium[2])) ** 2

    def potential_1(r1: np.ndarray) -> np.ndarray:
        coordinates = np.stack((r1, np.full(r1.size, equilibrium[1]), np.full(r1.size, equilibrium[2])))
        return potential(coordinates)

    def potential_2(r2: np.ndarray) -> np.ndarray:
        coordinates = np.stack((np.full(r2.size, equilibrium[0]), r2, np.full(r2.size, equilibrium[2])))
        return potential(coordinates)

    dvr_1 = build_SineDVR(1.2, 2.8, 16, masses[0], potential_1)
    dvr_2 = build_SineDVR(1.4, 3.0, 16, masses[2], potential_2)
    triatom = build_TriatomBasis(
        potential=potential,
        dvr_1=dvr_1,
        dvr_2=dvr_2,
        n_podvr=(3, 3),
        vmax=(1, 1),
        masses=masses,
        equilibrium=equilibrium,
        n_theta=12,
        j1max=2,
        j2max=1,
        tmax=3,
        parity_block_sign=1,
    )

    assert triatom.Eint[0, 0] == pytest.approx(0.0)
    assert np.all(np.isfinite(triatom.Eint[1]))
    assert triatom.K0_available is not None
    assert np.any(triatom.K0_available[1])
    assert np.allclose(triatom.positive_K_blocks[1].coefficients.T @ triatom.positive_K_blocks[1].coefficients, np.eye(4))
    assert np.allclose(triatom.K0_blocks[1].coefficients.T @ triatom.K0_blocks[1].coefficients, np.eye(triatom.K0_blocks[1].t_indices.size))


def test_K0_matching_does_not_force_unrelated_states_into_assignment() -> None:
    rows, columns = _match_K0_states(
        K0_energies=np.array([0.0, 2.0, 4.0, 100.0]),
        positive_energies=np.arange(6.0),
        tolerance=1.0e-12,
    )

    assert rows.tolist() == [0, 1, 2]
    assert columns.tolist() == [0, 2, 4]


@pytest.mark.parametrize(
    ("Eint", "jmax", "tmax", "message"),
    [
        (np.zeros(2), 1, 0, "two-dimensional"),
        (np.zeros((1, 1)), 1, 0, "does not cover"),
    ],
)
def test_triatom_basis_validates_energy_array(Eint: np.ndarray, jmax: int, tmax: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TriatomBasis(Eint=Eint, jmax=jmax, tmax=tmax)
