import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import roots_legendre

import pyticc as ticc
from pyticc.scattering import diabatic_atom_diatom


def _diabatic_basis(jpar: int | tuple[int, int] = 0) -> ticc.DiabaticDiatomBasis:
    def monomer(offset: float):
        def potential(r: NDArray[np.float64]) -> NDArray[np.float64]:
            return 0.03 * (r - 2.0) ** 2 + offset

        return potential

    dvrs = (
        ticc.build_SineDVR(1.0, 4.0, 24, 900.0, monomer(0.0)),
        ticc.build_SineDVR(1.0, 4.0, 24, 900.0, monomer(0.01)),
    )
    return ticc.build_DiabaticDiatomBasis(dvrs, n_podvr=6, vmax=0, jmax=0, mass=900.0, jpar=jpar)


def _pes(coupling: float, n_state: int = 2) -> ticc.DiabaticPESWrapper:
    def monomer(r: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros((r.size, n_state))

    def interaction(R: float, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = np.zeros((n_state, n_state))
        if n_state == 2:
            matrix[0, 1] = matrix[1, 0] = coupling * np.exp(-5.0 * (R - 3.0))
        return np.broadcast_to(matrix, (coordinates.shape[1], n_state, n_state))

    return ticc.DiabaticPESWrapper(n_state=n_state, monomer=monomer, interaction=interaction)


def _solve(
    diatom: ticc.DiabaticDiatomBasis,
    pes: ticc.DiabaticPESWrapper,
    *,
    n_theta: int = 6,
    propagation: ticc.Propagation | None = None,
) -> ticc.ScatteringResult:
    system = ticc.ScattSystem(
        ticc.AtomSpec(),
        diatom,
        Jtot=0,
        system_parity=1,
        potential=pes,
        reduced_mass=1000.0,
    )
    hamiltonian = diabatic_atom_diatom.build_hamiltonian(system, n_theta=n_theta)
    radial = ticc.Propagation((3.0, 4.0), (0.1,)) if propagation is None else propagation
    result = ticc.solve(hamiltonian, [0.05], radial)
    assert isinstance(result, ticc.ScatteringResult)
    return result


def test_solve_diabatic_atom_diatom_returns_unitary_coupled_state_smatrix() -> None:
    result = _solve(_diabatic_basis(), _pes(0.02))

    assert isinstance(result.basis, ticc.ChannelBasis)
    assert result.basis.n_channel == 2
    assert {channel.mis_Y.electronic_state for channel in result.basis} == {0, 1}
    assert result.Y_propagated.shape == (1, 2, 2)
    assert result.Smat[0].shape == (2, 2)
    assert abs(result.Smat[0][0, 1]) > 1.0e-4
    np.testing.assert_allclose(result.Smat[0].conj().T @ result.Smat[0], np.eye(2), atol=1.0e-11)


def test_solve_diabatic_atom_diatom_zero_coupling_separates_electronic_states() -> None:
    result = _solve(_diabatic_basis(), _pes(0.0))

    np.testing.assert_allclose(result.Smat[0] - np.diag(np.diag(result.Smat[0])), 0.0, atol=1.0e-13)


def test_solve_diabatic_atom_diatom_uses_half_angle_rule_when_all_states_have_exchange_parity() -> None:
    sampled_angles: list[np.ndarray] = []

    def monomer(r: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros((r.size, 2))

    def interaction(R: float, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        sampled_angles.append(np.unique(coordinates[1]))
        return np.zeros((coordinates.shape[1], 2, 2))

    _solve(
        _diabatic_basis(jpar=(1, 1)),
        ticc.DiabaticPESWrapper(n_state=2, monomer=monomer, interaction=interaction),
        n_theta=3,
        propagation=ticc.Propagation((3.0, 3.2), (0.1,)),
    )

    full_cos_theta, _ = roots_legendre(6)
    assert sampled_angles
    np.testing.assert_allclose(sampled_angles[0], np.sort(np.arccos(full_cos_theta[:3])))


def test_build_diabatic_atom_diatom_validates_electronic_state_count() -> None:
    with pytest.raises(ValueError, match="electronic states"):
        system = ticc.ScattSystem(
            ticc.AtomSpec(),
            _diabatic_basis(),
            Jtot=0,
            system_parity=1,
            potential=_pes(0.0, n_state=1),
            reduced_mass=1000.0,
        )
        diabatic_atom_diatom.build_hamiltonian(system)
