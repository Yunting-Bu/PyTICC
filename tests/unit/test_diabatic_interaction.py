import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import roots_legendre

from pyticc.basis.channel import Channel, ChannelBasis
from pyticc.basis.diabatic import DiabaticDiatomBasis, build_DiabaticDiatomBasis
from pyticc.basis.dvr import build_SineDVR
from pyticc.matrix.diabatic import (
    get_DiabaticVgrid_BF_atom_diatom,
    get_DiabaticVmat_BF,
    prepare_DiabaticVmat_BF_atom_diatom,
)
from pyticc.matrix.interaction import get_Vmat_BF, prepare_Vmat_BF_atom_diatom
from pyticc.pes.diabatic import DiabaticPESWrapper
from pyticc.system import MolInnerState


def _diabatic_basis(n_state: int = 2) -> DiabaticDiatomBasis:
    def potential(offset: float):
        def evaluate(r: NDArray[np.float64]) -> NDArray[np.float64]:
            return 0.03 * (r - 2.1) ** 2 + offset

        return evaluate

    dvrs = tuple(build_SineDVR(1.0, 4.0, 24, 900.0, potential(0.1 * state)) for state in range(n_state))
    return build_DiabaticDiatomBasis(
        dvrs,
        n_podvr=6,
        vmax=0,
        jmax=1,
        mass=900.0,
        energy_reference=0.0,
    )


def _channels(n_state: int = 2) -> ChannelBasis:
    quantum_numbers = tuple((state, j, K) for K, j in ((0, 0), (1, 1)) for state in range(n_state))
    channels = tuple(
        Channel(
            mis_X=MolInnerState(j=0),
            mis_Y=MolInnerState(v=0, j=j, electronic_state=state),
            j_couple=j,
            K=K,
            Jtot=1,
            system_parity=1,
            E_int=0.0,
            index=index,
        )
        for index, (state, j, K) in enumerate(quantum_numbers)
    )
    return ChannelBasis(channels)


def _constant_pes(matrix: NDArray[np.float64]) -> DiabaticPESWrapper:
    n_state = matrix.shape[0]

    def monomer(r: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros((r.size, n_state))

    def interaction(R: float, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.broadcast_to(matrix, (coordinates.shape[1], n_state, n_state))

    return DiabaticPESWrapper(n_state=n_state, monomer=monomer, interaction=interaction)


def test_diabatic_interaction_uses_state_specific_diagonal_and_shared_coupling_grids() -> None:
    diabatic_basis = _diabatic_basis()
    channels = _channels()
    cos_theta, weights = roots_legendre(6)
    V_basis = prepare_DiabaticVmat_BF_atom_diatom(channels, diabatic_basis, cos_theta, weights)

    assert tuple(grid.size for grid in V_basis.diagonal_grids) == (6, 6)
    assert V_basis.coupling_grid.size == 24
    assert V_basis.B_diagonal[(0, 0)].shape == (1, 6 * 6)
    assert V_basis.B_coupling[(0, 0)].shape == (1, 24 * 6)


def test_diabatic_interaction_contracts_diagonal_and_offdiagonal_blocks() -> None:
    diabatic_basis = _diabatic_basis()
    channels = _channels()
    cos_theta, weights = roots_legendre(6)
    V_basis = prepare_DiabaticVmat_BF_atom_diatom(channels, diabatic_basis, cos_theta, weights)
    coupling = 0.4
    pes = _constant_pes(np.array([[2.0, coupling], [coupling, 3.0]]))

    potential = get_DiabaticVgrid_BF_atom_diatom(pes, 5.0, V_basis)
    Vmat = get_DiabaticVmat_BF(V_basis, potential)

    radial_overlap_j0 = diabatic_basis.state(0).rovib_dvr.WF_vj[:, 0, 0] @ diabatic_basis.state(1).rovib_dvr.WF_vj[:, 0, 0]
    radial_overlap_j1 = diabatic_basis.state(0).rovib_dvr.WF_vj[:, 0, 1] @ diabatic_basis.state(1).rovib_dvr.WF_vj[:, 0, 1]
    expected = np.diag([2.0, 3.0, 2.0, 3.0])
    expected[0, 1] = expected[1, 0] = coupling * radial_overlap_j0
    expected[2, 3] = expected[3, 2] = coupling * radial_overlap_j1

    np.testing.assert_allclose(Vmat, expected, atol=1.0e-12)
    np.testing.assert_allclose(Vmat, Vmat.T, atol=0.0)
    assert Vmat[0, 3] == pytest.approx(0.0)


def test_diabatic_interaction_zero_coupling_separates_electronic_states() -> None:
    diabatic_basis = _diabatic_basis()
    channels = _channels()
    cos_theta, weights = roots_legendre(5)
    V_basis = prepare_DiabaticVmat_BF_atom_diatom(channels, diabatic_basis, cos_theta, weights)
    potential = get_DiabaticVgrid_BF_atom_diatom(_constant_pes(np.diag([1.5, 2.5])), 6.0, V_basis)

    Vmat = get_DiabaticVmat_BF(V_basis, potential)

    np.testing.assert_allclose(Vmat, np.diag([1.5, 2.5, 1.5, 2.5]), atol=1.0e-12)


def test_diabatic_interaction_radial_batch_matches_scalar_and_preserves_selection_order() -> None:
    diabatic_basis = _diabatic_basis()
    channels = _channels()
    cos_theta, weights = roots_legendre(5)
    V_basis = prepare_DiabaticVmat_BF_atom_diatom(channels, diabatic_basis, cos_theta, weights)

    def monomer(r: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros((r.size, 2))

    def interaction(R: float, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = np.array([[R, 0.1 * R], [0.1 * R, R + 1.0]])
        return np.broadcast_to(matrix, (coordinates.shape[1], 2, 2))

    pes = DiabaticPESWrapper(n_state=2, monomer=monomer, interaction=interaction)
    radial_points = np.array([4.0, 5.0])
    selected = (3, 0, 1)
    batched = get_DiabaticVmat_BF(V_basis, get_DiabaticVgrid_BF_atom_diatom(pes, radial_points, V_basis), selected)
    scalar = np.stack([get_DiabaticVmat_BF(V_basis, get_DiabaticVgrid_BF_atom_diatom(pes, R, V_basis), selected) for R in radial_points])

    assert batched.shape == (2, len(selected), len(selected))
    np.testing.assert_allclose(batched, scalar, atol=1.0e-13)


def test_diabatic_interaction_samples_all_state_grids_in_one_radial_batch() -> None:
    diabatic_basis = _diabatic_basis()
    channels = _channels()
    cos_theta, weights = roots_legendre(5)
    V_basis = prepare_DiabaticVmat_BF_atom_diatom(channels, diabatic_basis, cos_theta, weights)
    calls = 0

    def monomer(r: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros((r.size, 2))

    def interaction(R: float, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros((coordinates.shape[1], 2, 2))

    def interaction_many(R: NDArray[np.float64], coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal calls
        calls += 1
        return np.zeros((R.size, coordinates.shape[1], 2, 2))

    pes = DiabaticPESWrapper(n_state=2, monomer=monomer, interaction=interaction, interaction_many=interaction_many)
    potential = get_DiabaticVgrid_BF_atom_diatom(pes, np.array([4.0, 5.0]), V_basis)

    assert calls == 1
    assert tuple(values.shape for values in potential.diagonal) == ((2, 6, 5), (2, 6, 5))
    assert potential.coupling.shape == (2, 24, 5, 2, 2)


def test_one_state_diabatic_contraction_reduces_to_scalar_interaction() -> None:
    diabatic_basis = _diabatic_basis(n_state=1)
    channels = _channels(n_state=1)
    cos_theta, weights = roots_legendre(6)
    theta = np.arccos(cos_theta)
    new_basis = prepare_DiabaticVmat_BF_atom_diatom(channels, diabatic_basis, cos_theta, weights)
    old_basis = prepare_Vmat_BF_atom_diatom(channels, diabatic_basis.state(0).rovib, cos_theta, weights)

    def monomer(r: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros((r.size, 1))

    def interaction(R: float, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        r, angle = coordinates
        return (R + 0.2 * r + np.cos(angle))[:, None, None]

    pes = DiabaticPESWrapper(n_state=1, monomer=monomer, interaction=interaction)
    potential = get_DiabaticVgrid_BF_atom_diatom(pes, 5.0, new_basis)
    new_Vmat = get_DiabaticVmat_BF(new_basis, potential)
    r_grid = diabatic_basis.state(0).rovib.grids
    scalar_grid = 5.0 + 0.2 * r_grid[:, None] + np.cos(theta)[None, :]
    old_Vmat = get_Vmat_BF(old_basis, scalar_grid)

    np.testing.assert_allclose(new_Vmat, old_Vmat, atol=1.0e-13)
