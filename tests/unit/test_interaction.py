import numpy as np
from scipy.special import roots_legendre

import pyticc.matrix.interaction.atom_diatom as atom_diatom
import pyticc.matrix.interaction.diatom_diatom as diatom_diatom
from pyticc.basis.angle import clebsch_gordan
from pyticc.basis.channel import Channel, ChannelBasis
from pyticc.basis.podvr import RovibPODVR
from pyticc.matrix.interaction import contract
from pyticc.system import MolInnerState


def make_rovib(n_grid: int = 1, vmax: int = 0, jmax: int = 2) -> RovibPODVR:
    grids = np.arange(1, n_grid + 1, dtype=np.float64)
    WF_vj = np.ones((n_grid, vmax + 1, jmax + 1), dtype=np.float64)
    if n_grid > 1:
        WF_vj[:, 0, :] = np.eye(n_grid)[:, 0, None]
    return RovibPODVR(grids=grids, E_vj=np.zeros((vmax + 1, jmax + 1)), WF_vj=WF_vj)


def make_atom_diatom_basis() -> ChannelBasis:
    quantum_numbers = ((0, 0), (1, 0), (1, 1))
    channels = tuple(
        Channel(
            mis_X=MolInnerState(j=0),
            mis_Y=MolInnerState(v=0, j=j),
            j_couple=j,
            K=K,
            Jtot=1,
            system_parity=1,
            E_int=0.0,
            index=index,
        )
        for index, (j, K) in enumerate(quantum_numbers)
    )
    return ChannelBasis(channels)


def make_diatom_diatom_basis() -> ChannelBasis:
    quantum_numbers = (
        (0, 0, 0, 0),
        (1, 1, 0, 0),
        (1, 1, 2, 0),
        (1, 0, 1, 1),
    )
    channels = tuple(
        Channel(
            mis_X=MolInnerState(v=0, j=j_X),
            mis_Y=MolInnerState(v=0, j=j_Y),
            j_couple=j_couple,
            K=K,
            Jtot=2,
            system_parity=1,
            E_int=0.0,
            index=index,
        )
        for index, (j_X, j_Y, j_couple, K) in enumerate(quantum_numbers)
    )
    return ChannelBasis(channels)


def test_clebsch_gordan_uses_reference_phase_convention() -> None:
    np.testing.assert_allclose(clebsch_gordan(1, 1, 1, 0, 1), 1.0 / np.sqrt(2.0))
    np.testing.assert_allclose(clebsch_gordan(1, 0, 1, 1, 1), -1.0 / np.sqrt(2.0))


def test_constant_potential_is_identity_for_atom_diatom_basis() -> None:
    basis = make_atom_diatom_basis()
    rovib = make_rovib()
    cos_theta, theta_weights = roots_legendre(5)
    V_basis = atom_diatom.prepare(basis, rovib, cos_theta, theta_weights)

    Vmat = contract(V_basis, np.full(V_basis.grid_shape, 2.5))

    np.testing.assert_allclose(Vmat, 2.5 * np.eye(basis.n_channel), atol=1.0e-13)
    assert V_basis.grid_shape == (rovib.grids.size, cos_theta.size)


def test_constant_potential_is_identity_for_diatom_diatom_basis() -> None:
    basis = make_diatom_diatom_basis()
    rovib_X = make_rovib()
    rovib_Y = make_rovib()
    cos_theta_X, theta_weights_X = roots_legendre(5)
    cos_theta_Y, theta_weights_Y = roots_legendre(5)
    phi_x, phi_x_weights = roots_legendre(16)
    phi = 0.5 * np.pi * (phi_x + 1.0)
    phi_weights = 0.5 * np.pi * phi_x_weights
    V_basis = diatom_diatom.prepare(
        basis,
        rovib_X,
        rovib_Y,
        cos_theta_X,
        theta_weights_X,
        cos_theta_Y,
        theta_weights_Y,
        phi,
        phi_weights,
    )

    Vmat = contract(V_basis, np.full(V_basis.grid_shape, 1.75))

    np.testing.assert_allclose(Vmat, 1.75 * np.eye(basis.n_channel), atol=1.0e-12)


def test_contract_preserves_nncc_channel_order() -> None:
    basis = make_atom_diatom_basis()
    rovib = make_rovib()
    cos_theta, theta_weights = roots_legendre(5)
    V_basis = atom_diatom.prepare(basis, rovib, cos_theta, theta_weights)
    potential = np.broadcast_to(1.0 + np.arccos(cos_theta), V_basis.grid_shape)
    indices = (2, 0)

    full_Vmat = contract(V_basis, potential)
    block_Vmat = contract(V_basis, potential, indices)

    np.testing.assert_allclose(block_Vmat, full_Vmat[np.ix_(indices, indices)])


def test_contract_accepts_radial_batch() -> None:
    basis = make_atom_diatom_basis()
    rovib = make_rovib()
    cos_theta, theta_weights = roots_legendre(5)
    V_basis = atom_diatom.prepare(basis, rovib, cos_theta, theta_weights)
    potential = np.stack([np.full(V_basis.grid_shape, value) for value in (1.0, 2.0)])

    Vmat = contract(V_basis, potential)

    assert Vmat.shape == (2, basis.n_channel, basis.n_channel)
    np.testing.assert_allclose(Vmat[0], np.eye(basis.n_channel), atol=1.0e-13)
    np.testing.assert_allclose(Vmat[1], 2.0 * np.eye(basis.n_channel), atol=1.0e-13)
