import numpy as np
from scipy.special import roots_legendre

import pyticc.matrix.interaction.atom_diatom as atom_diatom
import pyticc.matrix.interaction.diatom_diatom as diatom_diatom
from pyticc.basis.angle import clebsch_gordan, gauss_legendre_dvr
from pyticc.basis.channel import Channel, ChannelBasis, ChannelBasisElectricSF, build_ChannelBasisElectricSF
from pyticc.basis.monomer.diatom_electric import DiatomElectricBasis, DiatomElectricBlock
from pyticc.basis.rovib import RovibBasis
from pyticc.matrix.interaction import contract
from pyticc.pes import PESWrapper, get_Vgrid_atom_diatom_electric_sf
from pyticc.system import MolInnerState


def make_rovib(n_grid: int = 1, vmax: int = 0, jmax: int = 2) -> RovibBasis:
    grids = np.arange(1, n_grid + 1, dtype=np.float64)
    WF_vj = np.ones((n_grid, vmax + 1, jmax + 1), dtype=np.float64)
    if n_grid > 1:
        WF_vj[:, 0, :] = np.eye(n_grid)[:, 0, None]
    return RovibBasis(grids=grids, E_vj=np.zeros((vmax + 1, jmax + 1)), WF_vj=WF_vj)


def make_atom_diatom_basis() -> ChannelBasis:
    quantum_numbers = ((0, 0), (1, 0), (1, 1))
    channels = tuple(
        Channel(
            mis_X=MolInnerState(j=0),
            mis_Y=MolInnerState(v=0, j=j),
            j_couple=j,
            K=K,
            E_int=0.0,
        )
        for j, K in quantum_numbers
    )
    return ChannelBasis(channels, Jtot=1, system_parity=1)


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
            E_int=0.0,
        )
        for j_X, j_Y, j_couple, K in quantum_numbers
    )
    return ChannelBasis(channels, Jtot=2, system_parity=1)


def make_electric_monomer_basis() -> DiatomElectricBasis:
    blocks = []
    for m in (-1, 0, 1):
        j_values = np.arange(abs(m), 2, dtype=np.int64)
        energies = np.array([0.01]) if m else np.array([0.0, 0.02])
        coefficients = np.zeros((1, j_values.size, energies.size))
        for state in range(energies.size):
            coefficients[0, state, state] = 1.0
        blocks.append(DiatomElectricBlock(m=m, j_values=j_values, energies=energies, coefficients=coefficients))
    return DiatomElectricBasis(
        grids=np.array([1.75]),
        blocks=tuple(blocks),
        energy_zero=0.0,
        electric_strength=1.0e-3,
        jmax=1,
        mass=1000.0,
    )


def make_electric_sf_interaction_basis(
    delta: np.ndarray,
    delta_weights: np.ndarray,
) -> tuple[ChannelBasisElectricSF, atom_diatom.AtomDiatomVBasisElectricSF]:
    monomer = make_electric_monomer_basis()
    channels = build_ChannelBasisElectricSF(monomer, M=0, lmax=1)
    cos_theta_r, theta_r_weights = roots_legendre(5)
    cos_theta_R, theta_R_weights = roots_legendre(5)
    V_basis = atom_diatom.build_AtomDiatomVBasisElectricSF(
        channels,
        monomer,
        cos_theta_r,
        theta_r_weights,
        cos_theta_R,
        theta_R_weights,
        delta,
        delta_weights,
    )
    return channels, V_basis


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


def test_constant_potential_is_identity_for_electric_field_sf_basis() -> None:
    delta, delta_weights = gauss_legendre_dvr(0.0, 2.0 * np.pi, 24)
    channels, V_basis = make_electric_sf_interaction_basis(delta, delta_weights)

    Vmat = atom_diatom.contract_electric_sf(V_basis, np.full(V_basis.grid_shape, 2.5))

    assert V_basis.grid_shape == (1, 5, 5, 24)
    np.testing.assert_allclose(Vmat, 2.5 * np.eye(channels.n_channel), atol=2.0e-13)


def test_electric_sf_geometry_sampling_passes_r_and_gamma_to_pes() -> None:
    delta, delta_weights = gauss_legendre_dvr(0.0, 2.0 * np.pi, 12)
    _, V_basis = make_electric_sf_interaction_basis(delta, delta_weights)

    def interaction(R: float, coordinates: np.ndarray) -> np.ndarray:
        return R + 2.0 * coordinates[0] + np.cos(coordinates[1])

    radial_points = np.array([3.0, 4.0])
    potential = get_Vgrid_atom_diatom_electric_sf(PESWrapper(interaction=interaction), radial_points, V_basis.r, V_basis.gamma)

    expected = radial_points[:, None, None, None, None] + 2.0 * V_basis.r[None, :, None, None, None] + np.cos(V_basis.gamma)[None, None, :, :, :]
    assert potential.shape == (2, *V_basis.grid_shape)
    np.testing.assert_allclose(potential, expected)


def test_electric_sf_half_delta_rule_matches_full_interval_for_scalar_pes() -> None:
    full_delta, full_weights = gauss_legendre_dvr(0.0, 2.0 * np.pi, 24)
    half_delta, half_weights = gauss_legendre_dvr(0.0, 2.0 * np.pi, 12, symmetry=True)
    channels, full_basis = make_electric_sf_interaction_basis(full_delta, full_weights)
    _, half_basis = make_electric_sf_interaction_basis(half_delta, half_weights)
    full_potential = 1.0 + 0.25 * np.cos(full_basis.gamma) + 0.1 * np.cos(full_basis.gamma) ** 2
    half_potential = 1.0 + 0.25 * np.cos(half_basis.gamma) + 0.1 * np.cos(half_basis.gamma) ** 2

    full_matrix = atom_diatom.contract_electric_sf(full_basis, np.broadcast_to(full_potential, full_basis.grid_shape))
    half_matrix = atom_diatom.contract_electric_sf(half_basis, np.broadcast_to(half_potential, half_basis.grid_shape))

    assert full_matrix.shape == (channels.n_channel, channels.n_channel)
    np.testing.assert_allclose(half_matrix, full_matrix, atol=2.0e-13)


def test_contract_electric_sf_accepts_radial_batches_and_channel_order() -> None:
    delta, delta_weights = gauss_legendre_dvr(0.0, 2.0 * np.pi, 16)
    channels, V_basis = make_electric_sf_interaction_basis(delta, delta_weights)
    potential = np.stack(
        [
            np.ones(V_basis.grid_shape),
            np.broadcast_to(1.0 + 0.2 * np.cos(V_basis.gamma)[None, ...], V_basis.grid_shape),
        ]
    )
    indices = (channels.n_channel - 1, 0)

    full_matrix = atom_diatom.contract_electric_sf(V_basis, potential)
    selected_matrix = atom_diatom.contract_electric_sf(V_basis, potential, indices)

    assert full_matrix.shape == (2, channels.n_channel, channels.n_channel)
    np.testing.assert_allclose(selected_matrix, full_matrix[:, indices, :][:, :, indices])
