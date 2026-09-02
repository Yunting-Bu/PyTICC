import jax
import numpy as np
import pytest

import pyticc as ticc
import pyticc.matrix.interaction.fs_diatom_diatom as scalar_vmat
import pyticc.matrix.interaction.fs_diatom_diatom_spin as spin_vmat
from pyticc.basis.angle import gauss_legendre_dvr
from pyticc.basis.podvr import VibPODVR
from pyticc.fine_structure import build_fs_monomer_basis
from pyticc.match.asymptotic import get_Bmat_FS_DiatomDiatom_BF_to_SF
from pyticc.pes import allowed_total_spins, get_spin_resolved_grid_diatom_diatom, orbital_product_states
from pyticc.scattering.energy_transfer import fine_structure_diatom_diatom as fsdd_scattering


def _monomer(two_j_values: tuple[int, ...], two_lambda_abs: int, two_S: int) -> ticc.FSMonomerBasis:
    vib = VibPODVR(np.array([2.0]), np.array([0.0]), np.ones((1, 1)))
    return build_fs_monomer_basis(
        vib,
        two_j_values=two_j_values,
        two_lambda_abs=two_lambda_abs,
        two_S=two_S,
        constants=ticc.FSConstants(),
    )


def _quadrature() -> tuple[np.ndarray, ...]:
    cos_X, weight_X = gauss_legendre_dvr(-1.0, 1.0, 4)
    cos_Y, weight_Y = gauss_legendre_dvr(-1.0, 1.0, 4)
    phi, weight_phi = gauss_legendre_dvr(0.0, np.pi, 6)
    return np.arccos(cos_X), weight_X, np.arccos(cos_Y), weight_Y, phi, weight_phi


def test_spin_and_orbital_metadata_helpers() -> None:
    assert allowed_total_spins(1, 1) == (0, 2)
    assert allowed_total_spins(1, 2) == (1, 3)
    assert orbital_product_states(2, 0) == (ticc.OrbitalState(-2, 0), ticc.OrbitalState(2, 0))


def test_spin_resolved_pes_grid_validates_dense_hermitian_contract() -> None:
    orbitals = (ticc.OrbitalState(0, 0), ticc.OrbitalState(2, 0))

    def interaction(R: float, coordinates: np.ndarray) -> np.ndarray:
        values = np.zeros((coordinates.shape[1], 2, 2, 2))
        values[:, 0] = np.array([[R, 0.25], [0.25, 2.0 * R]])
        values[:, 1] = np.array([[3.0 * R, -0.5], [-0.5, 4.0 * R]])
        return values

    pes = ticc.SpinResolvedDiatomDiatomPES(interaction, (0, 2), orbitals)
    grid = get_spin_resolved_grid_diatom_diatom(
        pes,
        np.array([3.0, 4.0]),
        np.array([1.0]),
        np.array([1.5]),
        np.array([0.2, 1.0]),
        np.array([0.4]),
        np.array([0.1, 0.8]),
    )
    assert grid.shape == (2, 1, 1, 2, 1, 2, 2, 2, 2)
    np.testing.assert_allclose(grid[0, 0, 0, 0, 0, 0, 0], np.array([[3.0, 0.25], [0.25, 6.0]]))

    bad = ticc.SpinResolvedDiatomDiatomPES(
        lambda R, coordinates: np.broadcast_to(np.array([[[[0.0, 1.0], [2.0, 0.0]]]]), (coordinates.shape[1], 1, 2, 2)),
        (0,),
        orbitals,
    )
    with pytest.raises(ValueError, match="Hermitian"):
        get_spin_resolved_grid_diatom_diatom(
            bad,
            3.0,
            np.array([1.0]),
            np.array([1.5]),
            np.array([0.2]),
            np.array([0.4]),
            np.array([0.1]),
        )

    def complex_interaction(R: float, coordinates: np.ndarray) -> np.ndarray:
        values = np.empty((coordinates.shape[1], 1, 2, 2), dtype=np.complex128)
        values[:, 0] = np.array([[R, 0.25 + 0.5j], [0.25 - 0.5j, 2.0 * R]])
        return values

    complex_pes = ticc.SpinResolvedDiatomDiatomPES(complex_interaction, (0,), orbitals)
    complex_grid = get_spin_resolved_grid_diatom_diatom(
        complex_pes,
        3.0,
        np.array([1.0]),
        np.array([1.5]),
        np.array([0.2]),
        np.array([0.4]),
        np.array([0.1]),
    )
    np.testing.assert_allclose(complex_grid[0, 0, 0, 0, 0, 0], np.array([[3.0, 0.25 + 0.5j], [0.25 - 0.5j, 6.0]]))


def test_projector_completeness_exactly_reproduces_scalar_anisotropic_kernel() -> None:
    monomer = _monomer((1,), two_lambda_abs=0, two_S=1)
    basis = ticc.build_fs_diatom_diatom_channels(monomer, monomer, two_J=2, system_parity=1)
    theta_X, weight_X, theta_Y, weight_Y, phi, weight_phi = _quadrature()
    scalar_basis = scalar_vmat.prepare(basis, theta_X, weight_X, theta_Y, weight_Y, phi, weight_phi)
    spin_basis = spin_vmat.prepare(
        basis,
        (0, 2),
        (ticc.OrbitalState(0, 0),),
        theta_X,
        weight_X,
        theta_Y,
        weight_Y,
        phi,
        weight_phi,
    )
    scalar_grid = np.random.default_rng(20260901).normal(size=scalar_basis.grid_shape)
    spin_grid = np.repeat(scalar_grid[..., None, None, None], 2, axis=-3)

    expected = scalar_vmat.contract(scalar_basis, scalar_grid)
    actual = spin_vmat.contract(spin_basis, spin_grid)
    np.testing.assert_allclose(actual, expected, atol=1.0e-14)

    device = jax.devices("cpu")[0]
    device_matrix = spin_vmat.contract_device(
        spin_basis,
        spin_vmat.device_basis(spin_basis, device),
        spin_grid,
        device,
    )
    np.testing.assert_allclose(np.asarray(device_matrix), expected, atol=1.0e-14)


def test_distinct_singlet_triplet_surfaces_change_the_channel_interaction() -> None:
    monomer = _monomer((1,), two_lambda_abs=0, two_S=1)
    basis = ticc.build_fs_diatom_diatom_channels(monomer, monomer, two_J=2, system_parity=1)
    theta_X, weight_X, theta_Y, weight_Y, phi, weight_phi = _quadrature()
    spin_basis = spin_vmat.prepare(
        basis,
        (0, 2),
        (ticc.OrbitalState(0, 0),),
        theta_X,
        weight_X,
        theta_Y,
        weight_Y,
        phi,
        weight_phi,
    )
    grid = np.empty((*spin_basis.grid_shape, 2, 1, 1))
    grid[..., 0, 0, 0] = 1.0
    grid[..., 1, 0, 0] = 3.0
    matrix = spin_vmat.contract(spin_basis, grid)

    np.testing.assert_allclose(matrix, matrix.conj().T, atol=0.0)
    assert not np.allclose(matrix, 2.0 * np.eye(basis.n_channel))
    for row, bra in enumerate(basis):
        for column, ket in enumerate(basis):
            if bra.two_K != ket.two_K:
                assert matrix[row, column] == 0.0


def test_direct_spin_dipole_kernel_reproduces_bf_spin_operator() -> None:
    theta = np.array([0.0])
    phi = np.array([0.0])
    diagonal = spin_vmat._spin_dipole_kernel(1, 1, 1, 1, 1, 1, theta, theta, phi)
    spin_exchange = spin_vmat._spin_dipole_kernel(1, 1, 1, -1, -1, 1, theta, theta, phi)

    np.testing.assert_allclose(diagonal, -0.5, atol=1.0e-15)
    np.testing.assert_allclose(spin_exchange, 0.5, atol=1.0e-15)


def test_magnetic_dipole_matrix_is_real_symmetric_and_vanishes_for_zero_spin() -> None:
    doublet = _monomer((1,), two_lambda_abs=0, two_S=1)
    singlet = _monomer((0,), two_lambda_abs=0, two_S=0)
    theta_X, weight_X, theta_Y, weight_Y, phi, weight_phi = _quadrature()

    doublet_basis = ticc.build_fs_diatom_diatom_channels(doublet, doublet, two_J=2, system_parity=1)
    matrix = spin_vmat.magnetic_dipole_matrix(
        doublet_basis,
        theta_X,
        weight_X,
        theta_Y,
        weight_Y,
        phi,
        weight_phi,
    )
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=0.0)
    assert np.max(np.abs(matrix)) > 0.1
    for row, bra in enumerate(doublet_basis):
        for column, ket in enumerate(doublet_basis):
            if bra.two_K != ket.two_K:
                assert matrix[row, column] == 0.0

    mixed_basis = ticc.build_fs_diatom_diatom_channels(doublet, singlet, two_J=1, system_parity=1)
    zero_matrix = spin_vmat.magnetic_dipole_matrix(
        mixed_basis,
        theta_X,
        weight_X,
        theta_Y,
        weight_Y,
        phi,
        weight_phi,
    )
    np.testing.assert_allclose(zero_matrix, 0.0, atol=0.0)


def test_magnetic_dipole_coefficient_adds_inverse_cube_hamiltonian_term() -> None:
    monomer = _monomer((1,), two_lambda_abs=0, two_S=1)
    zero_pes = ticc.PESWrapper(interaction=lambda R, coordinates: np.zeros(coordinates.shape[1]))
    coefficient = 2.5e-5
    system = ticc.build_ScattSystem(
        monomer,
        monomer,
        scattering_type="AB+CD_fine_structure",
        two_J=2,
        system_parity=1,
        potential=zero_pes,
        reduced_mass=2.0,
        magnetic_dipole_coefficient=coefficient,
    )
    hamiltonian = fsdd_scattering.build_hamiltonian(system, n_theta_X=4, n_theta_Y=4, n_phi=6)
    matrix_at_4 = hamiltonian.interaction(4.0)
    matrix_at_8 = hamiltonian.interaction(8.0)

    assert np.max(np.abs(matrix_at_4)) > 0.0
    np.testing.assert_allclose(matrix_at_4, 8.0 * matrix_at_8, atol=1.0e-18)


def test_orbital_off_diagonal_surface_couples_signed_lambda_components() -> None:
    pi_doublet = _monomer((3,), two_lambda_abs=2, two_S=1)
    sigma_singlet = _monomer((0,), two_lambda_abs=0, two_S=0)
    basis = ticc.build_fs_diatom_diatom_channels(
        pi_doublet,
        sigma_singlet,
        two_J=3,
        system_parity=1,
    )
    theta_X, weight_X, theta_Y, weight_Y, phi, weight_phi = _quadrature()
    spin_basis = spin_vmat.prepare(
        basis,
        (1,),
        orbital_product_states(2, 0),
        theta_X,
        weight_X,
        theta_Y,
        weight_Y,
        phi,
        weight_phi,
    )
    grid = np.zeros((*spin_basis.grid_shape, 1, 2, 2))
    grid[..., 0, 0, 1] = 1.0
    grid[..., 0, 1, 0] = 1.0
    matrix = spin_vmat.contract(spin_basis, grid)

    np.testing.assert_allclose(matrix, matrix.conj().T, atol=0.0)
    assert np.max(np.abs(matrix)) > 0.5
    np.testing.assert_allclose(np.diag(matrix), 0.0, atol=2.0e-16)

    complex_grid = np.zeros((*spin_basis.grid_shape, 1, 2, 2), dtype=np.complex128)
    complex_grid[..., 0, 0, 1] = 1.0j
    complex_grid[..., 0, 1, 0] = -1.0j
    complex_matrix = spin_vmat.contract(spin_basis, complex_grid)
    np.testing.assert_allclose(complex_matrix, complex_matrix.conj().T, atol=0.0)
    assert np.max(np.abs(complex_matrix)) > 0.5

    device = jax.devices("cpu")[0]
    device_matrix = spin_vmat.contract_device(
        spin_basis,
        spin_vmat.device_basis(spin_basis, device),
        complex_grid,
        device,
    )
    np.testing.assert_allclose(np.asarray(device_matrix), complex_matrix, atol=1.0e-14)


def test_oh_h2_single_expansion_terms_match_hibridon_basis_20() -> None:
    vib = VibPODVR(np.array([1.0]), np.array([0.0]), np.ones((1, 1)))
    oh = build_fs_monomer_basis(
        vib,
        (1, 3),
        two_lambda_abs=2,
        two_S=1,
        constants=ticc.FSConstants.from_unit("cm-1", A=-139.21, B=18.548),
    )
    h2 = build_fs_monomer_basis(
        vib,
        (0,),
        two_lambda_abs=0,
        two_S=0,
        constants=ticc.FSConstants.from_unit("cm-1", B=59.322),
    )
    basis = ticc.build_fs_diatom_diatom_channels(oh, h2, two_J=1, system_parity=1)
    transform, orbital_angular_momenta = get_Bmat_FS_DiatomDiatom_BF_to_SF(basis)
    theta_X, weight_X, theta_Y, weight_Y, phi, weight_phi = _quadrature()
    V_basis = spin_vmat.prepare(
        basis,
        (1,),
        orbital_product_states(2, 0),
        theta_X,
        weight_X,
        theta_Y,
        weight_Y,
        phi,
        weight_phi,
    )

    metadata = []
    for index, channel in enumerate(basis):
        block = oh.blocks[channel.block_X]
        epsilon_hibridon = block.parity * (-1) ** ((block.two_j - 1) // 2)
        metadata.append(
            (
                block.two_j,
                channel.tau_X,
                epsilon_hibridon,
                channel.two_j12,
                int(round(orbital_angular_momenta[index])),
            )
        )
    hibridon_order = (
        (3, 0, -1, 3, 2),
        (3, 0, 1, 3, 1),
        (1, 0, -1, 1, 1),
        (1, 0, 1, 1, 0),
        (3, 1, -1, 3, 2),
        (3, 1, 1, 3, 1),
    )
    permutation = np.asarray([metadata.index(label) for label in hibridon_order])
    phases = np.asarray((-1.0, -1.0, 1.0, 1.0, -1.0, -1.0))

    def hibridon_frame(values: np.ndarray) -> np.ndarray:
        matrix = transform.T @ spin_vmat.contract(V_basis, values) @ transform
        matrix = matrix[np.ix_(permutation, permutation)]
        return phases[:, None] * matrix * phases[None, :]

    diagonal_grid = np.zeros((*V_basis.grid_shape, 1, 2, 2))
    cos_theta_X = np.cos(theta_X)[None, None, :, None, None]
    diagonal_grid[..., 0, 0, 0] = cos_theta_X
    diagonal_grid[..., 0, 1, 1] = cos_theta_X
    expected_diagonal = np.zeros((6, 6))
    for row, column, value in (
        (1, 0, 0.1959701374),
        (2, 0, 0.0819538749),
        (5, 0, 0.0228270720),
        (3, 1, 0.0819538749),
        (4, 1, 0.0228270720),
        (3, 2, 0.3333333333),
        (4, 2, -0.4642260060),
        (5, 3, -0.4642260060),
        (5, 4, 0.0706965292),
    ):
        expected_diagonal[row, column] = expected_diagonal[column, row] = value
    np.testing.assert_allclose(hibridon_frame(diagonal_grid), expected_diagonal, atol=6.0e-11)

    off_diagonal_grid = np.zeros_like(diagonal_grid)
    d_02 = np.sqrt(3.0 / 8.0) * np.sin(theta_X)[None, None, :, None, None] ** 2
    off_diagonal_grid[..., 0, 0, 1] = d_02
    off_diagonal_grid[..., 0, 1, 0] = d_02
    expected_off_diagonal = np.zeros((6, 6))
    for row, column, value in (
        (0, 0, 0.0968470644),
        (3, 0, -0.3939088282),
        (4, 0, -0.2657454536),
        (1, 1, -0.0968470644),
        (2, 1, 0.3939088282),
        (5, 1, 0.2657454536),
        (5, 2, 0.0695401688),
        (4, 3, -0.0695401688),
        (4, 4, -0.0968470644),
        (5, 5, 0.0968470644),
    ):
        expected_off_diagonal[row, column] = expected_off_diagonal[column, row] = value
    np.testing.assert_allclose(hibridon_frame(off_diagonal_grid), expected_off_diagonal, atol=6.0e-11)


def test_spin_resolved_system_runs_through_cached_grid_and_solver() -> None:
    monomer = _monomer((1,), two_lambda_abs=0, two_S=1)

    def interaction(R: float, coordinates: np.ndarray) -> np.ndarray:
        values = np.zeros((coordinates.shape[1], 2, 1, 1))
        values[:, 0, 0, 0] = 0.001 / R**6
        values[:, 1, 0, 0] = 0.002 / R**6
        return values

    pes = ticc.SpinResolvedDiatomDiatomPES(
        interaction,
        two_total_spins=(0, 2),
        orbital_states=(ticc.OrbitalState(0, 0),),
    )
    system = ticc.build_ScattSystem(
        monomer,
        monomer,
        scattering_type="AB+CD_fine_structure",
        two_J=0,
        system_parity=1,
        potential=pes,
        reduced_mass=2.0,
    )
    potential_grid = ticc.prepare_potential(
        system,
        (3.0, 3.2),
        (0.1,),
        n_theta_X=3,
        n_theta_Y=3,
        n_phi=4,
    )
    result = ticc.solve(system, [0.1], potential_grid, ticc.Propagation())

    assert np.asarray(potential_grid.values).shape[-3:] == (2, 1, 1)
    assert isinstance(result, ticc.ScatteringResult)
    np.testing.assert_allclose(
        result.Smat[0].conj().T @ result.Smat[0],
        np.eye(system.n_channel),
        atol=2.0e-12,
    )


def test_system_rejects_incomplete_spin_or_orbital_metadata() -> None:
    monomer = _monomer((1,), two_lambda_abs=0, two_S=1)
    incomplete_spin = ticc.SpinResolvedDiatomDiatomPES(
        lambda R, coordinates: np.zeros((coordinates.shape[1], 1, 1, 1)),
        (0,),
        (ticc.OrbitalState(0, 0),),
    )
    with pytest.raises(ValueError, match="total spins"):
        ticc.build_ScattSystem(
            monomer,
            monomer,
            scattering_type="AB+CD_fine_structure",
            two_J=0,
            system_parity=1,
            potential=incomplete_spin,
        )

    wrong_orbital = ticc.SpinResolvedDiatomDiatomPES(
        lambda R, coordinates: np.zeros((coordinates.shape[1], 2, 1, 1)),
        (0, 2),
        (ticc.OrbitalState(2, 0),),
    )
    with pytest.raises(ValueError, match="orbital_states"):
        ticc.build_ScattSystem(
            monomer,
            monomer,
            scattering_type="AB+CD_fine_structure",
            two_J=0,
            system_parity=1,
            potential=wrong_orbital,
        )
