from typing import cast

import jax
import numpy as np
import pytest

import pyticc as ticc
import pyticc.matrix.interaction.diatom_diatom as closed_shell_vmat
import pyticc.matrix.interaction.fs_diatom_diatom as fs_vmat
from pyticc.basis.angle import clebsch_gordan, clebsch_gordan_half, gauss_legendre_dvr
from pyticc.basis.channel import ChannelBasis
from pyticc.basis.monomer import DiatomBasis
from pyticc.basis.podvr import VibPODVR
from pyticc.basis.rovib import RovibBasis
from pyticc.fine_structure import build_fs_monomer_basis
from pyticc.match.asymptotic import get_Bmat_FS_DiatomDiatom_BF_to_SF
from pyticc.matrix.centrifugal import get_Umat_FS_DiatomDiatom_BF
from pyticc.matrix.interaction import contract as contract_scalar


def _monomer(two_j_values: tuple[int, ...], two_S: int) -> ticc.FSMonomerBasis:
    vib = VibPODVR(np.array([2.0]), np.array([0.0]), np.ones((1, 1)))
    return build_fs_monomer_basis(
        vib,
        two_j_values=two_j_values,
        two_lambda_abs=0,
        two_S=two_S,
        constants=ticc.FSConstants(),
    )


def test_channels_obey_j12_K_and_K0_parity_rules() -> None:
    monomer = _monomer((0, 2), two_S=0)
    basis = ticc.build_fs_diatom_diatom_channels(monomer, monomer, two_J=2, system_parity=1)

    assert basis.n_channel > 0
    for channel in basis:
        block_X = monomer.blocks[channel.block_X]
        block_Y = monomer.blocks[channel.block_Y]
        assert abs(block_X.two_j - block_Y.two_j) <= channel.two_j12 <= block_X.two_j + block_Y.two_j
        assert (channel.two_j12 - abs(block_X.two_j - block_Y.two_j)) % 2 == 0
        assert channel.two_j12 % 2 == basis.two_J % 2
        assert 0 <= channel.two_K <= min(basis.two_J, channel.two_j12)
        if channel.two_K == 0:
            exponent = (basis.two_J + channel.two_j12) // 2
            assert basis.system_parity * block_X.parity * block_Y.parity * (-1) ** exponent == 1

    j0_block = next(index for index, block in enumerate(monomer.blocks) if block.two_j == 0)
    assert not any(channel.block_X == j0_block and channel.block_Y == j0_block for channel in basis)


def test_centrifugal_matrix_is_hermitian_and_couples_only_one_K_ladder() -> None:
    monomer = _monomer((0, 2), two_S=0)
    basis = ticc.build_fs_diatom_diatom_channels(monomer, monomer, two_J=4, system_parity=-1)

    matrix = get_Umat_FS_DiatomDiatom_BF(basis)

    np.testing.assert_allclose(matrix, matrix.T, atol=0.0)
    for row, row_channel in enumerate(basis):
        for column, column_channel in enumerate(basis):
            if row == column or matrix[row, column] == 0.0:
                continue
            row_key = (
                row_channel.block_X,
                row_channel.tau_X,
                row_channel.block_Y,
                row_channel.tau_Y,
                row_channel.two_j12,
            )
            column_key = (
                column_channel.block_X,
                column_channel.tau_X,
                column_channel.block_Y,
                column_channel.tau_Y,
                column_channel.two_j12,
            )
            assert row_key == column_key
            assert abs(row_channel.two_K - column_channel.two_K) == 2


@pytest.mark.parametrize("system_parity", (-1, 1))
def test_half_integer_BF_to_SF_boundary_produces_integral_L(system_parity: int) -> None:
    half_integer = _monomer((1,), two_S=1)
    integer = _monomer((0,), two_S=0)
    basis = ticc.build_fs_diatom_diatom_channels(
        half_integer,
        integer,
        two_J=1,
        system_parity=system_parity,
    )

    transform, orbital_angular_momenta = get_Bmat_FS_DiatomDiatom_BF_to_SF(basis)

    np.testing.assert_allclose(transform.T @ transform, np.eye(basis.n_channel), atol=1.0e-14)
    np.testing.assert_allclose(orbital_angular_momenta, np.rint(orbital_angular_momenta), atol=1.0e-14)
    expected = []
    for channel in basis:
        epsilon_X = half_integer.blocks[channel.block_X].parity
        epsilon_Y = integer.blocks[channel.block_Y].parity
        expected.append(0 if system_parity * epsilon_X * epsilon_Y == 1 else 1)
    np.testing.assert_allclose(orbital_angular_momenta, expected, atol=1.0e-14)


def test_build_scatt_system_prepares_scalar_fs_diatom_diatom_basis() -> None:
    half_integer = _monomer((1,), two_S=1)
    integer = _monomer((0,), two_S=0)
    scalar_pes = ticc.PESWrapper(interaction=lambda R, coordinates: np.zeros(coordinates.shape[1]))

    system = ticc.build_ScattSystem(
        half_integer,
        integer,
        scattering_type="AB+CD_fine_structure",
        two_J=1,
        system_parity=1,
        potential=scalar_pes,
        reduced_mass=2.0,
    )

    assert system.scattering_type is ticc.ScatteringType.DIATOM_DIATOM_FINE_STRUCTURE
    assert isinstance(system.basis, ticc.FSDiatomDiatomBasis)
    assert system.n_channel == 2
    assert "j_12" in ticc.report.channels(system.basis)
    hamiltonian = ticc.ScattHamiltonian(
        basis=system.basis,
        reduced_mass=2.0,
        interaction=lambda R: np.zeros((system.n_channel, system.n_channel)),
    )
    np.testing.assert_allclose(hamiltonian.U, get_Umat_FS_DiatomDiatom_BF(system.basis))


def test_scalar_system_requires_a_potential_and_rejects_coupled_states() -> None:
    monomer = _monomer((0,), two_S=0)

    with pytest.raises(TypeError, match="scalar PESWrapper"):
        ticc.build_ScattSystem(
            monomer,
            monomer,
            scattering_type="AB+CD_fine_structure",
            two_J=0,
            system_parity=1,
        )

    with pytest.raises(ValueError, match="only exact coupled channels"):
        ticc.build_ScattSystem(
            monomer,
            monomer,
            scattering_type="AB+CD_fine_structure",
            two_J=0,
            system_parity=1,
            approx=ticc.Approx.CS,
            potential=ticc.PESWrapper(interaction=lambda R, coordinates: np.zeros(coordinates.shape[1])),
        )


def test_doubled_clebsch_gordan_reduces_to_integer_implementation() -> None:
    for j_X, m_X, j_Y, m_Y, j12 in ((1, -1, 1, 1, 0), (1, 0, 2, 1, 2), (2, -1, 2, 2, 3)):
        assert clebsch_gordan_half(2 * j_X, 2 * m_X, 2 * j_Y, 2 * m_Y, 2 * j12) == pytest.approx(clebsch_gordan(j_X, m_X, j_Y, m_Y, j12))


def test_closed_shell_limit_matches_existing_diatom_diatom_kernel() -> None:
    radial_grids = np.array([1.3, 1.8])
    radial_wavefunction = np.array([[0.6], [0.8]])
    vib = VibPODVR(radial_grids, np.array([0.0]), radial_wavefunction)
    fs_monomer = build_fs_monomer_basis(vib, (0, 2), 0, 0, ticc.FSConstants())
    rovib = RovibBasis(
        grids=radial_grids,
        E_vj=np.zeros((1, 2)),
        WF_vj=np.repeat(radial_wavefunction[:, :, None], 2, axis=2),
    )
    closed_shell_monomer = DiatomBasis(rovib=rovib, energy_zero=0.0)
    scalar_pes = ticc.PESWrapper(interaction=lambda R, coordinates: np.zeros(coordinates.shape[1]))
    closed_shell_system = ticc.build_ScattSystem(
        closed_shell_monomer,
        closed_shell_monomer,
        scattering_type="AB+CD",
        Jtot=1,
        system_parity=-1,
        potential=scalar_pes,
    )
    fs_system = ticc.build_ScattSystem(
        fs_monomer,
        fs_monomer,
        scattering_type="AB+CD_fine_structure",
        two_J=2,
        system_parity=-1,
        potential=scalar_pes,
    )
    assert isinstance(closed_shell_system.basis, ChannelBasis)
    assert isinstance(fs_system.basis, ticc.FSDiatomDiatomBasis)

    cos_theta_X, weight_X = gauss_legendre_dvr(-1.0, 1.0, 4)
    cos_theta_Y, weight_Y = gauss_legendre_dvr(-1.0, 1.0, 5)
    phi, weight_phi = gauss_legendre_dvr(0.0, np.pi, 6)
    old_basis = closed_shell_vmat.prepare(
        closed_shell_system.basis,
        rovib,
        rovib,
        cos_theta_X,
        weight_X,
        cos_theta_Y,
        weight_Y,
        phi,
        weight_phi,
    )
    new_basis = fs_vmat.prepare(
        fs_system.basis,
        np.arccos(cos_theta_X),
        weight_X,
        np.arccos(cos_theta_Y),
        weight_Y,
        phi,
        weight_phi,
    )
    rng = np.random.default_rng(20260901)
    potential = rng.normal(size=new_basis.grid_shape)
    old_matrix = contract_scalar(old_basis, potential)
    new_matrix = fs_vmat.contract(new_basis, potential)

    old_keys = [
        (
            cast(int, channel.mis_X.v),
            channel.mis_X.j,
            cast(int, channel.mis_Y.v),
            channel.mis_Y.j,
            channel.j_couple,
            channel.K,
        )
        for channel in closed_shell_system.basis
    ]
    new_indices = {
        (
            fs_system.basis.monomer_X.blocks[channel.block_X].v,
            fs_system.basis.monomer_X.blocks[channel.block_X].two_j // 2,
            fs_system.basis.monomer_Y.blocks[channel.block_Y].v,
            fs_system.basis.monomer_Y.blocks[channel.block_Y].two_j // 2,
            channel.two_j12 // 2,
            channel.two_K // 2,
        ): index
        for index, channel in enumerate(fs_system.basis)
    }
    permutation = np.asarray([new_indices[key] for key in old_keys], dtype=np.int64)
    np.testing.assert_allclose(new_matrix[np.ix_(permutation, permutation)], old_matrix, atol=2.0e-14)


def test_constant_scalar_potential_is_identity_for_open_shell_monomers() -> None:
    monomer_X = _monomer((1,), two_S=1)
    monomer_Y = _monomer((2,), two_S=2)
    basis = ticc.build_fs_diatom_diatom_channels(monomer_X, monomer_Y, two_J=3, system_parity=1)
    cos_theta_X, weight_X = gauss_legendre_dvr(-1.0, 1.0, 5)
    cos_theta_Y, weight_Y = gauss_legendre_dvr(-1.0, 1.0, 5)
    phi, weight_phi = gauss_legendre_dvr(0.0, np.pi, 8)
    V_basis = fs_vmat.prepare(
        basis,
        np.arccos(cos_theta_X),
        weight_X,
        np.arccos(cos_theta_Y),
        weight_Y,
        phi,
        weight_phi,
    )
    potential = np.full(V_basis.grid_shape, 2.5)

    host_matrix = fs_vmat.contract(V_basis, potential)
    device = jax.devices("cpu")[0]
    device_matrix = fs_vmat.contract_device(V_basis, fs_vmat.device_basis(V_basis, device), potential, device)

    np.testing.assert_allclose(host_matrix, 2.5 * np.eye(basis.n_channel), atol=3.0e-14)
    np.testing.assert_allclose(np.asarray(device_matrix), host_matrix, atol=3.0e-14)

    anisotropic = fs_vmat.contract(V_basis, np.random.default_rng(31).normal(size=V_basis.grid_shape))
    np.testing.assert_allclose(anisotropic, anisotropic.T, atol=0.0)
    for row, bra in enumerate(basis):
        for column, ket in enumerate(basis):
            if bra.two_K != ket.two_K:
                assert anisotropic[row, column] == 0.0


def test_scalar_fine_structure_diatom_diatom_runs_end_to_end() -> None:
    monomer = _monomer((0,), two_S=0)
    scalar_pes = ticc.PESWrapper(interaction=lambda R, coordinates: np.zeros(coordinates.shape[1]))
    system = ticc.build_ScattSystem(
        monomer,
        monomer,
        scattering_type="AB+CD_fine_structure",
        two_J=0,
        system_parity=1,
        potential=scalar_pes,
        reduced_mass=2.0,
    )
    potential_grid = ticc.prepare_potential(
        system,
        (3.0, 3.2),
        (0.1,),
        n_theta_X=2,
        n_theta_Y=2,
        n_phi=2,
    )
    result = ticc.solve(system, [0.1], potential_grid, ticc.Propagation())

    assert isinstance(result, ticc.ScatteringResult)
    assert isinstance(result.basis, ticc.FSDiatomDiatomBasis)
    np.testing.assert_allclose(result.Smat[0].conj().T @ result.Smat[0], np.eye(1), atol=1.0e-13)
