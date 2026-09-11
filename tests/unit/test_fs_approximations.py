from dataclasses import replace

import jax
import numpy as np
import pytest

import pyticc as ticc
from pyticc import report
from pyticc.basis.kblock import build_cs_blocks, build_nncc_blocks
from pyticc.basis.podvr import VibPODVR
from pyticc.basis.rovib import RovibBasis
from pyticc.fine_structure import build_fs_channels, build_fs_monomer_basis
from pyticc.fine_structure.channel import FSChannelBasis
from pyticc.matrix.centrifugal import get_Umat_FS_BF
from pyticc.scattering.energy_transfer import fine_structure_atom_diatom, fine_structure_diatom_diatom
from pyticc.scattering.hamiltonian import ScattHamiltonian
from pyticc.scattering.solver import build_k_blocks


def _monomer(two_S: int, two_lambda: int = 2, max_two_j: int = 5) -> ticc.FSMonomerBasis:
    vib = VibPODVR(np.array([2.0]), np.array([0.0]), np.ones((1, 1)))
    return build_fs_monomer_basis(
        vib,
        tuple(range(two_S % 2, max_two_j + 1, 2)),
        two_lambda,
        two_S,
        ticc.FSConstants(A=0.002, B=0.001, gamma=0.00001, lambda_ss=0.00002, O=0.000003, P=0.000002, Q=0.000001),
    )


def _atom_system(two_S: int, two_J: int, parity: int = 1, two_lambda: int = 2) -> ticc.ScattSystem:
    def potential(R: float, coordinates: np.ndarray) -> np.ndarray:
        _, theta = coordinates
        radial = 0.001 * max(4.5 - R, 0.0) ** 2
        return radial * np.column_stack((1 + 0.3 * np.cos(theta), 0.1 * np.sin(theta) ** 2))

    return ticc.build_ScattSystem(
        ticc.AtomSpec(),
        _monomer(two_S, two_lambda),
        two_J=two_J,
        system_parity=parity,
        potential=ticc.LambdaPES(potential),
        reduced_mass=2.0,
    )


@pytest.mark.parametrize("two_S,two_lambda", [(0, 0), (1, 2), (2, 2), (3, 4)])
def test_fs_windows_own_each_integer_or_half_integer_channel_once(two_S: int, two_lambda: int) -> None:
    monomer = _monomer(two_S, two_lambda, max_two_j=10)
    two_J = 8 + two_S % 2
    basis = build_fs_channels(monomer, two_J, -1)
    offset = 0.5 if two_S % 2 else 0
    for blocks in (build_cs_blocks(basis), build_nncc_blocks(basis, 1), build_nncc_blocks(basis, 2)):
        assert sorted(i for block in blocks for i in block.owned_channel_indices) == list(range(len(basis)))
        for block in blocks:
            assert block.center_K % 1 == offset
            assert set(block.owned_channel_indices) <= set(block.channel_indices)
            assert all(basis[i].K in block.K_values for i in block.channel_indices)
    nncc = build_nncc_blocks(basis, 1)
    Kmin = min(c.K for c in basis)
    n_centers = int(max(c.K for c in basis) - Kmin) - 1
    assert [block.K_values for block in nncc] == [tuple(Kmin + k for k in range(start, start + 3)) for start in range(n_centers)]
    assert len(build_nncc_blocks(basis, 20)) == 1


@pytest.mark.parametrize("two_S,two_J,two_lambda", [(0, 4, 0), (1, 5, 2), (2, 4, 2), (3, 5, 4)])
@pytest.mark.parametrize("parity", [-1, 1])
def test_fs_nncc_full_window_restores_exact(two_S: int, two_J: int, two_lambda: int, parity: int) -> None:
    system = _atom_system(two_S, two_J, parity, two_lambda)
    grid = fine_structure_atom_diatom.prepare_potential(system, (3.0, 4.0, 5.0), (0.25, 0.25), n_theta=12)
    energies = [0.004, 0.04]
    exact = ticc.solve(system, energies, grid, ticc.Propagation())
    nncc = ticc.solve(replace(system, approx=ticc.Approx.NNCC, K_delta=2), energies, grid, ticc.Propagation())
    assert isinstance(exact, ticc.ScatteringResult)
    assert isinstance(nncc, ticc.CoupledStatesResult)
    assert len(nncc.blocks) == 1
    block = nncc.blocks[0]
    np.testing.assert_allclose(block.Y_BF, exact.Y_propagated, atol=2e-12)
    np.testing.assert_allclose(block.Bmat, exact.asymptotic_transform, atol=2e-13)
    np.testing.assert_allclose(block.L, exact.L, atol=2e-13)
    for energy_index, matrix in enumerate(exact.Smat):
        np.testing.assert_allclose(block.Smat_asymptotic[energy_index], matrix, atol=2e-12)
        opened = exact.open_channel_indices[energy_index]
        B = exact.asymptotic_transform[np.ix_(opened, opened)]
        np.testing.assert_allclose(block.Smat_BF[energy_index], B @ matrix @ B.T, atol=2e-12)
    assert report.smatrix(nncc) == report.smatrix(exact)


@pytest.mark.parametrize("parity", [-1, 1])
def test_strict_cs_removes_folded_half_integer_coriolis_and_matches_same_operator(parity: int) -> None:
    system = replace(_atom_system(1, 5, parity), approx=ticc.Approx.CS)
    grid = fine_structure_atom_diatom.prepare_potential(system, (3.0, 4.0), (0.25,), n_theta=12)
    hamiltonian = fine_structure_atom_diatom.build_hamiltonian(system, potential_grid=grid)
    assert isinstance(hamiltonian.basis, FSChannelBasis)
    basis = hamiltonian.basis
    J = basis.Jtot
    expected = np.array([J * (J + 1) + (j := basis.monomer.blocks[c.block].two_j / 2) * (j + 1) - 2 * c.K**2 for c in basis])
    np.testing.assert_allclose(hamiltonian.U, np.diag(expected), atol=1e-14)
    assert not np.allclose(np.diag(get_Umat_FS_BF(basis)), expected)
    result = ticc.solve(system, [0.004, 0.04], grid, ticc.Propagation())
    assert isinstance(result, ticc.CoupledStatesResult)
    for block in result.blocks:
        assert len(block.block.K_values) == 1
        indices = np.asarray(block.block.channel_indices)
        np.testing.assert_allclose(block.Bmat @ np.diag(block.L * (block.L + 1)) @ block.Bmat.T, np.diag(expected[indices]), atol=2e-13)
        for matrix in block.Smat_BF:
            np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(matrix.shape[0]), atol=2e-12)
    assert "tau" in report.smatrix(result, block_index=0)
    with pytest.raises(ValueError, match="block_index is required"):
        report.smatrix(result)


def test_fs_cpu_and_device_window_contractions_match_complete_matrix() -> None:
    system = _atom_system(1, 5)
    system = replace(system, approx=ticc.Approx.NNCC)
    grid = fine_structure_atom_diatom.prepare_potential(system, (3.0, 4.0), (0.25,), n_theta=12)
    hamiltonian = fine_structure_atom_diatom.build_hamiltonian(system, potential_grid=grid)
    assert isinstance(hamiltonian.basis, FSChannelBasis)
    blocks = tuple(block.channel_indices for block in build_cs_blocks(hamiltonian.basis))
    radial = np.array([3.0, 3.25, 3.5])
    full = hamiltonian.V(radial)
    assert hamiltonian.block_interaction is not None
    assert hamiltonian.device_block_interaction is not None
    for matrices in (hamiltonian.block_interaction(radial, blocks), hamiltonian.device_block_interaction(radial, blocks, jax.devices("cpu")[0])):
        for indices, matrix in zip(blocks, matrices, strict=True):
            np.testing.assert_allclose(matrix, full[:, indices, :][:, :, indices], atol=2e-13)


@pytest.mark.parametrize("approx", [ticc.Approx.CS, ticc.Approx.NNCC])
@pytest.mark.parametrize("spin_resolved", [False, True])
def test_two_fs_diatoms_use_shared_approximation_pipeline(approx: ticc.Approx, spin_resolved: bool) -> None:
    monomer_X = _monomer(1, 0, 3)
    monomer_Y = _monomer(2 if spin_resolved else 0, 0, 2)

    def scalar(R: float, c: np.ndarray) -> np.ndarray:
        return 0.001 * max(4.5 - R, 0) ** 2 * (1 + 0.2 * np.cos(c[2]) * np.cos(c[3]))

    potential = (
        ticc.SpinResolvedDiatomDiatomPES(
            lambda R, c: scalar(R, c)[:, None, None, None] * np.array([1.0, 1.3])[None, :, None, None], (1, 3), (ticc.OrbitalState(0, 0),)
        )
        if spin_resolved
        else ticc.PESWrapper(interaction=scalar)
    )
    system = ticc.build_ScattSystem(
        monomer_X,
        monomer_Y,
        two_J=3,
        system_parity=1,
        approx=approx,
        K_delta=2,
        potential=potential,
        magnetic_dipole_coefficient=1e-5 if spin_resolved else 0.0,
        reduced_mass=2.0,
    )
    grid = fine_structure_diatom_diatom.prepare_potential(system, (3.0, 4.0), (0.25,), n_theta_X=4, n_theta_Y=4, n_phi=4)
    ham = fine_structure_diatom_diatom.build_hamiltonian(system, potential_grid=grid)
    blocks = tuple(b.channel_indices for b in build_k_blocks(ham))
    radial = np.array([3.0, 3.25])
    assert ham.block_interaction is not None
    assert ham.device_block_interaction is not None
    if approx is ticc.Approx.CS:
        assert isinstance(ham.basis, ticc.FSDiatomDiatomBasis)
        J = ham.basis.Jtot
        expected = [J * (J + 1) + (j := c.two_j12 / 2) * (j + 1) - 2 * c.K**2 for c in ham.basis]
        np.testing.assert_allclose(ham.U, np.diag(expected), atol=1e-14)
    for matrices in (ham.block_interaction(radial, blocks), ham.device_block_interaction(radial, blocks, jax.devices("cpu")[0])):
        for indices, matrix in zip(blocks, matrices, strict=True):
            np.testing.assert_allclose(matrix, ham.V(radial)[:, indices, :][:, :, indices], atol=2e-13)
    result = ticc.solve(system, [0.04], grid, ticc.Propagation())
    assert isinstance(result, ticc.CoupledStatesResult)
    assert "tau_X" in report.smatrix(result, block_index=0)
    if approx is ticc.Approx.NNCC:
        exact = ticc.solve(replace(system, approx=ticc.Approx.EXACT), [0.04], grid, ticc.Propagation())
        assert isinstance(exact, ticc.ScatteringResult)
        np.testing.assert_allclose(result.blocks[0].Smat_asymptotic[0], exact.Smat[0], atol=2e-12)


def test_overlapping_fs_nncc_windows_equal_independent_projected_problems() -> None:
    template = _atom_system(1, 9)
    system = ticc.build_ScattSystem(
        ticc.AtomSpec(),
        _monomer(1, 2, 9),
        two_J=9,
        system_parity=1,
        potential=template.potential,
        reduced_mass=2.0,
        approx=ticc.Approx.NNCC,
        K_delta=1,
    )
    grid = fine_structure_atom_diatom.prepare_potential(system, (3.0, 4.0), (0.25,), n_theta=12)
    ham = fine_structure_atom_diatom.build_hamiltonian(system, potential_grid=grid)
    assert isinstance(ham.basis, FSChannelBasis)
    result = ticc.solve(system, [0.004, 0.06], grid, ticc.Propagation())
    assert isinstance(result, ticc.CoupledStatesResult)
    assert len(result.blocks) == 3
    for block in result.blocks:
        indices = block.block.channel_indices
        subset = replace(ham.basis, channels=tuple(ham.basis[i] for i in indices))

        def interaction(R: float | np.ndarray, indices: tuple[int, ...] = indices) -> np.ndarray:
            return ham.V(R)[..., np.asarray(indices)[:, None], np.asarray(indices)]

        reference = ticc.solve(ScattHamiltonian(subset, 2.0, interaction), result.Etot, grid.sectors, ticc.Propagation())
        assert isinstance(reference, ticc.ScatteringResult)
        np.testing.assert_allclose(block.Y_BF, reference.Y_propagated, atol=2e-12)
        for actual, expected in zip(block.Smat_asymptotic, reference.Smat, strict=True):
            np.testing.assert_allclose(actual, expected, atol=2e-12)


@pytest.mark.parametrize("approx", [ticc.Approx.CS, ticc.Approx.NNCC])
def test_singlet_sigma_approximations_reduce_to_spin_free_scattering(approx: ticc.Approx) -> None:
    def potential(R: float, coordinates: np.ndarray) -> np.ndarray:
        return 0.001 * max(4.5 - R, 0.0) ** 2 * (1 + 0.3 * np.cos(coordinates[1]))

    j_values = np.arange(5)
    energies = np.asarray(0.001 * (j_values * (j_values + 1))[None, :], dtype=np.float64)
    rotor = ticc.DiatomBasis(RovibBasis(np.array([2.0]), energies, np.ones((1, 1, 5))), energy_zero=0.0)
    monomer = _monomer(0, 0, 8)
    fs_system = ticc.build_ScattSystem(
        ticc.AtomSpec(),
        monomer,
        two_J=8,
        system_parity=1,
        approx=approx,
        potential=ticc.LambdaPES(lambda R, c: np.column_stack((potential(R, c), np.zeros(c.shape[1])))),
        reduced_mass=2.0,
    )
    spin_free_system = ticc.build_ScattSystem(
        ticc.AtomSpec(),
        rotor,
        Jtot=4,
        system_parity=1,
        approx=approx,
        potential=ticc.PESWrapper(interaction=potential),
        reduced_mass=2.0,
    )
    results = []
    for system in (fs_system, spin_free_system):
        grid = ticc.prepare_potential(system, (3.0, 4.0), (0.25,), n_theta=12)
        result = ticc.solve(system, [0.004, 0.04], grid, ticc.Propagation())
        assert isinstance(result, ticc.CoupledStatesResult)
        results.append(result)
    for fs_block, spin_free_block in zip(results[0].blocks, results[1].blocks, strict=True):
        assert fs_block.block.K_values == spin_free_block.block.K_values
        np.testing.assert_allclose(fs_block.Y_BF, spin_free_block.Y_BF, atol=2e-12)
        for fs_matrix, spin_free_matrix in zip(fs_block.Smat_BF, spin_free_block.Smat_BF, strict=True):
            np.testing.assert_allclose(fs_matrix, spin_free_matrix, atol=2e-12)
