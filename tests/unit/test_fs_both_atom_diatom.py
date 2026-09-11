from dataclasses import replace

import jax
import numpy as np
import pytest

import pyticc as ticc
from pyticc.basis.kblock import build_nncc_blocks
from pyticc.basis.podvr import VibPODVR
from pyticc.constants import CM2AU, EV2AU
from pyticc.fine_structure import build_fs_monomer_basis
from pyticc.fine_structure.atom_diatom import build_fs_atom_diatom_channels
from pyticc.fine_structure.channel import build_fs_channels
from pyticc.matrix.interaction.fs_both_atom_diatom import _spin_projector_kernel
from pyticc.pes import allowed_total_spins
from pyticc.pes.spin_resolved_atom_diatom import (
    SpinResolvedAtomDiatomPES,
    as_spin_resolved_atom_diatom_pes,
    atom_diatom_orbital_states,
    get_spin_resolved_grid_atom_diatom,
)
from pyticc.scattering.energy_transfer import both_fs_atom_diatom, fine_structure_atom_diatom


def _monomer(two_S: int, two_lambda: int = 2, max_two_j: int = 5) -> ticc.FSMonomerBasis:
    vib = VibPODVR(np.array([2.0]), np.array([0.0]), np.ones((1, 1)))
    return build_fs_monomer_basis(
        vib,
        tuple(range(two_S % 2, max_two_j + 1, 2)),
        two_lambda,
        two_S,
        ticc.FSConstants(A=0.002, B=0.001, gamma=0.00001, lambda_ss=0.00002, O=0.000003, P=0.000002, Q=0.000001),
    )


def _atom(two_L: int, two_S: int, two_j_levels: tuple[int, ...], energy: float = 0.0) -> ticc.FSAtomBasis:
    parity = -1 if (two_L // 2) % 2 else 1
    return ticc.build_fs_atom_basis(two_L, two_S, parity, two_j_levels, energy, unit="au")


def _scalar_pes(scale_dif: float = 0.0) -> ticc.PESWrapper:
    def interaction(R: float, coordinates: np.ndarray) -> np.ndarray:
        _, theta = coordinates
        radial = 0.001 * max(4.5 - R, 0.0) ** 2
        return radial * (1 + 0.3 * np.cos(theta) + scale_dif * 0.1 * np.sin(theta) ** 2)

    return ticc.PESWrapper(interaction)


def _sum_dif_spin_resolved(two_L: int, two_lambda_Y_abs: int, two_spins: tuple[int, ...]) -> SpinResolvedAtomDiatomPES:
    """A sum/dif molecular operator, diagonal in the atomic orbital projections."""
    orbitals = atom_diatom_orbital_states(two_L, two_lambda_Y_abs)

    def interaction(R: float, coordinates: np.ndarray) -> np.ndarray:
        _, theta = coordinates
        radial = 0.001 * max(4.5 - R, 0.0) ** 2
        v_sum = radial * (1 + 0.3 * np.cos(theta))
        v_dif = radial * 0.1 * np.sin(theta) ** 2
        n_grid = v_sum.size
        values = np.zeros((n_grid, len(two_spins), len(orbitals), len(orbitals)))
        for i, orbital_bra in enumerate(orbitals):
            for j, orbital_ket in enumerate(orbitals):
                if orbital_bra == orbital_ket:
                    values[:, :, i, j] = v_sum[:, None]
                elif orbital_bra.two_lambda_atom == orbital_ket.two_lambda_atom and orbital_bra.two_lambda_Y == -orbital_ket.two_lambda_Y:
                    values[:, :, i, j] = v_dif[:, None]
        return values

    return SpinResolvedAtomDiatomPES(
        interaction=interaction,
        two_total_spins=two_spins,
        orbital_states=orbitals,
    )


def _both_fs_system(atom: ticc.FSAtomBasis, monomer: ticc.FSMonomerBasis, two_J: int, parity: int, pes) -> ticc.ScattSystem:
    return ticc.build_ScattSystem(
        atom,
        monomer,
        two_J=two_J,
        system_parity=parity,
        potential=pes,
        reduced_mass=2.0,
    )


# ----------------------------------------------------------------------------------------
def test_fs_atom_basis_units_and_validation() -> None:
    basis_cm = ticc.build_fs_atom_basis(2, 1, -1, (1, 3), (0.0, 100.0), unit="cm-1")
    basis_ev = ticc.build_fs_atom_basis(2, 1, -1, (1, 3), (0.0, 100.0 * CM2AU / EV2AU), unit="eV")
    basis_au = ticc.build_fs_atom_basis(2, 1, -1, (1, 3), (0.0, 100.0 * CM2AU), unit="au")
    np.testing.assert_allclose(basis_cm.level_energies, basis_au.level_energies)
    np.testing.assert_allclose(basis_ev.level_energies, basis_au.level_energies)
    assert basis_cm.energy_zero == basis_cm.level_energies.min()
    np.testing.assert_allclose(basis_cm.thresholds, (0.0, 100.0 * CM2AU))
    degenerate = ticc.build_fs_atom_basis(2, 1, -1, (1, 3), 40.0, unit="cm-1")
    assert np.allclose(degenerate.level_energies, 40.0 * CM2AU)
    with pytest.raises(ValueError, match="fine-structure level"):
        ticc.build_fs_atom_basis(2, 1, -1, (5,), 0.0)
    with pytest.raises(ValueError, match="even integer"):
        ticc.build_fs_atom_basis(3, 1, -1, (1,), 0.0)
    with pytest.raises(ValueError, match="atom_parity"):
        ticc.build_fs_atom_basis(2, 1, 0, (1,), 0.0)
    with pytest.raises(ValueError, match="level energies"):
        ticc.build_fs_atom_basis(2, 1, -1, (1, 3), (0.0,))


# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("two_S_atom,two_L,two_S_Y,two_lambda_Y,two_J", [(1, 2, 1, 2, 4), (1, 2, 0, 0, 5), (3, 0, 1, 2, 4), (1, 4, 3, 4, 4)])
def test_fs_atom_diatom_channels_match_combinatorics(two_S_atom: int, two_L: int, two_S_Y: int, two_lambda_Y: int, two_J: int) -> None:
    atom = _atom(two_L, two_S_atom, tuple(range(abs(two_L - two_S_atom), two_L + two_S_atom + 1, 2)))
    monomer = _monomer(two_S_Y, two_lambda_Y, max_two_j=9)
    parity = -1
    basis = build_fs_atom_diatom_channels(atom, monomer, two_J, parity)
    expected = set()
    for level, two_j_X in enumerate(atom.two_j_levels):
        for block in monomer.blocks:
            for tau in range(block.energies.size):
                for two_j12 in range(abs(two_j_X - block.two_j), two_j_X + block.two_j + 1, 2):
                    if two_j12 % 2 != two_J % 2:
                        continue
                    for two_K in range(two_J % 2, min(two_J, two_j12) + 1, 2):
                        if two_K == 0:
                            phase = parity * atom.atom_parity * block.parity * (-1) ** ((two_J + two_j12) // 2)
                            if phase != 1:
                                continue
                        expected.add((level, two_j_X, block.v, block.two_j, tau, block.parity, two_j12, two_K))
    actual = {
        (
            c.level_X,
            atom.two_j_levels[c.level_X],
            monomer.blocks[c.block_Y].v,
            monomer.blocks[c.block_Y].two_j,
            c.tau_Y,
            monomer.blocks[c.block_Y].parity,
            c.two_j12,
            c.two_K,
        )
        for c in basis
    }
    assert actual == expected
    thresholds = [
        float(atom.level_energies[c.level_X] - atom.energy_zero + monomer.blocks[c.block_Y].energies[c.tau_Y] - monomer.energy_zero) for c in basis
    ]
    np.testing.assert_allclose(basis.E_int, thresholds)
    assert list(basis.E_int) == sorted(basis.E_int)


# ----------------------------------------------------------------------------------------
def test_spin_projector_completeness() -> None:
    """Summing the spin kernel over all total spins gives the identity."""
    two_S_X, two_S_Y = 1, 1
    spins = (0, 2)
    theta = np.array([0.3, 1.2, 2.6])
    for two_sigma_bra in range(-two_S_Y, two_S_Y + 1, 2):
        for two_sigma_ket in range(-two_S_Y, two_S_Y + 1, 2):
            for two_mu_bra in range(-two_S_X, two_S_X + 1, 2):
                for two_mu_ket in range(-two_S_X, two_S_X + 1, 2):
                    total = sum(
                        _spin_projector_kernel(two_S_X, two_S_Y, spin, two_mu_bra, two_sigma_bra, two_mu_ket, two_sigma_ket, theta) for spin in spins
                    )
                    expected = np.ones_like(total) * float(two_mu_bra == two_mu_ket) * float(two_sigma_bra == two_sigma_ket)
                    np.testing.assert_allclose(total, expected, atol=1e-13)


# ----------------------------------------------------------------------------------------
def test_promoted_scalar_pes_axes() -> None:
    promotion = as_spin_resolved_atom_diatom_pes(_scalar_pes(), two_S_atom=0, two_L_atom=0, two_S_Y=1, two_lambda_Y_abs=2)
    assert promotion.two_total_spins == (1,)
    assert promotion.orbital_states == atom_diatom_orbital_states(0, 2)
    coordinates = np.asarray([[2.0, 2.0], [0.4, 1.2]])
    values = promotion.interaction(4.0, coordinates)
    assert values.shape == (2, 1, 2, 2)
    np.testing.assert_allclose(values[:, 0, 0, 0], values[:, 0, 1, 1])
    np.testing.assert_allclose(values[:, 0, 0, 1], 0.0)


# ----------------------------------------------------------------------------------------
def test_both_fs_reduces_to_structureless_atom_fine_structure() -> None:
    """An S-state atom must reproduce the existing A+BC fine-structure path."""
    monomer = _monomer(1, 2, max_two_j=5)
    two_J = 5
    parity = 1
    scalar_pes = _scalar_pes()
    scalar = ticc.build_ScattSystem(
        ticc.AtomSpec(),
        monomer,
        two_J=two_J,
        system_parity=parity,
        potential=ticc.as_lambda_pes(scalar_pes),
        reduced_mass=2.0,
    )
    atom = _atom(0, 0, (0,))
    both = _both_fs_system(
        atom, monomer, two_J, parity, as_spin_resolved_atom_diatom_pes(scalar_pes, two_S_atom=0, two_L_atom=0, two_S_Y=1, two_lambda_Y_abs=2)
    )

    old_basis = build_fs_channels(monomer, two_J, parity)
    assert both.n_channel == old_basis.n_channel
    np.testing.assert_allclose(both.basis.E_int, old_basis.E_int, atol=1e-14)

    grid_old = fine_structure_atom_diatom.prepare_potential(scalar, (3.0, 4.0, 5.0), (0.25, 0.25), n_theta=12)
    grid_new = both_fs_atom_diatom.prepare_potential(both, (3.0, 4.0, 5.0), (0.25, 0.25), n_theta=12)
    energies = [0.004, 0.04]
    result_old = ticc.solve(scalar, energies, grid_old, ticc.Propagation())
    result_new = ticc.solve(both, energies, grid_new, ticc.Propagation())
    assert isinstance(result_old, ticc.ScatteringResult)
    assert isinstance(result_new, ticc.ScatteringResult)
    np.testing.assert_allclose(result_new.Y_propagated, result_old.Y_propagated, atol=2e-12)
    np.testing.assert_allclose(result_new.asymptotic_transform, result_old.asymptotic_transform, atol=2e-13)
    np.testing.assert_allclose(result_new.L, result_old.L, atol=1e-13)
    for smat_new, smat_old in zip(result_new.Smat, result_old.Smat, strict=True):
        np.testing.assert_allclose(smat_new, smat_old, atol=1e-12)


# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("two_S_Y,two_lambda_Y,two_J,parity", [(1, 2, 4, 1), (2, 2, 5, -1)])
def test_both_fs_nncc_full_window_restores_exact(two_S_Y: int, two_lambda_Y: int, two_J: int, parity: int) -> None:
    atom = _atom(2, 1, (1, 3))
    monomer = _monomer(two_S_Y, two_lambda_Y, max_two_j=5)
    pes = _sum_dif_spin_resolved(2, two_lambda_Y, allowed_total_spins(atom.two_S, monomer.two_S))
    system = _both_fs_system(atom, monomer, two_J, parity, pes)
    grid = both_fs_atom_diatom.prepare_potential(system, (3.0, 4.0, 5.0), (0.25, 0.25), n_theta=12)
    energies = [0.004, 0.04]
    exact = ticc.solve(system, energies, grid, ticc.Propagation())
    nncc = ticc.solve(replace(system, approx=ticc.Approx.NNCC, K_delta=20), energies, grid, ticc.Propagation())
    assert isinstance(exact, ticc.ScatteringResult)
    assert isinstance(nncc, ticc.CoupledStatesResult)
    assert len(nncc.blocks) == 1
    block = nncc.blocks[0]
    np.testing.assert_allclose(block.Y_BF, exact.Y_propagated, atol=2e-12)
    np.testing.assert_allclose(block.Bmat, exact.asymptotic_transform, atol=2e-13)
    np.testing.assert_allclose(block.L, exact.L, atol=2e-13)
    for energy_index, matrix in enumerate(exact.Smat):
        np.testing.assert_allclose(block.Smat_asymptotic[energy_index], matrix, atol=1e-12)
    assert ticc.report.smatrix(nncc) == ticc.report.smatrix(exact)


# ----------------------------------------------------------------------------------------
def test_both_fs_cpu_and_device_window_contractions_match_complete_matrix() -> None:
    atom = _atom(2, 1, (1, 3))
    monomer = _monomer(1, 2, max_two_j=7)
    pes = _sum_dif_spin_resolved(2, 2, (0, 2))
    system = replace(_both_fs_system(atom, monomer, 4, -1, pes), approx=ticc.Approx.NNCC, K_delta=2)
    grid = both_fs_atom_diatom.prepare_potential(system, (3.0, 4.0), (0.25,), n_theta=12)
    hamiltonian = both_fs_atom_diatom.build_hamiltonian(system, potential_grid=grid)
    blocks = tuple(tuple(block.channel_indices) for block in build_nncc_blocks(hamiltonian.basis, 2))
    radial = np.array([3.0, 3.25, 3.5])
    full = np.asarray(hamiltonian.V(radial))
    assert hamiltonian.block_interaction is not None
    assert hamiltonian.device_block_interaction is not None
    host = hamiltonian.block_interaction(radial, blocks)
    device = hamiltonian.device_block_interaction(radial, blocks, jax.devices("cpu")[0])
    for indices, host_matrix, device_matrix in zip(blocks, host, device, strict=True):
        np.testing.assert_allclose(np.asarray(host_matrix), full[:, indices, :][:, :, indices], atol=2e-13)
        np.testing.assert_allclose(np.asarray(device_matrix), full[:, indices, :][:, :, indices], atol=2e-13)


# ----------------------------------------------------------------------------------------
def test_both_fs_interaction_selection_rules() -> None:
    atom = _atom(2, 1, (1, 3))
    monomer = _monomer(1, 2, max_two_j=7)
    pes = _sum_dif_spin_resolved(2, 2, (0, 2))
    system = _both_fs_system(atom, monomer, 4, -1, pes)
    theta = np.arccos(np.polynomial.legendre.leggauss(14)[0])
    hamiltonian = both_fs_atom_diatom.build_hamiltonian(system, n_theta=14)
    matrix = np.asarray(hamiltonian.V(4.0))
    assert matrix.shape == (system.n_channel, system.n_channel)
    two_K = np.asarray([channel.two_K for channel in system.basis])
    mask = two_K[:, None] != two_K[None, :]
    assert not np.any(mask & (np.abs(matrix) > 1.0e-14))
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-12)
    assert np.all(np.isfinite(matrix))

    values = get_spin_resolved_grid_atom_diatom(pes, 4.0, monomer.vib.grids, theta)
    assert values.shape == (monomer.vib.grids.size, theta.size, 2, 6, 6)
    np.testing.assert_allclose(values, np.swapaxes(np.conj(values), -1, -2), atol=1e-14)
