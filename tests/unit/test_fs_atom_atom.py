import numpy as np
import pytest

from pyticc.fine_structure.atom import build_fs_atom_basis
from pyticc.fine_structure.atom_atom import build_fs_atom_atom_channels
from pyticc.matrix.centrifugal import get_Umat_FS_AtomAtom_BF
from pyticc.matrix.interaction.fs_atom_atom import magnetic_dipole_matrix, prepare_fs_atom_atom_vbasis
from pyticc.pes.spin_resolved_atom_atom import SpinResolvedAtomAtomPES, atom_atom_orbital_states
from pyticc.pes.spin_resolved_diatom_diatom import allowed_total_spins


@pytest.mark.parametrize("sx,sy,two_J", [(2, 2, 4), (1, 2, 3), (1, 1, 4), (0, 0, 2)])
@pytest.mark.parametrize("parity", [-1, 1])
def test_two_structured_atoms_identity_and_centrifugal(sx, sy, two_J, parity):
    X = build_fs_atom_basis(2, sx, -1, tuple(range(abs(2 - sx), 2 + sx + 1, 2)), 0)
    Y = build_fs_atom_basis(2, sy, -1, tuple(range(abs(2 - sy), 2 + sy + 1, 2)), 0)
    basis = build_fs_atom_atom_channels(X, Y, two_J, parity)
    orbitals = atom_atom_orbital_states(2, 2)
    spins = allowed_total_spins(sx, sy)
    values = np.broadcast_to(np.eye(len(orbitals)), (len(spins), len(orbitals), len(orbitals))).copy()
    pes = SpinResolvedAtomAtomPES(lambda r: values, spins, orbitals)
    vbasis = prepare_fs_atom_atom_vbasis(basis, pes)
    np.testing.assert_allclose(vbasis.projectors.sum(axis=0), np.eye((sx + 1) * (sy + 1)), atol=1e-14)
    for Q in vbasis.projectors:
        np.testing.assert_allclose(Q @ Q, Q, atol=1e-14)
    np.testing.assert_allclose(vbasis.contract(pes.evaluate(10)), np.eye(len(basis)), atol=1e-14)
    U = get_Umat_FS_AtomAtom_BF(basis)
    for ladder in {c.ladder for c in basis}:
        indices = [i for i, c in enumerate(basis) if c.ladder == ladder]
        two_j = ladder[-1]
        expected_l = [ell for ell in range(abs(two_J - two_j) // 2, (two_J + two_j) // 2 + 1) if (-1) ** ell == parity]
        np.testing.assert_allclose(np.linalg.eigvalsh(U[np.ix_(indices, indices)]), [ell * (ell + 1) for ell in expected_l], atol=1e-13)
    selected = tuple(range(len(basis) - 1, -1, -2))
    np.testing.assert_allclose(get_Umat_FS_AtomAtom_BF(basis, selected), U[np.ix_(selected, selected)])
    diagonal = [basis.Jtot * (basis.Jtot + 1) + c.two_j12 / 2 * (c.two_j12 / 2 + 1) - 2 * c.K**2 for c in basis]
    np.testing.assert_allclose(get_Umat_FS_AtomAtom_BF(basis, coriolis=False), np.diag(diagonal))


def test_singlet_triplet_are_pes_not_projector_values():
    X = build_fs_atom_basis(0, 1, 1, (1,), 0)
    basis = build_fs_atom_atom_channels(X, X, 2, 1)
    pes = SpinResolvedAtomAtomPES(lambda r: np.array([[[2.0]], [[5.0]]]), (0, 2), atom_atom_orbital_states(0, 0))
    vbasis = prepare_fs_atom_atom_vbasis(basis, pes)
    expected = [2.0 if c.two_j12 == 0 else 5.0 for c in basis]
    np.testing.assert_allclose(vbasis.contract(pes.evaluate(10)), np.diag(expected), atol=1e-14)


@pytest.mark.parametrize("approximation", ["EXACT", "CS", "NNCC"])
def test_atomic_pair_propagates_and_matches(approximation):
    import pyticc as ticc

    X = build_fs_atom_basis(2, 1, -1, (1, 3), (0, 2))
    Y = build_fs_atom_basis(0, 1, 1, (1,), 0)
    orbitals = atom_atom_orbital_states(2, 0)
    pes = SpinResolvedAtomAtomPES(lambda r: np.stack([np.diag([1.0, 2.0, 1.0]), np.diag([2.0, 3.0, 2.0])]) * 1e-4 / r**3, (0, 2), orbitals)
    system = ticc.build_ScattSystem(X, Y, two_J=2, system_parity=-1, potential=pes, reduced_mass=1000.0, approx=getattr(ticc.Approx, approximation))
    grid = ticc.prepare_potential(system, (3.0, 4.0, 5.0), (0.25, 0.25))
    result = ticc.solve(system, np.asarray([0.001]), grid, ticc.Propagation())
    assert result.basis is system.basis
    assert "j_X" in ticc.report.channels(system.basis)
    assert "Re(S)" in ticc.report.smatrix(result, **({} if approximation == "EXACT" else {"block_index": 0}))
    matrices = result.Smat if approximation == "EXACT" else tuple(m for block in result.blocks for m in block.Smat_asymptotic)
    for matrix in matrices:
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(len(matrix)), atol=1e-11)


@pytest.mark.parametrize("parity", [-1, 1])
def test_spin_half_dipole(parity):
    X = build_fs_atom_basis(0, 1, 1, (1,), 0)
    basis = build_fs_atom_atom_channels(X, X, 2, parity)
    pes = SpinResolvedAtomAtomPES(lambda r: np.zeros((2, 1, 1)), (0, 2), atom_atom_orbital_states(0, 0))
    vbasis = prepare_fs_atom_atom_vbasis(basis, pes)
    expected = [0 if c.two_j12 == 0 else (1 if c.two_K == 0 else -0.5) for c in basis]
    np.testing.assert_allclose(magnetic_dipole_matrix(basis, vbasis), np.diag(expected), atol=1e-14)


def test_atomic_input_example():
    import tomllib
    from pathlib import Path

    from pyticc.input.driver import _resolve_scattering_type
    from pyticc.input.fine_structure_atom_atom import build_system
    from pyticc.system import ScatteringType

    path = Path(__file__).resolve().parents[2] / "example" / "CaHe_3P" / "input.toml"
    config = tomllib.loads(path.read_text())
    assert _resolve_scattering_type(config) is ScatteringType.ATOM_ATOM_FINE_STRUCTURE
    pes = SpinResolvedAtomAtomPES(lambda r: np.zeros((1, 3, 3)), (2,), atom_atom_orbital_states(2, 0))
    system = build_system(config, path.parent, pes)
    vbasis = prepare_fs_atom_atom_vbasis(system.basis, pes)
    np.testing.assert_array_equal(magnetic_dipole_matrix(system.basis, vbasis), np.zeros((system.n_channel, system.n_channel)))
    assert system.basis.monomer_X.atom_parity == -1


def test_orbital_offdiagonal_is_preserved_and_invalid_pes_rejected():
    X = build_fs_atom_basis(2, 0, -1, (2,), 0)
    orbitals = atom_atom_orbital_states(2, 2)
    values = np.zeros((1, 9, 9))
    values[0, 2, 4] = values[0, 4, 2] = values[0, 6, 4] = values[0, 4, 6] = 0.25
    pes = SpinResolvedAtomAtomPES(lambda r: values, (0,), orbitals)
    basis = build_fs_atom_atom_channels(X, X, 2, -1)
    vbasis = prepare_fs_atom_atom_vbasis(basis, pes)
    matrix = vbasis.contract(pes.evaluate([8.0, 9.0]))
    assert np.max(np.abs(matrix)) > 0.1
    np.testing.assert_allclose(matrix, matrix.swapaxes(-1, -2), atol=1e-14)
    values[0, 0, 1] = values[0, 1, 0] = 1.0
    with pytest.raises(ValueError, match="conserve"):
        pes.evaluate(8.0)
