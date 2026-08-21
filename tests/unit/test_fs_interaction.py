import numpy as np

import pyticc as ticc
from pyticc.basis.podvr import VibPODVR
from pyticc.fine_structure import FSConstants, build_fs_channels, build_fs_monomer_basis
from pyticc.matrix.interaction import fs_atom_diatom as vmat
from pyticc.pes import LambdaPES
from pyticc.scattering import fine_structure_atom_diatom


def _pi_basis():
    vib = VibPODVR(np.array([2.0]), np.array([0.0]), np.ones((1, 1)))
    monomer = build_fs_monomer_basis(vib, (1, 3), 2, 1, FSConstants(A=0.01, B=0.001))
    return build_fs_channels(monomer, two_J=3, system_parity=1)


def test_constant_vsum_is_identity_and_vdif_is_hermitian() -> None:
    basis = _pi_basis()
    x, weights = np.polynomial.legendre.leggauss(24)
    V_basis = vmat.prepare(basis, np.arccos(x), weights)
    potential = np.empty((1, x.size, 2))
    potential[..., 0] = 2.5
    potential[..., 1] = 0.0
    sum_matrix = vmat.contract(V_basis, potential)

    np.testing.assert_allclose(sum_matrix, 2.5 * np.eye(basis.n_channel), atol=2.0e-13)
    potential[..., 0] = 0.0
    potential[..., 1] = 0.3 * (1.0 - x**2)
    matrix = vmat.contract(V_basis, potential)
    np.testing.assert_allclose(matrix, matrix.T, atol=2.0e-14)


def test_half_integer_fs_channels_propagate_and_match() -> None:
    basis = _pi_basis()
    potential = LambdaPES(lambda R, coordinates: np.zeros((coordinates.shape[1], 2)))
    system = ticc.build_ScattSystem(
        ticc.AtomSpec(),
        basis.monomer,
        two_J=basis.two_J,
        system_parity=basis.system_parity,
        potential=potential,
        reduced_mass=2.0,
    )
    hamiltonian = fine_structure_atom_diatom.build_hamiltonian(system, n_theta=16)
    energy = float(np.max(basis.E_int) + 0.1)

    result = ticc.solve(hamiltonian, [energy], ticc.Propagation((4.0, 4.2), (0.1,)))

    assert isinstance(result, ticc.ScatteringResult)
    assert result.basis.n_channel == basis.n_channel
    np.testing.assert_allclose(result.Smat[0].conj().T @ result.Smat[0], np.eye(basis.n_channel), atol=2.0e-12)
