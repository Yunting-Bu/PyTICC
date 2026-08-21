import numpy as np

from pyticc.fine_structure import FSConstants, build_primitive_states, diagonalize_block, effective_hamiltonian, parity_transform


def test_singlet_sigma_effective_hamiltonian_reduces_to_rigid_rotor() -> None:
    states = build_primitive_states((0,), (4,), two_lambda_abs=0, two_S=0)
    matrix = effective_hamiltonian(states, FSConstants(A=4.0, B=0.25, gamma=3.0, lambda_ss=2.0), vibrational_energy=1.5)

    np.testing.assert_allclose(matrix, [[3.0]])
    plus = parity_transform(states, parity=1, reflection_parity=1)
    minus = parity_transform(states, parity=-1, reflection_parity=1)
    assert sorted((plus.shape[1], minus.shape[1])) == [0, 1]


def test_fs_effective_hamiltonian_is_inversion_symmetric_and_hermitian() -> None:
    states = build_primitive_states((0,), (3,), two_lambda_abs=2, two_S=1)
    constants = FSConstants(A=0.01, B=0.002, gamma=0.0001, lambda_ss=0.0002)
    matrix = effective_hamiltonian(states, constants)
    reverse = np.array([states.index(state.partner) for state in states])

    np.testing.assert_allclose(matrix, matrix.T, atol=1.0e-15)
    np.testing.assert_allclose(matrix, matrix[np.ix_(reverse, reverse)], atol=1.0e-15)

    plus = diagonalize_block(states, constants, parity=1)
    minus = diagonalize_block(states, constants, parity=-1)
    np.testing.assert_allclose(plus.energies, minus.energies, atol=1.0e-15)
    np.testing.assert_allclose(plus.transform.T @ plus.transform, np.eye(plus.transform.shape[1]), atol=1.0e-15)


def test_pi_lambda_doubling_splits_parity_blocks() -> None:
    states = build_primitive_states((0,), (3,), two_lambda_abs=2, two_S=1)
    constants = FSConstants(A=0.01, B=0.002, P=3.0e-5, Q=1.0e-5)

    plus = diagonalize_block(states, constants, parity=1)
    minus = diagonalize_block(states, constants, parity=-1)

    assert not np.allclose(plus.energies, minus.energies, atol=1.0e-12, rtol=0.0)
    matrix = effective_hamiltonian(states, constants)
    np.testing.assert_allclose(matrix, matrix.T, atol=1.0e-15)
