import numpy as np

from pyticc.fine_structure import FSConstants, build_primitive_states, diagonalize_block, effective_hamiltonian, parity_transform
from pyticc.fine_structure.operators import molecular_rotation_element


def test_singlet_sigma_effective_hamiltonian_reduces_to_rigid_rotor() -> None:
    states = build_primitive_states((0,), (4,), two_lambda_abs=0, two_S=0)
    matrix = effective_hamiltonian(states, FSConstants(A=4.0, B=0.25, gamma=3.0, lambda_ss=2.0), vibrational_energy=1.5)

    np.testing.assert_allclose(matrix, [[3.0]])
    plus = parity_transform(states, parity=1, reflection_parity=1)
    minus = parity_transform(states, parity=-1, reflection_parity=1)
    assert sorted((plus.shape[1], minus.shape[1])) == [0, 1]


def test_singlet_sigma_centrifugal_distortion_reduces_to_scalar_polynomial() -> None:
    states = build_primitive_states((0,), (4,), two_lambda_abs=0, two_S=0)
    constants = FSConstants(B=0.25, D=0.01, H=0.001)

    matrix = effective_hamiltonian(states, constants)

    n_squared = 2.0 * 3.0
    expected = constants.B * n_squared - constants.D * n_squared**2 + constants.H * n_squared**3
    np.testing.assert_allclose(matrix, [[expected]])


def test_centrifugal_distortion_uses_full_n_squared_matrix() -> None:
    states = build_primitive_states((0,), (5,), two_lambda_abs=2, two_S=3)
    rigid = effective_hamiltonian(states, FSConstants(B=0.002))
    distorted = effective_hamiltonian(states, FSConstants(B=0.002, D=2.0e-6, H=3.0e-9))
    n_squared = np.asarray([[molecular_rotation_element(bra, ket, 1.0) for ket in states] for bra in states])
    expected = rigid - 2.0e-6 * np.linalg.matrix_power(n_squared, 2) + 3.0e-9 * np.linalg.matrix_power(n_squared, 3)

    np.testing.assert_allclose(distorted, expected, atol=1.0e-15)


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
