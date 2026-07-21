import numpy as np

from pyticc.match import get_Smat, modified_bessel_IK_logD, riccati_bessel_jy
from pyticc.propagation import propagate_logD


def _reference_matrices(
    energy: float,
    Rmatch: float,
    reduced_mass: float,
    E_int: np.ndarray,
    L: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    J = np.ones(E_int.size)
    N = np.ones(E_int.size)
    J_prime = np.empty(E_int.size)
    N_prime = np.empty(E_int.size)

    for index, threshold in enumerate(E_int):
        wave_number = np.sqrt(2.0 * reduced_mass * abs(energy - threshold))
        argument = wave_number * Rmatch
        if energy > threshold:
            j_value, n_value, j_derivative, n_derivative = riccati_bessel_jy(float(L[index]), argument)
            J[index] = j_value / np.sqrt(wave_number)
            N[index] = n_value / np.sqrt(wave_number)
            J_prime[index] = np.sqrt(wave_number) * j_derivative
            N_prime[index] = np.sqrt(wave_number) * n_derivative
        else:
            I_logD, K_logD = modified_bessel_IK_logD(float(L[index] + 0.5), argument)
            J_prime[index] = 0.5 / Rmatch + wave_number * I_logD
            N_prime[index] = 0.5 / Rmatch + wave_number * K_logD

    return np.diag(J), np.diag(N), np.diag(J_prime), np.diag(N_prime)


def _logD_from_reaction_matrix(J: np.ndarray, N: np.ndarray, J_prime: np.ndarray, N_prime: np.ndarray, Kmat: np.ndarray) -> np.ndarray:
    wavefunction = J + N @ Kmat
    derivative = J_prime + N_prime @ Kmat
    return derivative @ np.linalg.inv(wavefunction)


def test_get_Smat_free_single_channel_returns_identity() -> None:
    energy = 0.4
    Rmatch = 6.0
    reduced_mass = 2.0
    E_int = np.array([0.0])
    L = np.array([0.0])
    J, N, J_prime, N_prime = _reference_matrices(energy, Rmatch, reduced_mass, E_int, L)
    Ymat = _logD_from_reaction_matrix(J, N, J_prime, N_prime, np.zeros((1, 1)))

    (Smat,) = get_Smat(Ymat[None, :, :], Rmatch, [energy], reduced_mass, E_int, L)

    np.testing.assert_allclose(Smat, np.eye(1), rtol=1.0e-13, atol=1.0e-13)


def test_free_single_channel_propagation_and_matching_returns_identity() -> None:
    energy = 0.2
    reduced_mass = 2.0
    wave_number = np.sqrt(2.0 * reduced_mass * energy)
    R_start = 1.1
    Rmatch = 2.1
    half_steps = np.full(5, 0.1)
    W_base = np.zeros((5, 1, 1))
    Y_initial = np.array([[[wave_number / np.tan(wave_number * R_start)]]])

    Y_final = propagate_logD(
        Y_initial,
        np.array([energy]),
        reduced_mass,
        half_steps,
        W_base,
        W_base,
        W_base,
    )
    (Smat,) = get_Smat(Y_final, Rmatch, [energy], reduced_mass, [0.0], [0.0])

    np.testing.assert_allclose(Smat, np.eye(1), rtol=1.0e-12, atol=1.0e-12)


def test_get_Smat_real_open_channel_reaction_matrix_is_unitary() -> None:
    energy = 0.5
    Rmatch = 7.0
    reduced_mass = 2.0
    E_int = np.array([0.0, 0.1])
    L = np.array([0.0, 1.3])
    reaction_matrix = np.array([[0.2, 0.07], [0.07, -0.1]])
    J, N, J_prime, N_prime = _reference_matrices(energy, Rmatch, reduced_mass, E_int, L)
    Ymat = _logD_from_reaction_matrix(J, N, J_prime, N_prime, reaction_matrix)

    (Smat,) = get_Smat(Ymat[None, :, :], Rmatch, [energy], reduced_mass, E_int, L)
    identity = np.eye(2)
    expected = np.linalg.solve(identity + 1.0j * reaction_matrix, identity - 1.0j * reaction_matrix)

    np.testing.assert_allclose(Smat, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(Smat.conj().T @ Smat, identity, rtol=1.0e-12, atol=1.0e-12)


def test_get_Smat_includes_closed_channels_before_extracting_open_block() -> None:
    energy = 0.2
    Rmatch = 8.0
    reduced_mass = 3.0
    E_int = np.array([0.0, 0.5])
    L = np.array([0.0, 1.2])
    reaction_matrix = np.array([[0.25, 0.08], [0.08, -0.3]])
    J, N, J_prime, N_prime = _reference_matrices(energy, Rmatch, reduced_mass, E_int, L)
    Ymat = _logD_from_reaction_matrix(J, N, J_prime, N_prime, reaction_matrix)

    (Smat,) = get_Smat(Ymat[None, :, :], Rmatch, [energy], reduced_mass, E_int, L)
    expected = (1.0 - 0.25j) / (1.0 + 0.25j)

    assert Smat.shape == (1, 1)
    np.testing.assert_allclose(Smat[0, 0], expected, rtol=1.0e-12, atol=1.0e-12)


def test_get_Smat_complex_capture_boundary_is_nonunitary() -> None:
    energy = 0.4
    Rmatch = 6.0
    reduced_mass = 2.0
    E_int = np.array([0.0])
    L = np.array([0.0])
    reaction_matrix = np.array([[-0.2j]])
    J, N, J_prime, N_prime = _reference_matrices(energy, Rmatch, reduced_mass, E_int, L)
    Ymat = _logD_from_reaction_matrix(J, N, J_prime, N_prime, reaction_matrix)

    (Smat,) = get_Smat(Ymat[None, :, :], Rmatch, [energy], reduced_mass, E_int, L)

    np.testing.assert_allclose(Smat[0, 0], 2.0 / 3.0, rtol=1.0e-12, atol=1.0e-12)
    assert abs(Smat[0, 0]) < 1.0
