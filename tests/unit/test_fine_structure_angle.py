import numpy as np
from scipy.special import roots_legendre

from pyticc.basis.angle import norm_reduced_wigner_d, norm_reduced_wigner_d_half


def test_doubled_wigner_d_matches_integer_implementation() -> None:
    theta = np.linspace(0.1, 2.9, 17)
    for j in range(4):
        for K in range(-j, j + 1):
            for omega in range(-j, j + 1):
                np.testing.assert_allclose(
                    norm_reduced_wigner_d_half(2 * j, 2 * K, 2 * omega, theta),
                    norm_reduced_wigner_d(j, K, omega, theta),
                    atol=2.0e-14,
                    rtol=2.0e-14,
                )


def test_half_integer_wigner_d_is_normalized() -> None:
    x, weights = np.polynomial.legendre.leggauss(80)
    values = norm_reduced_wigner_d_half(3, 1, -1, np.arccos(x))

    np.testing.assert_allclose(np.sum(weights * values**2), 1.0, atol=2.0e-14)


def test_high_half_integer_wigner_d_remains_normalized() -> None:
    for two_j, two_K, two_omega in ((97, 1, -3), (97, 1, 1), (201, 101, 1), (1001, 1001, 9)):
        n_theta = max(120, (two_j + 5) // 2 + 20)
        x, weights = roots_legendre(n_theta)
        values = norm_reduced_wigner_d_half(two_j, two_K, two_omega, np.arccos(x))
        np.testing.assert_allclose(np.sum(weights * values**2), 1.0, atol=2.0e-12)


def test_high_half_integer_wigner_d_matrix_is_orthogonal() -> None:
    two_j = 97
    projections = range(-two_j, two_j + 1, 2)
    normalization = np.sqrt((two_j + 1.0) / 2.0)
    matrix = (
        np.asarray([[norm_reduced_wigner_d_half(two_j, two_K, two_omega, 0.731) for two_omega in projections] for two_K in projections])
        / normalization
    )

    np.testing.assert_allclose(matrix @ matrix.T, np.eye(two_j + 1), atol=2.0e-12)


def test_high_integer_wigner_d_remains_normalized() -> None:
    x, weights = roots_legendre(240)
    values = norm_reduced_wigner_d(200, 73, 4, np.arccos(x))

    np.testing.assert_allclose(np.sum(weights * values**2), 1.0, atol=2.0e-12)
