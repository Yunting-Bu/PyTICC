import numpy as np

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
