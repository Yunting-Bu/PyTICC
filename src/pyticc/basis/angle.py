from functools import lru_cache

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.special import gammaln, lpmv, roots_legendre
from sympy.physics.wigner import clebsch_gordan as sympy_clebsch_gordan


# --------------------------------------------------------------------------------
def gauss_legendre_dvr(lower: float, upper: float, n_points: int, symmetry: bool = False) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return Gauss-Legendre nodes and weights on an interval.

    With ``symmetry=True``, ``n_points`` nodes are retained from one half of a
    ``2 * n_points`` rule and their weights are doubled. This is valid when the
    complete integrand is symmetric about the interval center.

    Inputs:
        lower: float - lower integration bound
        upper: float - upper integration bound
        n_points: int - number of returned quadrature points
        symmetry: bool - whether to retain only one half of a symmetric rule

    Returns:
        grids: NDArray[np.float64] - quadrature nodes, shape (n_points,)
        weights: NDArray[np.float64] - quadrature weights, shape (n_points,)
    """
    if n_points < 1:
        message = f"n_points must be positive, but got {n_points}"
        logger.error(message)
        raise ValueError(message)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        message = f"Quadrature bounds must be finite and increasing, but got lower={lower}, upper={upper}"
        logger.error(message)
        raise ValueError(message)

    n_full = 2 * n_points if symmetry else n_points
    x, w = roots_legendre(n_full)
    center = 0.5 * (upper + lower)
    half_range = 0.5 * (upper - lower)
    full_weights = half_range * w

    if symmetry:
        grids = (center + half_range * x[:n_points]).copy()
        weights = (2.0 * full_weights[:n_points]).copy()
    else:
        grids = (center + half_range * x).copy()
        weights = full_weights.copy()

    return grids, weights


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
def norm_YjK(j: int, K: int, x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    r"""
    Get the normalization factor for the associated Legendre polynomial Y_{jK}(x).

    Inputs:
        j: int - degree of the associated Legendre polynomial
        K: int - order of the associated Legendre polynomial
        x: float | NDArray[np.float64] - scalar argument or array of arguments with
            shape (...)

    Returns:
        norm_YjK: float | NDArray[np.float64] - scalar value for scalar input, or
            values with the same shape (...) as x
    """

    m = abs(K)
    if m > j:
        message = f"Invalid input: |K|={m} exceeds j={j}"
        logger.error(message)
        raise ValueError(message)

    log_factor = 0.5 * (np.log((2.0 * j + 1.0) / 2.0) + gammaln(j - m + 1.0) - gammaln(j + m + 1.0))

    factor = np.exp(log_factor)

    if K < 0 and m % 2 == 1:
        factor *= -1.0

    return factor * lpmv(m, j, x)


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
def norm_reduced_wigner_d(
    j: int,
    K: int,
    omega: int,
    theta: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    r"""
    Get a reduced Wigner d function normalized for integration over cos(theta).

    This follows ``DJMM(..., ID=3)`` in the ABC+D reference code, including its
    phase convention.

    Formula:
        D_tilde^{j}_{K,omega}(theta)
            = sqrt((2j+1)/2) d^{j}_{K,omega}(theta)

    Inputs:
        j: int - angular momentum quantum number
        K: int - body-fixed projection on the intermolecular axis
        omega: int - projection on the triatomic internal axis
        theta: float | NDArray[np.float64] - polar angle in radians, scalar or
            array with shape (...)

    Returns:
        value: float | NDArray[np.float64] - normalized reduced Wigner d value,
            with the same shape (...) as theta
    """
    if abs(K) > j or abs(omega) > j:
        message = f"Invalid Wigner d indices: j={j}, K={K}, omega={omega}"
        logger.error(message)
        raise ValueError(message)

    angles = np.asarray(theta, dtype=np.float64)
    cosine = np.cos(0.5 * angles)
    sine = np.sin(0.5 * angles)
    log_prefactor = 0.5 * (gammaln(j + omega + 1.0) + gammaln(j - omega + 1.0) + gammaln(j + K + 1.0) + gammaln(j - K + 1.0))
    result = np.zeros_like(angles)

    for k in range(max(0, omega - K), min(j - K, j + omega) + 1):
        cosine_power = 2 * j + omega - K - 2 * k
        sine_power = K - omega + 2 * k
        log_denominator = gammaln(j - K - k + 1.0) + gammaln(j + omega - k + 1.0) + gammaln(K - omega + k + 1.0) + gammaln(k + 1.0)
        phase = -1.0 if (k + sine_power) % 2 else 1.0
        result += phase * np.exp(log_prefactor - log_denominator) * cosine**cosine_power * sine**sine_power

    result *= np.sqrt((2.0 * j + 1.0) / 2.0)
    return float(result) if result.ndim == 0 else result


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
@lru_cache
def clebsch_gordan(j1: int, m1: int, j2: int, m2: int, j_couple: int) -> float:
    r"""
    Get the Clebsch-Gordan coefficient ``<j1 m1, j2 m2 | j_couple, m1+m2>``.

    Inputs:
        j1: int - angular momentum of the first rotor
        m1: int - body-fixed projection of j1
        j2: int - angular momentum of the second rotor
        m2: int - body-fixed projection of j2
        j_couple: int - coupled angular momentum

    Returns:
        coefficient: float - Clebsch-Gordan coefficient
    """
    M = m1 + m2
    return float(sympy_clebsch_gordan(j1, j2, j_couple, m1, m2, M))


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
def lambda_plus(j, K) -> float:
    r"""
    Get lambda_plus for the given j and K.

    Formula:
        \lambda_+ = \sqrt{j(j+1)-K(K+1)}

    Inputs:
        j: int - angular momentum quantum number
        K: int - projection of the angular momentum

    Returns:
        lambda_plus: float - value of lambda_plus
    """
    return np.sqrt(j * (j + 1) - K * (K + 1))


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
def lambda_minus(j, K) -> float:
    r"""
    Get lambda_minus for the given j and K.

    Formula:
        \lambda_- = \sqrt{j(j+1)-K(K-1)}

    Inputs:
        j: int - angular momentum quantum number
        K: int - projection of the angular momentum
    """
    return np.sqrt(j * (j + 1) - K * (K - 1))
