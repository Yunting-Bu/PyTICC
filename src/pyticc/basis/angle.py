from functools import lru_cache

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.special import gammaln, lpmv, roots_legendre
from sympy.physics.wigner import clebsch_gordan as sympy_clebsch_gordan


# --------------------------------------------------------------------------------
def gauss_legendre_dvr(theta_min: float, theta_max: float, nth: int, sysmetry: bool = False) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Get Gauss-Legendre DVR grids and weights in the interval [theta_min, theta_max] with nth points.
    If sysmetry is True, the grids and weights are symmetrically distributed around the center of the interval.
    If sysmetry is False, the grids and weights are distributed in the whole interval.

    Inputs:
        tehta_min: float - left boundary of the interval in radians
        theta_max: float - right boundary of the interval in radians
        nth: int - number of points
        sysmetry: bool - whether to use symmetry or not

    Returns:
        grids: NDArray[np.float64] - Gauss-Legendre DVR grids
        weights: NDArray[np.float64] - Gauss-Legendre DVR weights
    """

    n_full = 2 * nth if sysmetry else nth
    x, w = roots_legendre(n_full)
    center_theta = 0.5 * (theta_max + theta_min)
    half_range = 0.5 * (theta_max - theta_min)
    full_weights = half_range * w

    if sysmetry:
        grids = (center_theta + half_range * x[:nth]).copy()
        weights = (2.0 * full_weights[:nth]).copy()
    else:
        grids = (center_theta + half_range * x).copy()
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
        x: float | NDArray[np.float64] - argument of the associated Legendre polynomial

    Returns:
        norm_YjK: float | NDArray[np.float64] - normalized associated Legendre polynomial
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
@lru_cache
def clebsch_gordan(j1: int, m1: int, j2: int, m2: int, j_couple: int) -> float:
    r"""
    Get the Clebsch-Gordan coefficient ``<j1 m1, j2 m2 | j_couple, m1+m2>``.

    SymPy uses the Condon-Shortley phase convention, consistent with the ``CG`` and
    ``F3J`` routines in the reference TICC code.

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
