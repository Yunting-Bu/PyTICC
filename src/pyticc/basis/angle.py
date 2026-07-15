import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.special import gammaln, lpmv, roots_legendre


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
def norm_YjK(j: int, K: int, x: float) -> float:
    r"""
    Get the normalization factor for the associated Legendre polynomial Y_{jK}(x).

    Inputs:
        j: int - degree of the associated Legendre polynomial
        K: int - order of the associated Legendre polynomial
        x: float - argument of the associated Legendre polynomial

    Returns:
        norm_YjK: float - normalization factor for the associated Legendre polynomial Y_{jK}(x)
    """

    m = abs(K)
    if m > j:
        logger.error("Invalid input: |K| > j")
        raise ValueError("Invalid input: |K| > j")

    log_factor = 0.5 * (np.log((2.0 * j + 1.0) / 2.0) + gammaln(j - m + 1.0) - gammaln(j + m + 1.0))

    factor = np.exp(log_factor)

    if K < 0 and m % 2 == 1:
        factor *= -1.0

    return factor * lpmv(m, j, x)
