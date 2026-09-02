from functools import lru_cache
from typing import Any, overload

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.special import eval_jacobi, gammaln, lpmv, roots_legendre
from sympy import Rational
from sympy.physics.wigner import clebsch_gordan as sympy_clebsch_gordan


# ----------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@overload
def norm_YjK(j: int, K: int, x: float) -> float: ...


# ----------------------------------------------------------------------------------------
@overload
def norm_YjK(j: int, K: int, x: NDArray[Any]) -> NDArray[np.float64]: ...


# ----------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def norm_reduced_wigner_d(
    j: int,
    K: int,
    omega: int,
    theta: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    r"""
    Get a reduced Wigner d function normalized for integration over cos(theta).

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

    return norm_reduced_wigner_d_half(2 * j, 2 * K, 2 * omega, theta)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def norm_reduced_wigner_d_half(
    two_j: int,
    two_K: int,
    two_omega: int,
    theta: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    r"""
    Return a normalized reduced Wigner d function for integer or half-integer j.

    Formula:
        d^j_{K,Omega}(theta) is evaluated with a Jacobi-polynomial form and

        d_tilde^j_{K,Omega}(theta) = sqrt((2j+1)/2) d^j_{K,Omega}(theta).

        Every angular momentum and projection is supplied as twice its physical
        value, which avoids floating-point half-integer comparisons. Symmetry
        relations first map the projections to nonnegative Jacobi parameters;
        unlike the direct alternating factorial sum, this remains stable at
        high j.

    Inputs:
        two_j: int - twice j
        two_K: int - twice the BF projection K
        two_omega: int - twice the molecular-axis projection Omega
        theta: float | NDArray[np.float64] - angle in radians

    Returns:
        value: float | NDArray[np.float64] - normalized reduced Wigner d values
    """
    if min(two_j, two_j + two_K, two_j - two_K, two_j + two_omega, two_j - two_omega) < 0:
        message = f"Invalid doubled Wigner d indices: two_j={two_j}, two_K={two_K}, two_omega={two_omega}"
        logger.error(message)
        raise ValueError(message)
    if any(value % 2 for value in (two_j + two_K, two_j + two_omega)):
        message = "j, K, and Omega must have the same integer or half-integer character"
        logger.error(message)
        raise ValueError(message)

    # Map d^j_{K,Omega} to an equivalent element whose first projection m'
    # satisfies m' >= |m|.  This makes both Jacobi parameters nonnegative.
    # The symmetry phases follow the same convention as the finite factorial
    # sum formerly used here.
    if two_K >= abs(two_omega):
        two_m_prime, two_m = two_K, two_omega
        symmetry_phase = 1.0
    elif two_omega >= abs(two_K):
        two_m_prime, two_m = two_omega, two_K
        symmetry_phase = -1.0 if ((two_K - two_omega) // 2) % 2 else 1.0
    elif -two_K >= abs(two_omega):
        two_m_prime, two_m = -two_K, -two_omega
        symmetry_phase = -1.0 if ((two_K - two_omega) // 2) % 2 else 1.0
    else:
        two_m_prime, two_m = -two_omega, -two_K
        symmetry_phase = 1.0

    alpha = (two_m_prime - two_m) // 2
    beta = (two_m_prime + two_m) // 2
    degree = (two_j - two_m_prime) // 2
    angles = np.asarray(theta, dtype=np.float64)
    cosine = np.cos(0.5 * angles)
    sine = np.sin(0.5 * angles)
    log_prefactor = 0.5 * (
        gammaln((two_j + two_m_prime) // 2 + 1.0)
        + gammaln((two_j - two_m_prime) // 2 + 1.0)
        - gammaln((two_j + two_m) // 2 + 1.0)
        - gammaln((two_j - two_m) // 2 + 1.0)
    )
    jacobi_phase = -1.0 if alpha % 2 else 1.0
    result = symmetry_phase * jacobi_phase * np.exp(log_prefactor) * sine**alpha * cosine**beta * eval_jacobi(degree, alpha, beta, np.cos(angles))
    result *= np.sqrt((two_j + 1.0) / 2.0)
    return float(result) if result.ndim == 0 else result


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@lru_cache
def clebsch_gordan_half(two_j1: int, two_m1: int, two_j2: int, two_m2: int, two_j_couple: int) -> float:
    r"""
    Return a Clebsch--Gordan coefficient from doubled angular momenta.

    Formula:
        The returned real coefficient is

        <j_1 m_1, j_2 m_2 | j_12 M>,
        M=m_1+m_2,

        with ``j_i=two_j_i/2``, ``m_i=two_m_i/2``, and
        ``j_12=two_j_couple/2``. The Condon--Shortley convention is inherited
        from SymPy. All angular momenta and projections may be integer or
        half-integer, but each ``j_i`` and ``m_i`` pair must have the same
        integer character and satisfy ``|m_i|<=j_i``.

    Inputs:
        two_j1: int - twice the first angular momentum j_1
        two_m1: int - twice its projection m_1
        two_j2: int - twice the second angular momentum j_2
        two_m2: int - twice its projection m_2
        two_j_couple: int - twice the coupled angular momentum j_12

    Returns:
        coefficient: float - real Clebsch--Gordan coefficient
    """
    if min(two_j1, two_j2, two_j_couple) < 0:
        message = "Doubled angular momenta must be nonnegative"
        logger.error(message)
        raise ValueError(message)
    if abs(two_m1) > two_j1 or abs(two_m2) > two_j2:
        message = "Doubled projections must not exceed their angular momenta"
        logger.error(message)
        raise ValueError(message)
    if (two_j1 - two_m1) % 2 or (two_j2 - two_m2) % 2:
        message = "Each angular momentum and projection must have the same integer or half-integer character"
        logger.error(message)
        raise ValueError(message)
    if not abs(two_j1 - two_j2) <= two_j_couple <= two_j1 + two_j2 or (two_j1 + two_j2 - two_j_couple) % 2:
        return 0.0

    two_M = two_m1 + two_m2
    if abs(two_M) > two_j_couple or (two_j_couple - two_M) % 2:
        return 0.0
    return float(
        sympy_clebsch_gordan(
            Rational(two_j1, 2),
            Rational(two_j2, 2),
            Rational(two_j_couple, 2),
            Rational(two_m1, 2),
            Rational(two_m2, 2),
            Rational(two_M, 2),
        )
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------
