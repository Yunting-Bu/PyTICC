import numpy as np
from loguru import logger
from scipy import special


# ----------------------------------------------------------------------------------------
def riccati_bessel_jy(ell: float, x: float) -> tuple[float, float, float, float]:
    r"""
    Open channel Riccati-Bessel functions and derivatives.

    Inputs:
        ell : float - angular momentum quantum number
        x : float - argument of the Riccati-Bessel functions

    Returns:
        j_ell : float - Riccati-Bessel function of the first kind
        y_ell : float - Riccati-Bessel function of the second kind
        j_ell_prime : float - derivative of the Riccati-Bessel function of the first kind
        y_ell_prime : float - derivative of the Riccati-Bessel function of the second kind
    """
    if x <= 0.0 or ell < 0.0:
        message = f"x must be positive and ell must be non-negative, but got x={x}, ell={ell}"
        logger.error(message)
        raise ValueError(message)

    nu = ell + 0.5
    scale = np.sqrt(np.pi * x / 2.0)

    unscaled_j = special.jv(nu, x)
    unscaled_y = special.yv(nu, x)
    unscaled_j_prime = special.jvp(nu, x)
    unscaled_y_prime = special.yvp(nu, x)

    j_ell = scale * unscaled_j
    y_ell = scale * unscaled_y
    j_ell_prime = scale * (unscaled_j_prime + 0.5 * unscaled_j / x)
    y_ell_prime = scale * (unscaled_y_prime + 0.5 * unscaled_y / x)

    value = np.array([j_ell, y_ell, j_ell_prime, y_ell_prime], dtype=np.float64)
    if not np.isfinite(value).all():
        message = f"Riccati-Bessel functions and derivatives must be finite, but got {value}"
        logger.error(message)
        raise ValueError(message)

    return j_ell, y_ell, j_ell_prime, y_ell_prime


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def modified_bessel_IK_logD(nu: float, x: float) -> tuple[float, float]:
    """
    Closed channel modified Bessel log derivatives

    Inputs:
        nu: float - order of the modified Bessel function
        x: float - argument of the modified Bessel function

    Returns:
        I_logD: float - I'_nu(x) / I_nu(x)
        K_logD: float - K'_nu(x) / K_nu(x)
    """
    if x <= 0.0 or nu < 0.0:
        message = f"x must be positive and nu must be non-negative, but got x={x}, nu={nu}"
        logger.error(message)
        raise ValueError(message)

    I_nu = special.ive(nu, x)
    Im_nu = special.ive(nu - 1.0, x)
    K_nu = special.kve(nu, x)
    Km_nu = special.kve(nu - 1.0, x)

    I_logD = Im_nu / I_nu - nu / x
    K_logD = -Km_nu / K_nu - nu / x

    value = np.array([I_logD, K_logD], dtype=np.float64)
    if not np.isfinite(value).all():
        message = f"Modified Bessel log derivatives must be finite, but got {value}"
        logger.error(message)
        raise ValueError(message)

    return I_logD, K_logD


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def modified_bessel_K_logD(nu: float, x: float) -> float:
    r"""
    Evaluate the logarithmic derivative of the decaying closed-channel solution.

    The exponentially scaled function ``kve`` avoids underflow when the closed-channel
    wave number times the matching distance is large.

    Formula:
        K'_nu(x) / K_nu(x) = -K_{nu-1}(x) / K_nu(x) - nu / x,

        where ``nu`` is the modified-Bessel order and ``x > 0`` is its argument.

    Inputs:
        nu: float - non-negative modified-Bessel order
        x: float - positive dimensionless argument

    Returns:
        K_logD: float - logarithmic derivative ``K'_nu(x) / K_nu(x)``
    """
    if x <= 0.0 or nu < 0.0:
        message = f"x must be positive and nu must be non-negative, but got x={x}, nu={nu}"
        logger.error(message)
        raise ValueError(message)

    K_nu = special.kve(nu, x)
    Km_nu = special.kve(nu - 1.0, x)
    K_logD = float(-Km_nu / K_nu - nu / x)
    if not np.isfinite(K_logD):
        message = f"Modified Bessel K logarithmic derivative must be finite, but got {K_logD}"
        logger.error(message)
        raise ValueError(message)

    return K_logD


# ----------------------------------------------------------------------------------------
