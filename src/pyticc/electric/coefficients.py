import numpy as np
from numpy.typing import NDArray

from pyticc.electric.response import ElectricResponseValues


# ----------------------------------------------------------------------------------------
def electric_coefficients(response: ElectricResponseValues, electric_strength: float) -> NDArray[np.float64]:
    r"""
    Get the electric-interaction coefficients through first hyperpolarizability.

    Formula:
        Let x = cos(theta), and let the electric field be E along the SF-Z axis.
        The electric interaction is

        V_E(r,x) = sum_{n=0}^{3} a_n(r) x^n,

        where

        a_0(r) = -1/2 alpha_xx(r) E^2,

        a_1(r) = -mu_z(r) E - 1/2 beta_xxz(r) E^3,

        a_2(r) = -1/2 [alpha_zz(r) - alpha_xx(r)] E^2,

        a_3(r) = -1/6 [beta_zzz(r) - 3 beta_xxz(r)] E^3.

        This is equivalent to

        V_E(r,x)
          = -mu_z(r) E x
            - 1/2 E^2 {alpha_xx(r)
                       + [alpha_zz(r)-alpha_xx(r)] x^2}
            - 1/6 E^3 {3 beta_xxz(r) x
                       + [beta_zzz(r)-3 beta_xxz(r)] x^3}.

    Inputs:
        response: ElectricResponseValues - electric-response components
            evaluated on the desired radial grid, each with shape (...)
        electric_strength: float - electric-field strength E in atomic units

    Returns:
        coefficients: NDArray[np.float64] - coefficients a_n(r), indexed as
            coefficients[n, ...], shape (4, ...)
    """
    strength_2 = electric_strength**2
    strength_3 = electric_strength**3
    a_0 = -0.5 * response.alpha_xx * strength_2
    a_1 = -response.mu_z * electric_strength - 0.5 * response.beta_xxz * strength_3
    a_2 = -0.5 * (response.alpha_zz - response.alpha_xx) * strength_2
    a_3 = -(response.beta_zzz - 3.0 * response.beta_xxz) * strength_3 / 6.0
    return np.stack((a_0, a_1, a_2, a_3), axis=0)


# ----------------------------------------------------------------------------------------
