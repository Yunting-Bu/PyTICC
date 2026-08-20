from collections.abc import Callable
from typing import Any, TypeAlias, overload

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike, NDArray

from pyticc.pes.total import TotalPES

AsymptoticPotential: TypeAlias = Callable[[int, NDArray[np.float64]], NDArray[np.float64]]


# ----------------------------------------------------------------------------------------
def mass_scale(mass: ArrayLike) -> tuple[float, NDArray[np.float64]]:
    r"""
    Return the Delves hyperradial mass and arrangement scale factors.

    Formula:
        For total mass M and arrangement a = A, B, C,

        mu = sqrt(m_A m_B m_C / M),

        scale_a = sqrt[(m_a/mu)(1-m_a/M)].

        The scale factors connect ABC's mass-scaled Jacobi coordinates to
        physical Jacobi coordinates through

        physical_R = scaled_R / scale_a,
        physical_r = scale_a scaled_r.

    Inputs:
        mass: ArrayLike - masses (A,B,C) in any one common unit

    Returns:
        reduced_mass: float - hyperradial mass mu in the input mass unit
        scale: NDArray[np.float64] - dimensionless factors for arrangements
            (A+BC, B+CA, C+AB), shape (3,)
    """
    masses = _validate_mass(mass)
    total_mass = float(np.sum(masses))
    reduced_mass = float(np.sqrt(np.prod(masses) / total_mass))
    scale = np.sqrt((masses / reduced_mass) * (1.0 - masses / total_mass))
    return reduced_mass, scale


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def delves_bonds(
    scaled_R: ArrayLike,
    scaled_r: ArrayLike,
    cos_gamma: ArrayLike,
    arrangement: int,
    mass: ArrayLike,
) -> NDArray[np.float64]:
    r"""
    Convert arrangement-local Delves coordinates to the three physical bonds.

    The arrangement convention is the same as ABC:

        1 = A+BC, 2 = B+CA, 3 = C+AB.

    Unlike ABC's internal ``r(1)=BC`` storage, the returned coordinate axis is
    always ordered ``(r_AB, r_BC, r_CA)``. All coordinate inputs are broadcast
    together, so the function accepts scalars or arbitrary matching grids.

    Formula:
        Let a be the separated atom and (b,c) the diatom. With

        physical_R = scaled_R / scale_a,
        physical_r = scale_a scaled_r,

        shift_b = m_b physical_r/(m_b+m_c),
        shift_c = m_c physical_r/(m_b+m_c),

        r_bc = physical_r,
        r_ca = sqrt(physical_R^2 - 2 physical_R shift_b cos(gamma)
                    + shift_b^2),
        r_ab = sqrt(physical_R^2 + 2 physical_R shift_c cos(gamma)
                    + shift_c^2),

        where the last two labels rotate cyclically with the arrangement.

    Inputs:
        scaled_R: ArrayLike - mass-scaled atom--diatom separation in bohr
        scaled_r: ArrayLike - mass-scaled diatomic bond coordinate in bohr
        cos_gamma: ArrayLike - cosine of the Jacobi angle
        arrangement: int - one-based arrangement index, 1, 2, or 3
        mass: ArrayLike - masses (A,B,C) in any one common unit

    Returns:
        bonds: NDArray[np.float64] - physical bonds with leading coordinate axis
            ``(r_AB,r_BC,r_CA)`` and shape ``(3,*broadcast_shape)``
    """
    if arrangement not in (1, 2, 3):
        message = f"arrangement must be 1, 2, or 3, but got {arrangement}"
        logger.error(message)
        raise ValueError(message)

    masses = _validate_mass(mass)
    _, scale = mass_scale(masses)
    try:
        scaled_R_grid, scaled_r_grid, cos_gamma_grid = np.broadcast_arrays(
            np.asarray(scaled_R, dtype=np.float64),
            np.asarray(scaled_r, dtype=np.float64),
            np.asarray(cos_gamma, dtype=np.float64),
        )
    except ValueError as error:
        message = "scaled_R, scaled_r, and cos_gamma must be broadcast-compatible"
        logger.error(message)
        raise ValueError(message) from error

    if not np.all(np.isfinite(scaled_R_grid)) or np.any(scaled_R_grid < 0.0):
        message = "scaled_R must contain finite non-negative values"
        logger.error(message)
        raise ValueError(message)
    if not np.all(np.isfinite(scaled_r_grid)) or np.any(scaled_r_grid < 0.0):
        message = "scaled_r must contain finite non-negative values"
        logger.error(message)
        raise ValueError(message)
    if not np.all(np.isfinite(cos_gamma_grid)) or np.any(np.abs(cos_gamma_grid) > 1.0 + 1.0e-12):
        message = "cos_gamma must contain finite values in [-1,1]"
        logger.error(message)
        raise ValueError(message)
    cos_gamma_grid = np.clip(cos_gamma_grid, -1.0, 1.0)

    ia = arrangement - 1
    ib = (ia + 1) % 3
    ic = 3 - ia - ib
    physical_R = scaled_R_grid / scale[ia]
    physical_r = scale[ia] * scaled_r_grid
    shift_b = masses[ib] * physical_r / (masses[ib] + masses[ic])
    shift_c = masses[ic] * physical_r / (masses[ib] + masses[ic])

    opposite_bond = np.empty((3, *physical_r.shape), dtype=np.float64)
    opposite_bond[ia] = physical_r
    opposite_bond[ib] = np.sqrt(np.maximum(0.0, physical_R**2 - 2.0 * physical_R * shift_b * cos_gamma_grid + shift_b**2))
    opposite_bond[ic] = np.sqrt(np.maximum(0.0, physical_R**2 + 2.0 * physical_R * shift_c * cos_gamma_grid + shift_c**2))

    # opposite_bond is ABC's cyclic storage (BC,CA,AB); expose the PES order AB,BC,CA.
    return opposite_bond[[2, 0, 1]]


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@overload
def transform_delves_coordinates(
    theta_a: float,
    cos_gamma_a: float,
    arrangement_a: int,
    arrangement_b: int,
    mass: ArrayLike,
) -> tuple[float, float, float]: ...


@overload
def transform_delves_coordinates(
    theta_a: NDArray[Any],
    cos_gamma_a: ArrayLike,
    arrangement_a: int,
    arrangement_b: int,
    mass: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...


@overload
def transform_delves_coordinates(
    theta_a: float,
    cos_gamma_a: NDArray[Any],
    arrangement_a: int,
    arrangement_b: int,
    mass: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...


def transform_delves_coordinates(
    theta_a: ArrayLike,
    cos_gamma_a: ArrayLike,
    arrangement_a: int,
    arrangement_b: int,
    mass: ArrayLike,
) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64], float | NDArray[np.float64]]:
    r"""
    Transform fixed-hyperradius Delves angles between two arrangements.

    This is the vectorized form of ABC ``coords``. The hyperradius cancels, so
    only ``theta_a`` and ``cos_gamma_a`` are required. ``beta_ab`` is the signed
    angle from arrangement a's body-fixed Jacobi axis to arrangement b's axis.

    Formula:
        For arrangements a != b, define

        s0 = 1/(scale_a scale_b),
        c_ab = -mu s0/m_c,
        s_ab = epsilon_ab s0,

        where c is the atom not equal to a or b, and epsilon_ab=-1 when
        a-b=1 or b-a=2 and +1 otherwise. With R_a=cos(theta_a),
        r_a=sin(theta_a), and x_a=cos(gamma_a),

        R_b^2 = (c_ab R_a)^2 - 2 c_ab s_ab R_a r_a x_a
                + (s_ab r_a)^2,

        r_b^2 = (s_ab R_a)^2 + 2 c_ab s_ab R_a r_a x_a
                + (c_ab r_a)^2,

        x_b = [c_ab s_ab(R_a^2-r_a^2)
               +(c_ab^2-s_ab^2)R_a r_a x_a]/(R_b r_b),

        theta_b = atan2(r_b,R_b),

        beta_ab = epsilon_ab acos[(R_b c_ab+r_b x_b s_ab)/R_a].

        Here mu is the Delves hyperradial mass returned by ``mass_scale`` and
        scale_a are its arrangement scale factors. For a=b the transformation
        is the identity and beta_aa=0. Angles are in radians.

    Inputs:
        theta_a: ArrayLike - source hyperangle in 0<=theta<pi/2, shape (...)
        cos_gamma_a: ArrayLike - source Jacobi-angle cosine in [-1,1],
            broadcast-compatible with theta_a
        arrangement_a: int - one-based source arrangement, 1, 2, or 3
        arrangement_b: int - one-based target arrangement, 1, 2, or 3
        mass: ArrayLike - masses (A,B,C) in any one common unit

    Returns:
        theta_b: float | NDArray[np.float64] - target hyperangle, shape (...)
        cos_gamma_b: float | NDArray[np.float64] - target Jacobi-angle cosine,
            shape (...)
        beta_ab: float | NDArray[np.float64] - signed body-axis rotation angle,
            shape (...)
    """
    if arrangement_a not in (1, 2, 3) or arrangement_b not in (1, 2, 3):
        message = f"arrangements must be 1, 2, or 3, but got {(arrangement_a, arrangement_b)}"
        logger.error(message)
        raise ValueError(message)
    masses = _validate_mass(mass)
    try:
        theta_grid, cos_gamma_grid = np.broadcast_arrays(
            np.asarray(theta_a, dtype=np.float64),
            np.asarray(cos_gamma_a, dtype=np.float64),
        )
    except ValueError as error:
        message = "theta_a and cos_gamma_a must be broadcast-compatible"
        logger.error(message)
        raise ValueError(message) from error
    if not np.all(np.isfinite(theta_grid)) or np.any(theta_grid < 0.0) or np.any(theta_grid >= 0.5 * np.pi):
        message = "theta_a must contain finite values in [0,pi/2)"
        logger.error(message)
        raise ValueError(message)
    if not np.all(np.isfinite(cos_gamma_grid)) or np.any(np.abs(cos_gamma_grid) > 1.0 + 1.0e-12):
        message = "cos_gamma_a must contain finite values in [-1,1]"
        logger.error(message)
        raise ValueError(message)
    cos_gamma_grid = np.clip(cos_gamma_grid, -1.0, 1.0)

    if arrangement_a == arrangement_b:
        theta_b = theta_grid.copy()
        cos_gamma_b = cos_gamma_grid.copy()
        beta_ab = np.zeros_like(theta_grid)
    else:
        reduced_mass, scale = mass_scale(masses)
        ia = arrangement_a - 1
        ib = arrangement_b - 1
        ic = 3 - ia - ib
        sine_ab = 1.0 / (scale[ia] * scale[ib])
        cosine_ab = -reduced_mass * sine_ab / masses[ic]
        orientation = -1.0 if (arrangement_a - arrangement_b == 1 or arrangement_b - arrangement_a == 2) else 1.0
        sine_ab *= orientation

        scaled_R_a = np.cos(theta_grid)
        scaled_r_a = np.sin(theta_grid)
        product_a = scaled_R_a * scaled_r_a * cos_gamma_grid
        scaled_R_b = np.sqrt(np.maximum(0.0, (cosine_ab * scaled_R_a) ** 2 - 2.0 * cosine_ab * sine_ab * product_a + (sine_ab * scaled_r_a) ** 2))
        scaled_r_b = np.sqrt(np.maximum(0.0, (sine_ab * scaled_R_a) ** 2 + 2.0 * cosine_ab * sine_ab * product_a + (cosine_ab * scaled_r_a) ** 2))
        product_b = cosine_ab * sine_ab * (scaled_R_a - scaled_r_a) * (scaled_R_a + scaled_r_a)
        product_b += product_a * (cosine_ab - sine_ab) * (cosine_ab + sine_ab)
        denominator = scaled_R_b * scaled_r_b
        cos_gamma_b = np.divide(product_b, denominator, out=np.zeros_like(product_b), where=denominator > 0.0)
        cos_gamma_b = np.clip(cos_gamma_b, -1.0, 1.0)
        theta_b = np.arctan2(scaled_r_b, scaled_R_b)
        cos_beta = (scaled_R_b * cosine_ab + scaled_r_b * cos_gamma_b * sine_ab) / scaled_R_a
        beta_ab = orientation * np.arccos(np.clip(cos_beta, -1.0, 1.0))

    if theta_b.ndim == 0:
        return float(theta_b), float(cos_gamma_b), float(beta_ab)
    return theta_b, cos_gamma_b, beta_ab


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def asymptotic_potential(
    total_pes: TotalPES,
    mass: ArrayLike,
    *,
    scaled_R: float = 100.0,
    cos_gamma: float = 0.0,
) -> AsymptoticPotential:
    r"""
    Adapt a total three-bond PES to ``build_delves_basis``.

    This follows ABC's automatic-basis scan: the separated-atom coordinate is
    fixed at ``scaled_R=100`` bohr and ``cos(gamma)=0`` by default, then the
    complete triatomic potential is evaluated while ``scaled_r`` is scanned.
    No diatomic reference potential is subtracted.

    ``total_pes`` receives a Fortran-friendly array with shape ``(3,n_grid)``
    in physical-bond order ``(r_AB,r_BC,r_CA)`` and must return total energies
    with shape ``(n_grid,)``.

    Inputs:
        total_pes: TotalPES - scalar adiabatic total PES using physical bonds in
            bohr and returning Hartree
        mass: ArrayLike - masses (A,B,C) in any one common unit
        scaled_R: float - fixed asymptotic mass-scaled separation in bohr
        cos_gamma: float - fixed cosine of the Jacobi angle

    Returns:
        potential: AsymptoticPotential - callback accepting one-based
            arrangement and a scaled-r array, suitable for build_delves_basis
    """
    masses = tuple(_validate_mass(mass))
    if not np.isfinite(scaled_R) or scaled_R <= 0.0:
        message = f"scaled_R must be finite and positive, but got {scaled_R}"
        logger.error(message)
        raise ValueError(message)
    if not np.isfinite(cos_gamma) or abs(cos_gamma) > 1.0:
        message = f"cos_gamma must be finite and in [-1,1], but got {cos_gamma}"
        logger.error(message)
        raise ValueError(message)

    def potential(arrangement: int, scaled_r: NDArray[np.float64]) -> NDArray[np.float64]:
        scaled_r_grid = np.asarray(scaled_r, dtype=np.float64)
        if scaled_r_grid.ndim != 1 or not np.all(np.isfinite(scaled_r_grid)) or np.any(scaled_r_grid < 0.0):
            message = f"scaled_r must be a finite non-negative one-dimensional array, but got shape={scaled_r_grid.shape}"
            logger.error(message)
            raise ValueError(message)
        bonds = delves_bonds(scaled_R, scaled_r_grid, cos_gamma, arrangement, masses)
        values = np.asarray(total_pes(np.asfortranarray(bonds)), dtype=np.float64)
        if values.shape != scaled_r_grid.shape or not np.all(np.isfinite(values)):
            message = f"total_pes returned shape {values.shape}; expected finite values with shape {scaled_r_grid.shape}"
            logger.error(message)
            raise ValueError(message)
        return values

    return potential


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def get_Vgrid_delves(
    total_pes: TotalPES,
    rho: float,
    arrangement: int,
    theta: ArrayLike,
    cos_gamma: ArrayLike,
    mass: ArrayLike,
) -> NDArray[np.float64]:
    r"""
    Evaluate the total PES on one arrangement's fixed-hyperradius grid.

    Formula:
        scaled_R_q = rho cos(theta_q),
        scaled_r_q = rho sin(theta_q),

        V_qp^(a) = V_total[r_AB^(a)(q,p),r_BC^(a)(q,p),r_CA^(a)(q,p)],

        where a=1,2,3 denotes A+BC, B+CA, and C+AB. ``delves_bonds``
        performs the arrangement-dependent mass scaling and always supplies the
        total PES in physical-bond order ``(r_AB,r_BC,r_CA)``.

    Inputs:
        total_pes: TotalPES - scalar adiabatic total PES in bohr and Hartree
        rho: float - positive hyperradius in bohr
        arrangement: int - one-based arrangement index, 1, 2, or 3
        theta: ArrayLike - internal hyperangle nodes in radians, shape
            ``(n_theta,)`` with 0<theta<pi/2
        cos_gamma: ArrayLike - Jacobi-angle cosine nodes, shape ``(n_gamma,)``
        mass: ArrayLike - masses (A,B,C) in any one common unit

    Returns:
        potential: NDArray[np.float64] - total energies in Hartree indexed as
            ``[theta,cos_gamma]``, shape ``(n_theta,n_gamma)``
    """
    if not np.isfinite(rho) or rho <= 0.0:
        message = f"rho must be finite and positive, but got {rho}"
        logger.error(message)
        raise ValueError(message)
    theta_grid = np.asarray(theta, dtype=np.float64)
    cos_gamma_grid = np.asarray(cos_gamma, dtype=np.float64)
    if theta_grid.ndim != 1 or not np.all(np.isfinite(theta_grid)) or np.any(theta_grid <= 0.0) or np.any(theta_grid >= 0.5 * np.pi):
        message = f"theta must be a finite one-dimensional array inside (0,pi/2), but got shape={theta_grid.shape}"
        logger.error(message)
        raise ValueError(message)
    if cos_gamma_grid.ndim != 1 or not np.all(np.isfinite(cos_gamma_grid)) or np.any(np.abs(cos_gamma_grid) > 1.0):
        message = f"cos_gamma must be a finite one-dimensional array in [-1,1], but got shape={cos_gamma_grid.shape}"
        logger.error(message)
        raise ValueError(message)

    scaled_R = rho * np.cos(theta_grid[:, None])
    scaled_r = rho * np.sin(theta_grid[:, None])
    bonds = delves_bonds(scaled_R, scaled_r, cos_gamma_grid[None, :], arrangement, mass)
    grid_shape = (theta_grid.size, cos_gamma_grid.size)
    return total_pes(np.asfortranarray(bonds.reshape(3, -1))).reshape(grid_shape)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _validate_mass(mass: ArrayLike) -> NDArray[np.float64]:
    """Return three finite positive atomic masses as a float64 array."""
    masses = np.asarray(mass, dtype=np.float64)
    if masses.shape != (3,) or not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        message = f"mass must contain three finite positive values, but got {mass!r}"
        logger.error(message)
        raise ValueError(message)
    return masses


# ----------------------------------------------------------------------------------------
