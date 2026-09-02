from collections.abc import Sequence

import numpy as np
from loguru import logger
from numpy.typing import NDArray


# ----------------------------------------------------------------------------------------
def _coordinates(value: NDArray[np.float64], n_coordinate: int, name: str) -> NDArray[np.float64]:
    coordinates = np.asarray(value)
    if np.iscomplexobj(coordinates):
        message = f"{name} must be real"
        logger.error(message)
        raise ValueError(message)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[0] != n_coordinate:
        message = f"{name} must have shape ({n_coordinate},n_grid), but got {coordinates.shape}"
        logger.error(message)
        raise ValueError(message)
    if not np.all(np.isfinite(coordinates)):
        message = f"{name} must contain only finite values"
        logger.error(message)
        raise ValueError(message)
    return coordinates


# ----------------------------------------------------------------------------------------
def _masses(value: Sequence[float]) -> tuple[float, float, float]:
    masses = tuple(float(mass) for mass in value)
    if len(masses) != 3 or not all(np.isfinite(mass) and mass > 0.0 for mass in masses):
        message = f"masses must contain three finite positive values, but got {value!r}"
        logger.error(message)
        raise ValueError(message)
    return masses


# ----------------------------------------------------------------------------------------
def radau_triatom_cartesian(
    coordinates: NDArray[np.float64],
    masses: Sequence[float],
) -> NDArray[np.float64]:
    r"""
    Convert triatomic Radau coordinates to the corrected bisector-z MF frame.

    The returned origin is the ABC center of mass. The Radau canonical point F
    is generally not the center of mass and is not returned. Atom order is
    ``(A,B,C)`` and Cartesian-axis order is ``(x,y,z)``.

    Formula:
        Let ``q = (r_1,r_2,theta)`` and ``M = m_A + m_B + m_C``. With F as a
        temporary origin, define

        \vec r_1 = r_1 (sin(theta/2), 0, -cos(theta/2)),
        \vec r_2 = r_2 (-sin(theta/2), 0, -cos(theta/2)),

        \vec E = (m_A \vec r_1 + m_C \vec r_2)/(m_A+m_C),
        \vec B_F = [1-sqrt(M/m_B)] \vec E.

        The atom positions relative to F are ``A_F = r_1``, ``B_F`` above,
        and ``C_F = r_2``. Their center of mass is

        \vec O_F = (m_A \vec A_F + m_B \vec B_F + m_C \vec C_F)/M,

        and the returned coordinates are ``X_O = X_F - O_F``. This embedding
        obeys the corrected 2+3 definition

        z_MF || -(r_1 \vec r_2 + r_2 \vec r_1),
        y_MF || \vec r_1 x \vec r_2.

        All lengths and masses may use any internally consistent units. Angles
        are in radians. For ``n_grid`` geometries the return indexing is
        ``cartesian[atom,axis,grid]`` with shape ``(3,3,n_grid)``.

    Inputs:
        coordinates: NDArray[np.float64] - Radau coordinates ordered as
            ``(r_1,r_2,theta)``, shape ``(3,n_grid)``
        masses: Sequence[float] - masses ``(m_A,m_B,m_C)`` in units consistent
            with the caller

    Returns:
        cartesian: NDArray[np.float64] - center-of-mass Cartesian coordinates
            ordered as atoms ``(A,B,C)``, shape ``(3,3,n_grid)``
    """
    q = _coordinates(coordinates, 3, "Radau coordinates")
    mass_A, mass_B, mass_C = _masses(masses)
    r_1, r_2, theta = q
    if np.any(r_1 <= 0.0) or np.any(r_2 <= 0.0):
        message = "Radau lengths r_1 and r_2 must be positive"
        logger.error(message)
        raise ValueError(message)
    if np.any(theta < 0.0) or np.any(theta > np.pi):
        message = "Radau angle theta must lie in [0,pi]"
        logger.error(message)
        raise ValueError(message)

    half_theta = 0.5 * theta
    zeros = np.zeros_like(theta)
    atom_A = np.stack((r_1 * np.sin(half_theta), zeros, -r_1 * np.cos(half_theta)))
    atom_C = np.stack((-r_2 * np.sin(half_theta), zeros, -r_2 * np.cos(half_theta)))

    mass_AC = mass_A + mass_C
    mass_ABC = mass_AC + mass_B
    center_AC = (mass_A * atom_A + mass_C * atom_C) / mass_AC
    atom_B = (1.0 - np.sqrt(mass_ABC / mass_B)) * center_AC
    center_ABC = (mass_A * atom_A + mass_B * atom_B + mass_C * atom_C) / mass_ABC
    return np.asarray(np.stack((atom_A - center_ABC, atom_B - center_ABC, atom_C - center_ABC)), dtype=np.float64)


# ----------------------------------------------------------------------------------------
def atom_triatom_cartesian(
    R: float,
    coordinates: NDArray[np.float64],
    masses: Sequence[float],
) -> NDArray[np.float64]:
    r"""
    Convert 1+3 BAST coordinates to DF Cartesian atom positions.

    Formula:
        The input is ``Q = (r_1,r_2,theta_1,theta_2,phi)``. First construct
        ``X_MF`` for atoms ABC with :func:`radau_triatom_cartesian`. The active
        bisector-z embedding rotation is

        X_DF = R_y(theta_2) R_z(phi) X_MF,

        where column-vector rotation matrices are

        R_z(phi) = [[cos(phi),-sin(phi),0],
                    [sin(phi), cos(phi),0],
                    [0,        0,       1]],

        R_y(beta) = [[ cos(beta),0,sin(beta)],
                     [0,         1,0],
                     [-sin(beta),0,cos(beta)]].

        Thus ``theta_2`` is the polar angle between ``z_MF`` and the
        intermolecular ``z_DF`` axis, while ``phi`` rotates the triatomic plane
        about ``z_MF``. The triatomic center of mass is the DF origin and atom D
        is at ``(0,0,R)``. Angles are in radians. The return indexing is
        ``cartesian[atom,axis,grid]`` in atom order ``(A,B,C,D)``.

    Inputs:
        R: float - distance from the ABC center of mass to atom D
        coordinates: NDArray[np.float64] - BAST coordinates ordered as
            ``(r_1,r_2,theta_1,theta_2,phi)``, shape ``(5,n_grid)``
        masses: Sequence[float] - masses ``(m_A,m_B,m_C)`` in units consistent
            with the caller

    Returns:
        cartesian: NDArray[np.float64] - DF Cartesian positions ordered as
            atoms ``(A,B,C,D)``, shape ``(4,3,n_grid)``
    """
    q = _coordinates(coordinates, 5, "Atom-triatom BAST coordinates")
    separation = float(R)
    if not np.isfinite(separation) or separation <= 0.0:
        message = f"R must be finite and positive, but got {R!r}"
        logger.error(message)
        raise ValueError(message)
    theta_2 = q[3]
    if np.any(theta_2 < 0.0) or np.any(theta_2 > np.pi):
        message = "External polar angle theta_2 must lie in [0,pi]"
        logger.error(message)
        raise ValueError(message)

    monomer = radau_triatom_cartesian(q[:3], masses)
    phi = q[4]
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    x_phi = cos_phi[None, :] * monomer[:, 0] - sin_phi[None, :] * monomer[:, 1]
    y_phi = sin_phi[None, :] * monomer[:, 0] + cos_phi[None, :] * monomer[:, 1]
    z_phi = monomer[:, 2]

    cos_theta_2 = np.cos(theta_2)
    sin_theta_2 = np.sin(theta_2)
    x_df = cos_theta_2[None, :] * x_phi + sin_theta_2[None, :] * z_phi
    y_df = y_phi
    z_df = -sin_theta_2[None, :] * x_phi + cos_theta_2[None, :] * z_phi
    triatom = np.stack((x_df, y_df, z_df), axis=1)

    atom_D = np.zeros((1, 3, q.shape[1]), dtype=np.float64)
    atom_D[:, 2, :] = separation
    return np.asarray(np.concatenate((triatom, atom_D), axis=0), dtype=np.float64)


# ----------------------------------------------------------------------------------------
