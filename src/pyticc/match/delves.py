from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike, NDArray

from pyticc.basis.delves import DelvesBasis
from pyticc.matrix.delves.overlap import get_sector_overlap_delves
from pyticc.matrix.delves.surface import get_delves_reference_basis
from pyticc.pes.total import TotalPES


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DelvesAsymptoticBasis:
    """
    Fixed Jacobi channels and their finite-hyperradius Delves representatives.

    Members:
        qns: tuple[tuple[int, int, int, int], ...] - channel labels ``(a,v,j,K)``
            ordered as arrangement, vibration, rotation, and helicity; arrangements
            and vibrational levels are one- and zero-based, respectively
        energies: NDArray[np.float64] - fixed Jacobi rovibrational thresholds in
            Hartree, shape ``(n_channel,)``
        s_coefficients: NDArray[np.float64] - fixed-Jacobi vibrational coefficients
            in the normalized sine basis on ``0<s<scaled_r_max``, indexed as
            ``[sine,channel]`` with shape ``(basis.n_sine,n_channel)``
        rho_match: float - matching hyperradius in bohr
        theta_coefficients: NDArray[np.float64] - coefficients of the corresponding
            finite-rho, K-independent Delves functions in the complete primitive
            ``(a,j,K,n)`` basis, shape ``(basis.n_primitive,n_channel)``
        theta_energies: NDArray[np.float64] - finite-rho reference eigenvalues in
            Hartree, shape ``(n_channel,)``; these correlate by state index with
            ``energies`` but are not scattering thresholds
    """

    qns: tuple[tuple[int, int, int, int], ...]
    energies: NDArray[np.float64]
    s_coefficients: NDArray[np.float64]
    rho_match: float
    theta_coefficients: NDArray[np.float64]
    theta_energies: NDArray[np.float64]

    @property
    def n_channel(self) -> int:
        """Return the number of retained asymptotic channels."""
        return len(self.qns)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_delves_asymptotic_basis(
    basis: DelvesBasis,
    total_pes: TotalPES,
    rho_match: float,
) -> DelvesAsymptoticBasis:
    r"""
    Build ABC's fixed Jacobi channels and final finite-rho matching basis.

    The fixed Jacobi labels, thresholds, and sine coefficients have already been
    prepared by ``build_ScattSystem``. This function performs only ABC's final
    ``sbasis(mode=1)`` construction at ``rho_match`` in the same hyperangle
    primitive used by the propagated surface basis.

    Formula:
        Let ``mu=sqrt(m_A m_B m_C/(m_A+m_B+m_C))`` and

        At ``rho=rho_match``, define

        theta_max=asin[min(1,s_max/rho)],
        s=rho sin(theta),

        u_n(theta;rho)=sqrt(2/theta_max)
            sin(n pi theta/theta_max).

        For each retained ``(a,v,j,K)``, the state with the same eigenvalue index
        ``v`` is selected from the K-independent finite-rho reference problem

        H^(theta,aj)_nm = delta_nm
          {[(n pi/theta_max)^2-1/4]/(2 mu rho^2)}
          + integral_0^theta_max u_n(theta;rho)
            [j(j+1)/(2 mu rho^2 sin^2(theta))
             +V_a^asym(rho sin(theta))]u_m(theta;rho)dtheta.

        ``epsilon_avj`` stored in ``basis.energies`` remains the scattering threshold;
        the finite-rho eigenvalue only labels and constructs its Delves-correlated
        partner. All integrals use ``basis.n_vib_quad`` midpoint points.

    Inputs:
        basis: DelvesBasis - prepared Delves channels and primitive support;
            masses in electron masses
        total_pes: TotalPES - scalar total adiabatic PES in bohr and Hartree
        rho_match: float - final positive matching hyperradius in bohr

    Returns:
        channels: DelvesAsymptoticBasis - fixed thresholds, channel labels, and
            the two vibrational coefficient representations required by matching
    """
    if not np.isfinite(rho_match) or rho_match <= 0.0:
        message = f"rho_match must be finite and positive, but got {rho_match}"
        logger.error(message)
        raise ValueError(message)

    if not basis.n_channel:
        message = "Delves matching requires channels prepared by build_ScattSystem"
        logger.error(message)
        raise ValueError(message)

    theta_coefficients, theta_energies = get_delves_reference_basis(
        basis,
        total_pes,
        rho_match,
        asymptotic=True,
    )
    return DelvesAsymptoticBasis(
        qns=basis.qns,
        energies=basis.energies.copy(),
        s_coefficients=basis.s_coefficients.copy(),
        rho_match=float(rho_match),
        theta_coefficients=theta_coefficients,
        theta_energies=theta_energies,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def transform_logD_to_delves_channels(
    basis: DelvesBasis,
    rho_surface: float,
    surface_coefficients: ArrayLike,
    Ymat: ArrayLike,
    channels: DelvesAsymptoticBasis,
) -> NDArray[np.float64] | NDArray[np.complex128]:
    r"""
    Transform the propagated surface LogD to ABC's final Delves channel basis.

    Formula:
        Let ``C_surface`` contain the final primitive-to-surface eigenvectors and
        ``C_channel`` the finite-rho coefficients in ``channels``. With the
        directed primitive overlap

        P = <primitive(rho_surface)|primitive(rho_match)>,

        the surface-to-channel overlap and transformed LogD are

        T = C_surface^T P C_channel,

        Y_channel(E) = T^T Y_surface(E) T.

        This is the ``npp>n`` branch of ABC ``logder``. It is a pure basis
        transformation: no fictitious propagation from ``rho_surface`` to
        ``rho_match`` is performed. Normally the two radii are identical, but
        the directed overlap keeps the operation well-defined for a final
        surface centre distinct from the requested boundary.

    Inputs:
        basis: DelvesBasis - primitive Delves basis specification
        rho_surface: float - hyperradius of ``surface_coefficients`` in bohr
        surface_coefficients: ArrayLike - primitive-to-surface coefficients,
            shape ``(basis.n_primitive,n_surface)``
        Ymat: ArrayLike - final surface LogD matrix or energy batch, shape
            ``(...,n_surface,n_surface)``
        channels: DelvesAsymptoticBasis - final channel representation

    Returns:
        Y_channel: NDArray[np.float64] | NDArray[np.complex128] - transformed
            LogD matrix or batch, shape ``(...,n_channel,n_channel)``
    """
    surface = np.asarray(surface_coefficients, dtype=np.float64)
    Y = np.asarray(Ymat)
    if surface.ndim != 2 or surface.shape[0] != basis.n_primitive or surface.shape[1] < 1:
        message = f"surface_coefficients must have shape ({basis.n_primitive},n_surface), but got {surface.shape}"
        logger.error(message)
        raise ValueError(message)
    if Y.ndim < 2 or Y.shape[-2:] != (surface.shape[1], surface.shape[1]):
        message = f"Ymat must end with shape {(surface.shape[1], surface.shape[1])}, but got {Y.shape}"
        logger.error(message)
        raise ValueError(message)
    if channels.theta_coefficients.shape != (basis.n_primitive, channels.n_channel):
        message = "channels.theta_coefficients are incompatible with the supplied Delves basis"
        logger.error(message)
        raise ValueError(message)
    if not np.all(np.isfinite(surface)) or not np.all(np.isfinite(Y)):
        message = "surface_coefficients and Ymat must contain finite values"
        logger.error(message)
        raise ValueError(message)

    primitive_overlap = get_sector_overlap_delves(basis, rho_surface, channels.rho_match)
    transform = surface.T @ primitive_overlap @ channels.theta_coefficients
    result = np.einsum("pi,...pq,qj->...ij", transform, Y, transform, optimize=True)
    return np.asarray(result)
