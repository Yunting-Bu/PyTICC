from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.delves import DelvesBasis
from pyticc.matrix.delves_metric import get_sector_transform_delves
from pyticc.matrix.delves_reference import get_delves_reference_basis
from pyticc.matrix.delves_surface import get_surface_matrices_delves, solve_surface_delves
from pyticc.pes.total import TotalPES


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DelvesSurface:
    r"""
    Orthonormal adiabatic surface representation at one hyperradius.

    Formula:
        For a prepared ABC channel basis, finite-rho reference functions form
        the matrix ``B(rho)``.  Existing primitive matrices are contracted and
        the generalized eigenproblem is

        [B^T H B] c = [B^T S B] c diag(epsilon),

        C = B c,  C^T S C = I.

    Members:
        rho: float - surface hyperradius in bohr
        energies: NDArray[np.float64] - adiabatic surface energies in Hartree,
            shape ``(n_surface,)``
        coefficients: NDArray[np.float64] - primitive-to-surface coefficients,
            shape ``(n_primitive,n_surface)``
        overlap_eigenvalues: NDArray[np.float64] - reference-channel overlap
            eigenvalues before canonical truncation, shape ``(n_reference,)``
    """

    rho: float
    energies: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    overlap_eigenvalues: NDArray[np.float64]


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DelvesHamiltonian:
    r"""
    Reactive Delves Hamiltonian in a rho-dependent adiabatic surface basis.

    This object occupies the same architectural layer as ``ScattHamiltonian``.
    A conventional scattering Hamiltonian owns one fixed channel basis and
    evaluates ``H(R)`` directly. A Delves Hamiltonian keeps fixed channel labels
    but rebuilds their ABC ``sbasis(mode=0)`` representatives at every requested
    hyperradius, solves a generalized surface problem, and supplies the overlap
    transformation between two such surface representations.

    Formula:
        At hyperradius ``rho``,

        H(rho) C(rho) = S(rho) C(rho) diag(epsilon(rho)),

        C(rho)^T S(rho) C(rho) = I.

        For surfaces p and q, with directed primitive overlap
        ``P(rho_p,rho_q)``, the sector transformation is

        T_pq = C_p^T P(rho_p,rho_q) C_q.

    Members:
        basis: DelvesBasis - prepared channels and automatically resolved
            primitive Delves support
        total_potential: TotalPES - scalar total three-body PES in bohr and
            Hartree
        overlap_cut: float - canonical primitive-overlap eigenvalue cutoff
        energy_zero: float - native-PES energy subtracted from the total
            potential, in Hartree; zero means the native PES convention
    """

    basis: DelvesBasis
    total_potential: TotalPES
    overlap_cut: float = 1.0e-4
    energy_zero: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.basis, DelvesBasis):
            message = "DelvesHamiltonian requires a DelvesBasis"
            logger.error(message)
            raise TypeError(message)
        if not isinstance(self.total_potential, TotalPES):
            message = "DelvesHamiltonian requires a TotalPES"
            logger.error(message)
            raise TypeError(message)
        if not np.isfinite(self.energy_zero):
            message = f"energy_zero must be finite, but got {self.energy_zero}"
            logger.error(message)
            raise ValueError(message)
        if not np.isfinite(self.overlap_cut) or self.overlap_cut <= 0.0:
            message = f"overlap_cut must be finite and positive, but got {self.overlap_cut}"
            logger.error(message)
            raise ValueError(message)

    def surface(self, rho: float) -> DelvesSurface:
        r"""
        Solve the adiabatic surface eigenproblem at one hyperradius.

        Formula:
            Let ``B(rho)`` be the primitive-to-reference coefficient matrix.
            Canonical orthogonalization of the contracted matrices gives

            [B^T H B] c = [B^T S B] c diag(epsilon),

            C = B c,  C^T S C = I,

            retaining only reference-overlap eigenvalues greater than
            ``self.overlap_cut``.

        Inputs:
            rho: float - positive hyperradius in bohr
        Returns:
            surface: DelvesSurface - energies, coefficients, and overlap
                spectrum at the requested hyperradius
        """
        primitive_H, primitive_S = get_surface_matrices_delves(self.basis, self.total_potential, rho)
        if self.basis.n_channel:
            reference_coefficients, _ = get_delves_reference_basis(
                self.basis,
                self.total_potential,
                rho,
                asymptotic=False,
            )
            channel_H = reference_coefficients.T @ primitive_H @ reference_coefficients
            channel_S = reference_coefficients.T @ primitive_S @ reference_coefficients
            energies, contraction, overlap_eigenvalues = solve_surface_delves(
                channel_H,
                channel_S,
                overlap_cut=self.overlap_cut,
            )
            coefficients = reference_coefficients @ contraction
        else:
            energies, coefficients, overlap_eigenvalues = solve_surface_delves(
                primitive_H,
                primitive_S,
                overlap_cut=self.overlap_cut,
            )
        return DelvesSurface(
            rho=float(rho),
            energies=energies,
            coefficients=coefficients,
            overlap_eigenvalues=overlap_eigenvalues,
        )

    def transform(self, surface_a: DelvesSurface, surface_b: DelvesSurface) -> NDArray[np.float64]:
        r"""
        Construct the directed overlap between two surface representations.

        Formula:
            For primitive overlap ``P(rho_a,rho_b)``,

            T_ab = C_a^T P(rho_a,rho_b) C_b.

        Inputs:
            surface_a: DelvesSurface - source surface representation
            surface_b: DelvesSurface - target surface representation

        Returns:
            transform: NDArray[np.float64] - source-to-target overlap matrix,
                shape ``(n_surface_a,n_surface_b)``
        """
        return get_sector_transform_delves(
            self.basis,
            surface_a.rho,
            surface_a.coefficients,
            surface_b.rho,
            surface_b.coefficients,
        )


# ----------------------------------------------------------------------------------------
