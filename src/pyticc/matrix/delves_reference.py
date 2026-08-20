import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.delves import (
    DelvesBasis,
    clean_sine_phases,
    delves_theta_basis,
    sine_kinetic,
    sine_reference_hamiltonian,
)
from pyticc.matrix.delves import asymptotic_potential, mass_scale
from pyticc.pes.total import TotalPES


# ----------------------------------------------------------------------------------------
def get_delves_reference_basis(
    basis: DelvesBasis,
    total_pes: TotalPES,
    rho: float,
    *,
    asymptotic: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Construct ABC ``sbasis`` functions for the preselected reaction channels.

    ``asymptotic=False`` implements ``sbasis(mode=0)`` and includes the
    K-dependent end-over-end term used inside every propagation sector.
    ``asymptotic=True`` implements ``sbasis(mode=1)`` at the final matching
    boundary and omits that term.

    Formula:
        For channel ``Q=(a,v,j,K)`` and

        u_n(theta;rho)=sqrt(2/theta_max) sin(n pi theta/theta_max),

        the reference Hamiltonian is

        H_nm^Q(rho) = delta_nm
          {[(n pi/theta_max)^2-1/4]/(2 mu rho^2)}

          + integral u_n(theta)
            {j(j+1)/[2 mu rho^2 sin^2(theta)]
             + delta_mode [J(J+1)+j(j+1)-2K^2]
               /[2 mu rho^2 cos^2(theta)]
             + V_a^asym[rho sin(theta)]}u_m(theta) dtheta,

        where ``delta_mode=1`` for the surface basis and zero for the final
        asymptotic basis.  Eigenvector ``v`` is placed in the primitive block
        ``(a,j,K,n)``.

    Inputs:
        basis: DelvesBasis - preselected channels and primitive specification
        total_pes: TotalPES - energy-zero-adjusted total three-body PES
        rho: float - positive hyperradius in bohr
        asymptotic: bool - select ABC ``mode=1`` instead of ``mode=0``

    Returns:
        coefficients: NDArray[np.float64] - primitive-to-reference-channel
            coefficients, shape ``(basis.n_primitive,basis.n_channel)``
        energies: NDArray[np.float64] - finite-rho reference energies aligned
            with ``basis.qns``, shape ``(basis.n_channel,)``
    """
    if basis.n_channel < 1:
        message = "ABC reference functions require preselected Delves channels"
        logger.error(message)
        raise ValueError(message)
    if not np.isfinite(rho) or rho <= 0.0:
        message = f"rho must be finite and positive, but got {rho}"
        logger.error(message)
        raise ValueError(message)

    reduced_mass, _ = mass_scale(basis.mass)
    theta, weights, values = delves_theta_basis(basis, rho)
    theta_limit = float(np.sum(weights))
    kinetic = sine_kinetic(basis.n_sine, theta_limit, reduced_mass * rho**2, theta=True)
    scaled_r = rho * np.sin(theta)
    reference = asymptotic_potential(total_pes, basis.mass)
    radial_factor = 1.0 / (2.0 * reduced_mass * rho**2)
    angular_positions = {label: index for index, label in enumerate(basis.angular_qns)}
    coefficients = np.zeros((basis.n_primitive, basis.n_channel), dtype=np.float64)
    energies = np.empty(basis.n_channel, dtype=np.float64)
    solved: dict[tuple[int, int, int | None], tuple[NDArray[np.float64], NDArray[np.float64]]] = {}

    for channel_index, (arrangement, v, j, K) in enumerate(basis.qns):
        key = (arrangement, j, None if asymptotic else K)
        if key not in solved:
            effective = reference(arrangement, scaled_r)
            effective = effective + radial_factor * j * (j + 1) / np.sin(theta) ** 2
            if not asymptotic:
                effective = effective + radial_factor * (basis.Jtot * (basis.Jtot + 1) + j * (j + 1) - 2 * K**2) / np.cos(theta) ** 2
            hamiltonian = sine_reference_hamiltonian(values, weights, kinetic, effective)
            eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
            solved[key] = eigenvalues, clean_sine_phases(eigenvectors, values)

        eigenvalues, eigenvectors = solved[key]
        if v >= eigenvalues.size:
            message = f"Channel {(arrangement, v, j, K)} exceeds the finite-rho sine basis dimension {eigenvalues.size}"
            logger.error(message)
            raise ValueError(message)
        primitive_position = angular_positions[(arrangement, j, K)]
        block = slice(primitive_position * basis.n_sine, (primitive_position + 1) * basis.n_sine)
        coefficients[block, channel_index] = eigenvectors[:, v]
        energies[channel_index] = eigenvalues[v]

    return coefficients, energies


# ----------------------------------------------------------------------------------------
