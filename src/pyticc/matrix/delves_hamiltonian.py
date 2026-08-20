import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.delves import DelvesBasis, delves_angular_basis, delves_theta_basis
from pyticc.matrix.delves import get_Vgrid_delves, mass_scale
from pyticc.pes.total import TotalPES


# ----------------------------------------------------------------------------------------
def get_Hmat_delves(
    basis: DelvesBasis,
    total_pes: TotalPES,
    rho: float,
    arrangement: int,
) -> NDArray[np.float64]:
    r"""
    Construct one arrangement's complete primitive Delves Hamiltonian.

    The matrix follows ``basis.angular_qns`` restricted to ``arrangement``;
    each ``(j,K)`` label contains consecutive sine indices n=1,...,n_sine.
    The total PES grid is evaluated once and reused for every K block.

    Formula:
        Fixed-K diagonal blocks are constructed by ``get_Hmat_delves_K``. The
        only off-diagonal helicity elements conserve j and connect K to K+1:

        <j,K,n|H|j,K+1,m>
            = C_jK <u_n|[2 mu rho^2 cos^2(theta)]^-1|u_m>,

        C_j0 = -sqrt[2 J(J+1) j(j+1)],

        C_jK = -sqrt{[J(J+1)-K(K+1)]
                     [j(j+1)-K(K+1)]},  K>0.

        These are the ABC ``cro(3)`` coefficients in the parity-adapted
        non-negative-K basis. Couplings absent from ``basis.angular_qns`` are
        not added.

    Inputs:
        basis: DelvesBasis - resolved primitive and quadrature specification;
            masses must be in atomic units
        total_pes: TotalPES - scalar adiabatic total PES in bohr and Hartree
        rho: float - positive hyperradius in bohr
        arrangement: int - one-based arrangement, 1, 2, or 3

    Returns:
        H: NDArray[np.float64] - real symmetric Hamiltonian in Hartree, shape
            ``(n_angular*n_sine,n_angular*n_sine)``
    """
    angular_qns = _delves_arrangement_qns(basis, arrangement)
    theta, theta_weights, sine_values = delves_theta_basis(basis, rho)
    cos_gamma, gamma_weights, angular_values = delves_angular_basis(basis)
    potential = get_Vgrid_delves(total_pes, rho, arrangement, theta, cos_gamma, basis.mass)
    return _get_Hmat_delves_grid(
        basis,
        rho,
        angular_qns,
        theta,
        theta_weights,
        sine_values,
        gamma_weights,
        angular_values,
        potential,
        coriolis=True,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _get_Hmat_delves_grid(
    basis: DelvesBasis,
    rho: float,
    angular_qns: tuple[tuple[int, int], ...],
    theta: NDArray[np.float64],
    theta_weights: NDArray[np.float64],
    sine_values: NDArray[np.float64],
    gamma_weights: NDArray[np.float64],
    angular_values: NDArray[np.float64],
    potential: NDArray[np.float64],
    *,
    coriolis: bool,
) -> NDArray[np.float64]:
    """Construct a same-coordinate primitive Hamiltonian from sampled grids."""
    n_sine = basis.n_sine
    H = np.zeros((len(angular_qns) * n_sine, len(angular_qns) * n_sine), dtype=np.float64)
    qn_position = {qn: index for index, qn in enumerate(angular_qns)}
    K_values = tuple(dict.fromkeys(K for _, K in angular_qns))
    for K in K_values:
        j_values = tuple(j for j, value_K in angular_qns if value_K == K)
        block = _get_Hmat_delves_K_grid(
            basis,
            rho,
            K,
            j_values,
            theta,
            theta_weights,
            sine_values,
            gamma_weights,
            angular_values,
            potential,
        )
        indices = np.concatenate([np.arange(qn_position[(j, K)] * n_sine, (qn_position[(j, K)] + 1) * n_sine) for j in j_values])
        H[np.ix_(indices, indices)] = block

    if not coriolis:
        return 0.5 * (H + H.T)

    reduced_mass, _ = mass_scale(basis.mass)
    radial_factor = 1.0 / (2.0 * reduced_mass * rho**2)
    inverse_cosine = sine_values.T @ (theta_weights[:, None] * sine_values / np.cos(theta[:, None]) ** 2)
    total_rotation = basis.Jtot * (basis.Jtot + 1)
    for j, K in angular_qns:
        upper_qn = (j, K + 1)
        if upper_qn not in qn_position:
            continue
        rotor = j * (j + 1)
        if K == 0:
            coefficient = -np.sqrt(2.0 * total_rotation * rotor)
        else:
            projection = K * (K + 1)
            coefficient = -np.sqrt((total_rotation - projection) * (rotor - projection))
        lower = slice(qn_position[(j, K)] * n_sine, (qn_position[(j, K)] + 1) * n_sine)
        upper = slice(qn_position[upper_qn] * n_sine, (qn_position[upper_qn] + 1) * n_sine)
        coupling = coefficient * radial_factor * inverse_cosine
        H[lower, upper] = coupling
        H[upper, lower] = coupling.T

    return 0.5 * (H + H.T)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def get_Hmat_delves_K(
    basis: DelvesBasis,
    total_pes: TotalPES,
    rho: float,
    arrangement: int,
    K: int,
) -> NDArray[np.float64]:
    r"""
    Construct one arrangement's fixed-K primitive Delves Hamiltonian block.

    No local vibrational contraction is applied. Matrix indices are ordered as
    ``(j,n)``: the allowed j values for ``(arrangement,K)`` are taken in the
    order present in ``basis.angular_qns``, and each j contains sine indices
    n=1,...,n_sine.

    Formula:
        For primitive functions

        Phi_njK(theta,x) = u_n(theta;rho) P_tilde_j^K(x),

        with x=cos(gamma), the fixed-K Hamiltonian is

        H = T_theta
            + [J(J+1)+j(j+1)-2K^2]/(2 mu rho^2 cos^2(theta))
            + j(j+1)/(2 mu rho^2 sin^2(theta))
            + V_total(r_AB,r_BC,r_CA),

        where mu=sqrt(m_A m_B m_C/(m_A+m_B+m_C)) and hbar=1. Thus

        <n|T_theta|m> = delta_nm
            {[(n pi/theta_max)^2-1/4]/(2 mu rho^2)}.

        The remaining terms are evaluated with the theta midpoint and
        x Gauss--Legendre rules stored in ``basis``. The full total PES is used
        directly; no asymptotic reference potential is subtracted.

    Inputs:
        basis: DelvesBasis - resolved primitive and quadrature specification;
            masses must be in atomic units
        total_pes: TotalPES - scalar adiabatic total PES in bohr and Hartree
        rho: float - positive hyperradius in bohr
        arrangement: int - one-based arrangement, 1, 2, or 3
        K: int - non-negative body-fixed helicity for this block

    Returns:
        H: NDArray[np.float64] - real symmetric Hamiltonian in Hartree, shape
            ``(n_j*n_sine,n_j*n_sine)``, indexed by flattened ``(j,n)``
    """
    j_values = _delves_K_j_values(basis, arrangement, K)
    theta, theta_weights, sine_values = delves_theta_basis(basis, rho)
    cos_gamma, gamma_weights, angular_values = delves_angular_basis(basis)
    potential = get_Vgrid_delves(total_pes, rho, arrangement, theta, cos_gamma, basis.mass)
    return _get_Hmat_delves_K_grid(
        basis,
        rho,
        K,
        j_values,
        theta,
        theta_weights,
        sine_values,
        gamma_weights,
        angular_values,
        potential,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _delves_K_j_values(basis: DelvesBasis, arrangement: int, K: int) -> tuple[int, ...]:
    """Return ordered j values represented by one arrangement and K block."""
    if arrangement not in (1, 2, 3):
        message = f"arrangement must be 1, 2, or 3, but got {arrangement}"
        logger.error(message)
        raise ValueError(message)
    if K < 0:
        message = f"K must be non-negative, but got {K}"
        logger.error(message)
        raise ValueError(message)
    j_values = tuple(dict.fromkeys(j for value_arrangement, j, value_K in basis.angular_qns if value_arrangement == arrangement and value_K == K))
    if not j_values:
        message = f"Delves basis contains no primitive functions for arrangement={arrangement}, K={K}"
        logger.error(message)
        raise ValueError(message)
    return j_values


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _delves_arrangement_qns(basis: DelvesBasis, arrangement: int) -> tuple[tuple[int, int], ...]:
    """Return ordered ``(j,K)`` labels represented by one arrangement."""
    if arrangement not in (1, 2, 3):
        message = f"arrangement must be 1, 2, or 3, but got {arrangement}"
        logger.error(message)
        raise ValueError(message)
    qns = tuple((j, K) for value_arrangement, j, K in basis.angular_qns if value_arrangement == arrangement)
    if not qns:
        message = f"Delves basis contains no primitive functions for arrangement={arrangement}"
        logger.error(message)
        raise ValueError(message)
    return qns


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _get_Hmat_delves_K_grid(
    basis: DelvesBasis,
    rho: float,
    K: int,
    j_values: tuple[int, ...],
    theta: NDArray[np.float64],
    theta_weights: NDArray[np.float64],
    sine_values: NDArray[np.float64],
    gamma_weights: NDArray[np.float64],
    angular_values: NDArray[np.float64],
    potential: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Construct one fixed-K block from a previously evaluated PES grid."""
    reduced_mass, _ = mass_scale(basis.mass)
    radial_factor = 1.0 / (2.0 * reduced_mass * rho**2)
    inverse_cosine = sine_values.T @ (theta_weights[:, None] * sine_values / np.cos(theta[:, None]) ** 2)
    inverse_sine = sine_values.T @ (theta_weights[:, None] * sine_values / np.sin(theta[:, None]) ** 2)
    angular = angular_values[:, j_values, K]
    angular_potential = np.einsum("pj,qp,pk,p->qjk", angular, potential, angular, gamma_weights, optimize=True)
    potential_matrix = np.einsum(
        "qn,qjk,qm,q->jnkm",
        sine_values,
        angular_potential,
        sine_values,
        theta_weights,
        optimize=True,
    )

    n_sine = basis.n_sine
    H = potential_matrix.reshape(len(j_values) * n_sine, len(j_values) * n_sine)
    theta_limit = float(np.sum(theta_weights))
    modes = np.arange(1, n_sine + 1, dtype=np.float64)
    kinetic = ((np.pi * modes / theta_limit) ** 2 - 0.25) * radial_factor
    total_rotation = basis.Jtot * (basis.Jtot + 1)
    for j_index, j in enumerate(j_values):
        block = slice(j_index * n_sine, (j_index + 1) * n_sine)
        rotor = j * (j + 1)
        H[block, block] += np.diag(kinetic)
        H[block, block] += radial_factor * (total_rotation + rotor - 2 * K**2) * inverse_cosine
        H[block, block] += radial_factor * rotor * inverse_sine

    return 0.5 * (H + H.T)


# ----------------------------------------------------------------------------------------
