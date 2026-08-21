import numpy as np
from loguru import logger
from numpy.typing import ArrayLike, NDArray

from pyticc.basis.angle import norm_reduced_wigner_d, norm_YjK
from pyticc.basis.delves import (
    DelvesBasis,
    delves_angular_basis,
    delves_theta_basis,
    sine_basis,
    theta_max,
)
from pyticc.matrix.delves.geometry import asymptotic_potential, get_Vgrid_delves, mass_scale, transform_delves_coordinates
from pyticc.matrix.delves.hamiltonian import _get_Hmat_delves_grid
from pyticc.pes.total import TotalPES


# ----------------------------------------------------------------------------------------
def parity_rotation(
    Jtot: int,
    system_parity: int,
    K_a: int,
    K_b: int,
    beta_ab: ArrayLike,
) -> float | NDArray[np.float64]:
    r"""
    Evaluate ABC's parity-adapted body-axis rotation matrix element.

    Formula:
        ABC defines

        d^J_KM(beta) = <J K|exp(+i beta J_y)|J M>,

        D^{JP}_{KM}(beta)
            = N_K N_M [d^J_KM(beta)
              +P(-1)^(J+K)d^J_{-K,M}(beta)],

        N_0=1/sqrt(2),  N_K=1 for K>0.

        ``norm_reduced_wigner_d`` stores
        sqrt[(2J+1)/2] d^J_KM with the opposite exponential convention, so it
        is evaluated at ``-beta`` and divided by that normalization. Here
        K=K_a and M=K_b are non-negative helicities and beta is in radians.

    Inputs:
        Jtot: int - conserved total angular momentum J>=0
        system_parity: int - total parity P, -1 or 1
        K_a: int - source-arrangement non-negative helicity
        K_b: int - target-arrangement non-negative helicity
        beta_ab: ArrayLike - signed body-axis rotation angle, shape (...)

    Returns:
        value: float | NDArray[np.float64] - parity-adapted rotation element,
            with the same scalar/array form as beta_ab
    """
    if Jtot < 0 or K_a < 0 or K_b < 0 or K_a > Jtot or K_b > Jtot:
        message = f"Require Jtot>=0 and 0<=K_a,K_b<=Jtot, but got {(Jtot, K_a, K_b)}"
        logger.error(message)
        raise ValueError(message)
    if system_parity not in (-1, 1):
        message = f"system_parity must be -1 or 1, but got {system_parity}"
        logger.error(message)
        raise ValueError(message)
    angles = np.asarray(beta_ab, dtype=np.float64)
    if not np.all(np.isfinite(angles)):
        message = "beta_ab must contain finite angles"
        logger.error(message)
        raise ValueError(message)

    normalization = np.sqrt((2.0 * Jtot + 1.0) / 2.0)
    positive = np.asarray(norm_reduced_wigner_d(Jtot, K_a, K_b, -angles)) / normalization
    negative = np.asarray(norm_reduced_wigner_d(Jtot, -K_a, K_b, -angles)) / normalization
    value = positive + system_parity * (-1) ** (Jtot + K_a) * negative
    if K_a == 0:
        value *= np.sqrt(0.5)
    if K_b == 0:
        value *= np.sqrt(0.5)
    return float(value) if value.ndim == 0 else value


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def get_Smat_delves(
    basis: DelvesBasis,
    rho: float,
    arrangement_a: int,
    arrangement_b: int,
) -> NDArray[np.float64]:
    r"""
    Construct a directed primitive overlap block between two arrangements.

    The rows follow arrangement a's ``(j,K,n)`` labels and the columns follow
    arrangement b's labels, with consecutive sine indices n=1,...,n_sine.
    This is the uncontracted counterpart of ABC ``exchng``. It intentionally
    returns the one-sided quadrature result; a global surface-matrix builder
    should combine both directions before canonical orthogonalization.

    For exchange-symmetric A+B2 calculations, arrangement 3 uses arrangement
    2's stored quantum-number labels, as in ABC ``arrang``. Thus the special
    2-to-3 block can be evaluated even though arrangement 3 is not explicit.

    Formula:
        With x_a=cos(gamma_a), q and p denoting the midpoint-theta and
        Gauss--Legendre nodes in arrangement a,

        S^(ab)_{ja Ka n,jb Kb m}
          = sum_qp w_q w_p sin(2 theta_aq)/sin(2 theta_bqp)
            u_n(theta_aq) P_tilde_ja^Ka(x_ap)
            D^{JP}_{Ka Kb}(beta_ab,qp)
            A_ab(jb) P_tilde_jb^Kb(x_bqp) u_m(theta_bqp),

        retaining only theta_b < theta_max(rho). The coordinate transformation
        is supplied by ``transform_delves_coordinates``. The exchange factor is

        A_12=A_21=sqrt(2),
        A_23(j)=p(-1)^j,
        A_ab=1 otherwise,

        when exchange_parity p is nonzero. The normalized sine and associated
        Legendre functions use dtheta and d(cos gamma) measures, respectively.

    Inputs:
        basis: DelvesBasis - resolved primitive basis and quadrature sizes
        rho: float - positive hyperradius in bohr
        arrangement_a: int - one-based row/source arrangement, 1, 2, or 3
        arrangement_b: int - one-based column/target arrangement, 1, 2, or 3

    Returns:
        S_ab: NDArray[np.float64] - directed real overlap block with shape
            ``(n_qn_a*n_sine,n_qn_b*n_sine)``
    """
    qns_a = _arrangement_qns(basis, arrangement_a)
    _arrangement_qns(basis, arrangement_b)
    if arrangement_a == arrangement_b:
        return np.eye(len(qns_a) * basis.n_sine, dtype=np.float64)
    S_ab, _, _, _ = _cross_integrals(basis, rho, arrangement_a, arrangement_b)
    return S_ab


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def get_HSmat_delves(
    basis: DelvesBasis,
    total_pes: TotalPES,
    rho: float,
    arrangement_a: int,
    arrangement_b: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Construct directed primitive Hamiltonian and overlap blocks between arrangements.

    This is the uncontracted equivalent of ABC ``exchng`` followed by its
    ``eint(i)*s(i,j)`` update. Both matrices are evaluated on arrangement a's
    quadrature, so a global builder must combine the a-to-b and b-to-a results
    before solving the surface generalized eigenvalue problem.

    Formula:
        Split the fixed-rho surface Hamiltonian in source arrangement a as

        H = H_ref^(a) + [V_total-V_ref^(a)] + T_Cor^(a),

        V_ref^(a)(theta_a)
          = V_total[R_a^sc=100, r_a^sc=rho sin(theta_a), cos(gamma_a)=0].

        In the uncontracted sine basis the first cross-arrangement term is a
        matrix product, not ABC's diagonal contracted-basis expression:

        H_ab = H_ref^(a) S_ab + DeltaV_ab^(a) + C_ab^(a).

        ``H_ref^(a)`` contains the theta kinetic energy and the diagonal-K
        centrifugal terms, but no K-changing Coriolis terms. ``DeltaV_ab`` is
        the same transformed-grid integral as S_ab with an extra factor
        ``V_total(theta_a,x_a)-V_ref^(a)(theta_a)``.

        For a source row (j,K,n), the Coriolis contribution is

        C_ab = sum_{s=-1,+1} c_jK^s
               <a,j,K+s,n|[2 mu rho^2 cos^2(theta_a)]^-1|b>,

        where the label K+s applies only to the source associated Legendre
        function and body-axis rotation inside the integral; the sine index n
        remains the row's index. The coefficients are

        c_j0^+ = -sqrt[2J(J+1)j(j+1)],

        c_jK^+ = -sqrt{[J(J+1)-K(K+1)]
                        [j(j+1)-K(K+1)]},  K>0,

        c_j1^- = -sqrt[2J(J+1)j(j+1)],

        c_jK^- = -sqrt{[J(J+1)-K(K-1)]
                        [j(j+1)-K(K-1)]},  K>1.

        The transformed-coordinate Jacobian, exchange factors, primitive
        normalizations, and matrix ordering are identical to ``get_Smat_delves``.
        Atomic units are used, with masses in electron masses, rho in bohr, and
        H_ab in Hartree.

    Inputs:
        basis: DelvesBasis - resolved primitive basis and quadrature sizes
        total_pes: TotalPES - scalar adiabatic total PES in bohr and Hartree
        rho: float - positive hyperradius in bohr
        arrangement_a: int - one-based row/source arrangement, 1, 2, or 3
        arrangement_b: int - one-based column/target arrangement, 1, 2, or 3

    Returns:
        H_ab: NDArray[np.float64] - directed Hamiltonian block in Hartree,
            shape ``(n_qn_a*n_sine,n_qn_b*n_sine)``
        S_ab: NDArray[np.float64] - directed overlap block with the same shape
    """
    return _get_HSmat_delves_source(basis, total_pes, rho, arrangement_a, arrangement_b)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _get_HSmat_delves_source(
    basis: DelvesBasis,
    total_pes: TotalPES,
    rho: float,
    arrangement_a: int,
    arrangement_b: int,
    *,
    total_grid: NDArray[np.float64] | None = None,
    reference: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Construct one directed block, optionally reusing source PES grids."""
    if arrangement_a == arrangement_b:
        message = "get_HSmat_delves requires two distinct arrangements"
        logger.error(message)
        raise ValueError(message)

    S_ab, residual_ab, coriolis_ab, reference = _cross_integrals(
        basis,
        rho,
        arrangement_a,
        arrangement_b,
        total_pes,
        total_grid=total_grid,
        reference=reference,
    )
    qns_a = _arrangement_qns(basis, arrangement_a)
    theta_a, theta_weights, sine_a = delves_theta_basis(basis, rho)
    cos_gamma_a, gamma_weights, angular_values = delves_angular_basis(basis)
    assert reference is not None
    reference_grid = np.broadcast_to(reference[:, None], (theta_a.size, cos_gamma_a.size))
    H_reference = _get_Hmat_delves_grid(
        basis,
        rho,
        qns_a,
        theta_a,
        theta_weights,
        sine_a,
        gamma_weights,
        angular_values,
        reference_grid,
        coriolis=False,
    )
    return H_reference @ S_ab + residual_ab + coriolis_ab, S_ab


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _cross_integrals(
    basis: DelvesBasis,
    rho: float,
    arrangement_a: int,
    arrangement_b: int,
    total_pes: TotalPES | None = None,
    *,
    total_grid: NDArray[np.float64] | None = None,
    reference: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Evaluate directed overlap, residual-potential, and Coriolis integrals."""
    qns_a = _arrangement_qns(basis, arrangement_a)
    qns_b = _arrangement_qns(basis, arrangement_b)
    theta_a, theta_weights, sine_a = delves_theta_basis(basis, rho)
    cos_gamma_a, gamma_weights, _ = delves_angular_basis(basis)
    theta_b, cos_gamma_b, beta_ab = transform_delves_coordinates(theta_a[:, None], cos_gamma_a[None, :], arrangement_a, arrangement_b, basis.mass)
    theta_limit = float(theta_max(rho, basis.scaled_r_max))
    valid = theta_b < theta_limit
    denominator = np.sin(2.0 * theta_b)
    jacobian = np.divide(
        np.sin(2.0 * theta_a[:, None]),
        denominator,
        out=np.zeros_like(theta_b),
        where=valid & (np.abs(denominator) > np.finfo(np.float64).tiny),
    )
    jacobian *= theta_weights[:, None] * gamma_weights[None, :]
    sine_b = sine_basis(0.0, theta_limit, basis.n_sine, theta_b.ravel()).reshape(*theta_b.shape, basis.n_sine)
    source_angular_labels = set(qns_a)
    if total_pes is not None:
        source_angular_labels |= {(j, K + shift) for j, K in qns_a for shift in (-1, 1) if _coriolis_coefficient(basis, j, K, shift) != 0.0}
    angular_a = {(j, K): np.asarray(norm_YjK(j, K, cos_gamma_a)) for j, K in source_angular_labels}
    angular_b = {(j, K): np.asarray(norm_YjK(j, K, cos_gamma_b)) for j, K in qns_b}
    K_source = {K for _, K in qns_a}
    if total_pes is not None:
        K_source |= {K - 1 for j, K in qns_a if _coriolis_coefficient(basis, j, K, -1) != 0.0}
        K_source |= {K + 1 for j, K in qns_a if _coriolis_coefficient(basis, j, K, +1) != 0.0}
    rotations = {
        (K_a, K_b): np.asarray(parity_rotation(basis.Jtot, basis.system_parity, K_a, K_b, beta_ab))
        for K_a in K_source
        for K_b in {K for _, K in qns_b}
    }

    grid_shape = (len(qns_a) * basis.n_sine, len(qns_b) * basis.n_sine)
    S_ab = np.zeros(grid_shape, dtype=np.float64)
    residual_ab = np.zeros(grid_shape, dtype=np.float64)
    coriolis_ab = np.zeros(grid_shape, dtype=np.float64)
    if (total_grid is None) != (reference is None):
        message = "total_grid and reference must either both be supplied or both be omitted"
        logger.error(message)
        raise ValueError(message)
    residual = None
    radial_factor = 0.0
    if total_pes is not None or total_grid is not None:
        if total_grid is None:
            assert total_pes is not None
            total_grid = get_Vgrid_delves(total_pes, rho, arrangement_a, theta_a, cos_gamma_a, basis.mass)
            reference = asymptotic_potential(total_pes, basis.mass)(arrangement_a, rho * np.sin(theta_a))
        assert reference is not None
        expected_shape = (theta_a.size, cos_gamma_a.size)
        if total_grid.shape != expected_shape or reference.shape != theta_a.shape:
            message = f"reused source grids must have shapes {expected_shape} and {theta_a.shape}, but got {total_grid.shape} and {reference.shape}"
            logger.error(message)
            raise ValueError(message)
        residual = total_grid - reference[:, None]
        reduced_mass, _ = mass_scale(basis.mass)
        radial_factor = 1.0 / (2.0 * reduced_mass * rho**2)

    n_sine = basis.n_sine
    for index_a, (j_a, K_a) in enumerate(qns_a):
        row = slice(index_a * n_sine, (index_a + 1) * n_sine)
        for index_b, (j_b, K_b) in enumerate(qns_b):
            column = slice(index_b * n_sine, (index_b + 1) * n_sine)
            exchange = _exchange_factor(basis, arrangement_a, arrangement_b, j_b)
            target = angular_b[(j_b, K_b)] * rotations[(K_a, K_b)]
            kernel = exchange * jacobian * angular_a[(j_a, K_a)][None, :] * target
            S_ab[row, column] = np.einsum("qn,qpm,qp->nm", sine_a, sine_b, kernel, optimize=True)
            if residual is None:
                continue
            residual_ab[row, column] = np.einsum("qn,qpm,qp->nm", sine_a, sine_b, kernel * residual, optimize=True)
            for shift in (-1, 1):
                coefficient = _coriolis_coefficient(basis, j_a, K_a, shift)
                if coefficient == 0.0:
                    continue
                shifted_K = K_a + shift
                shifted_kernel = exchange * jacobian * angular_a[(j_a, shifted_K)][None, :] * angular_b[(j_b, K_b)]
                shifted_kernel *= rotations[(shifted_K, K_b)] / np.cos(theta_a[:, None]) ** 2
                coriolis_ab[row, column] += coefficient * radial_factor * np.einsum("qn,qpm,qp->nm", sine_a, sine_b, shifted_kernel, optimize=True)
    return S_ab, residual_ab, coriolis_ab, reference


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _coriolis_coefficient(basis: DelvesBasis, j: int, K: int, shift: int) -> float:
    """Return an allowed ABC non-negative-K Coriolis coefficient."""
    K_min = 0 if basis.system_parity == (-1) ** basis.Jtot else 1
    shifted_K = K + shift
    if shift not in (-1, 1) or shifted_K < K_min or shifted_K > min(j, basis.Jtot):
        return 0.0
    total_rotation = basis.Jtot * (basis.Jtot + 1)
    rotor = j * (j + 1)
    if (K, shift) in ((0, 1), (1, -1)):
        return float(-np.sqrt(2.0 * total_rotation * rotor))
    projection = K * shifted_K
    return float(-np.sqrt((total_rotation - projection) * (rotor - projection)))


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _arrangement_qns(basis: DelvesBasis, arrangement: int) -> tuple[tuple[int, int], ...]:
    """Return stored ``(j,K)`` labels, aliasing arrangement 3 when symmetric."""
    if arrangement not in (1, 2, 3):
        message = f"arrangement must be 1, 2, or 3, but got {arrangement}"
        logger.error(message)
        raise ValueError(message)
    stored_arrangement = 2 if arrangement == 3 and basis.exchange_parity != 0 else arrangement
    qns = tuple((j, K) for value_arrangement, j, K in basis.angular_qns if value_arrangement == stored_arrangement)
    if not qns:
        message = f"Delves basis contains no primitive functions for arrangement={arrangement}"
        logger.error(message)
        raise ValueError(message)
    return qns


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _exchange_factor(basis: DelvesBasis, arrangement_a: int, arrangement_b: int, j_b: int) -> float:
    """Return ABC's A+B2 arrangement normalization or exchange phase."""
    if basis.exchange_parity == 0:
        return 1.0
    if arrangement_a + arrangement_b == 3:
        return float(np.sqrt(2.0))
    if arrangement_a + arrangement_b == 5:
        return float(basis.exchange_parity * (-1) ** j_b)
    return 1.0


# ----------------------------------------------------------------------------------------
