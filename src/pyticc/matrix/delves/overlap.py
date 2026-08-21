import numpy as np
from loguru import logger
from numpy.typing import ArrayLike, NDArray

from pyticc.basis.angle import norm_YjK
from pyticc.basis.delves import DelvesBasis, delves_angular_basis, delves_theta_basis, sine_basis, theta_max
from pyticc.matrix.delves.coupling import _arrangement_qns, _exchange_factor, parity_rotation
from pyticc.matrix.delves.geometry import transform_delves_coordinates
from pyticc.matrix.delves.surface import _arrangement_slices


# ----------------------------------------------------------------------------------------
def get_sector_overlap_delves(
    basis: DelvesBasis,
    rho_a: float,
    rho_b: float,
) -> NDArray[np.float64]:
    r"""
    Construct the directed primitive overlap between two Delves sectors.

    Rows belong to the primitive basis at ``rho_a`` and columns to the basis at
    ``rho_b``. This implements ABC ``overlp``; unlike the fixed-rho surface
    overlap, the reverse quadrature is not averaged because the propagator
    requires a directed old-to-new sector transformation.

    Formula:
        Define

        theta_max^a = asin[min(1,scaled_r_max/rho_a)],
        theta_max^b = asin[min(1,scaled_r_max/rho_b)].

        Within one arrangement, angular labels are conserved and

        P^(aa)_{jKn,j'K'm}
          = delta_jj' delta_KK'
            integral_0^min(theta_max^a,theta_max^b)
            u_n(theta;rho_a)u_m(theta;rho_b)dtheta.

        The integral is evaluated on rho_a's midpoint grid, matching ABC
        ``overdi``. Between arrangements a!=b,

        P^(ab)_{ja Ka n,jb Kb m}
          = sum_qp w_q^a w_p sin(2theta_aq)/sin(2theta_bqp)
            u_n(theta_aq;rho_a) P_tilde_ja^Ka(x_ap)
            D^JP_KaKb(beta_ab,qp) A_ab(jb)
            P_tilde_jb^Kb(x_bqp)u_m(theta_bqp;rho_b),

        retaining only theta_bqp < theta_max^b. Coordinate transformation,
        parity rotation, normalized angular functions, and exchange factors
        are identical to the fixed-rho overlap.

        For exchange_parity p=+/-1, arrangement 3 is represented by
        arrangement 2's stored labels and its directed 2-to-3 contribution is
        added to the explicit (2,2) block, as in ABC ``overlp``.

    Inputs:
        basis: DelvesBasis - resolved primitive multiple-arrangement basis
        rho_a: float - positive old/source sector hyperradius in bohr
        rho_b: float - positive new/target sector hyperradius in bohr

    Returns:
        overlap: NDArray[np.float64] - directed primitive overlap indexed as
            ``[primitive_at_rho_a,primitive_at_rho_b]``, shape
            ``(basis.n_primitive,basis.n_primitive)``
    """
    _validate_rho(rho_a, "rho_a")
    _validate_rho(rho_b, "rho_b")
    arrangements = tuple(dict.fromkeys(arrangement for arrangement, _, _ in basis.angular_qns))
    blocks = _arrangement_slices(basis, arrangements)
    overlap = np.zeros((basis.n_primitive, basis.n_primitive), dtype=np.float64)

    for arrangement_a in arrangements:
        block_a = blocks[arrangement_a]
        for arrangement_b in arrangements:
            block_b = blocks[arrangement_b]
            if arrangement_a == arrangement_b:
                overlap[block_a, block_b] = _same_arrangement_overlap(basis, rho_a, rho_b, arrangement_a)
            else:
                overlap[block_a, block_b] = _cross_arrangement_overlap(basis, rho_a, rho_b, arrangement_a, arrangement_b)

    if basis.exchange_parity != 0:
        if arrangements != (1, 2):
            message = f"exchange-symmetric Delves basis must contain explicit arrangements (1,2), but got {arrangements}"
            logger.error(message)
            raise ValueError(message)
        block_2 = blocks[2]
        overlap[block_2, block_2] += _cross_arrangement_overlap(basis, rho_a, rho_b, 2, 3)
    return overlap


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def get_sector_transform_delves(
    basis: DelvesBasis,
    rho_a: float,
    coefficients_a: ArrayLike,
    rho_b: float,
    coefficients_b: ArrayLike,
) -> NDArray[np.float64]:
    r"""
    Transform between adjacent orthonormal Delves surface bases.

    Formula:
        Let C_a and C_b contain primitive-to-surface coefficients returned by
        ``solve_surface_delves`` at rho_a and rho_b. With the directed primitive
        overlap P(rho_a,rho_b),

        T(rho_a,rho_b) = C_a^T P(rho_a,rho_b) C_b.

        Consequently

        T_ij = <Phi_i(rho_a)|Phi_j(rho_b)>,

        and its shape is ``(n_surface_a,n_surface_b)``. The two dimensions may
        differ when canonical orthogonalization retains different numbers of
        overlap eigenvectors in adjacent sectors.

    Inputs:
        basis: DelvesBasis - resolved primitive multiple-arrangement basis
        rho_a: float - positive old/source sector hyperradius in bohr
        coefficients_a: ArrayLike - old primitive-to-surface coefficients,
            shape ``(basis.n_primitive,n_surface_a)``
        rho_b: float - positive new/target sector hyperradius in bohr
        coefficients_b: ArrayLike - new primitive-to-surface coefficients,
            shape ``(basis.n_primitive,n_surface_b)``

    Returns:
        transform: NDArray[np.float64] - old-to-new surface overlap, shape
            ``(n_surface_a,n_surface_b)``
    """
    C_a = _validate_coefficients(coefficients_a, basis.n_primitive, "coefficients_a")
    C_b = _validate_coefficients(coefficients_b, basis.n_primitive, "coefficients_b")
    primitive_overlap = get_sector_overlap_delves(basis, rho_a, rho_b)
    return C_a.T @ primitive_overlap @ C_b


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _same_arrangement_overlap(basis: DelvesBasis, rho_a: float, rho_b: float, arrangement: int) -> NDArray[np.float64]:
    """Return one same-arrangement sector-overlap block."""
    qns = _arrangement_qns(basis, arrangement)
    theta_a, weights_a, sine_a = delves_theta_basis(basis, rho_a)
    limit_b = float(theta_max(rho_b, basis.scaled_r_max))
    valid = theta_a < limit_b
    sine_b = sine_basis(0.0, limit_b, basis.n_sine, theta_a)
    radial_overlap = sine_a.T @ (weights_a[:, None] * valid[:, None] * sine_b)
    return np.asarray(np.kron(np.eye(len(qns)), radial_overlap), dtype=np.float64)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _cross_arrangement_overlap(
    basis: DelvesBasis,
    rho_a: float,
    rho_b: float,
    arrangement_a: int,
    arrangement_b: int,
) -> NDArray[np.float64]:
    """Return one directed cross-arrangement sector-overlap block."""
    qns_a = _arrangement_qns(basis, arrangement_a)
    qns_b = _arrangement_qns(basis, arrangement_b)
    theta_a, theta_weights, sine_a = delves_theta_basis(basis, rho_a)
    cos_gamma_a, gamma_weights, _ = delves_angular_basis(basis)
    theta_b, cos_gamma_b, beta_ab = transform_delves_coordinates(theta_a[:, None], cos_gamma_a[None, :], arrangement_a, arrangement_b, basis.mass)
    limit_b = float(theta_max(rho_b, basis.scaled_r_max))
    valid = theta_b < limit_b
    denominator = np.sin(2.0 * theta_b)
    jacobian = np.divide(
        np.sin(2.0 * theta_a[:, None]),
        denominator,
        out=np.zeros_like(theta_b),
        where=valid & (np.abs(denominator) > np.finfo(np.float64).tiny),
    )
    jacobian *= theta_weights[:, None] * gamma_weights[None, :]
    sine_b = sine_basis(0.0, limit_b, basis.n_sine, theta_b.ravel()).reshape(*theta_b.shape, basis.n_sine)
    angular_a = {(j, K): np.asarray(norm_YjK(j, K, cos_gamma_a)) for j, K in qns_a}
    angular_b = {(j, K): np.asarray(norm_YjK(j, K, cos_gamma_b)) for j, K in qns_b}
    rotations = {
        (K_a, K_b): np.asarray(parity_rotation(basis.Jtot, basis.system_parity, K_a, K_b, beta_ab))
        for K_a in {K for _, K in qns_a}
        for K_b in {K for _, K in qns_b}
    }

    n_sine = basis.n_sine
    overlap = np.zeros((len(qns_a) * n_sine, len(qns_b) * n_sine), dtype=np.float64)
    for index_a, (j_a, K_a) in enumerate(qns_a):
        row = slice(index_a * n_sine, (index_a + 1) * n_sine)
        for index_b, (j_b, K_b) in enumerate(qns_b):
            column = slice(index_b * n_sine, (index_b + 1) * n_sine)
            kernel = jacobian * angular_a[(j_a, K_a)][None, :] * angular_b[(j_b, K_b)] * rotations[(K_a, K_b)]
            kernel *= _exchange_factor(basis, arrangement_a, arrangement_b, j_b)
            overlap[row, column] = np.einsum("qn,qpm,qp->nm", sine_a, sine_b, kernel, optimize=True)
    return overlap


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _validate_rho(rho: float, name: str) -> None:
    """Validate one positive finite sector hyperradius."""
    if not np.isfinite(rho) or rho <= 0.0:
        message = f"{name} must be finite and positive, but got {rho}"
        logger.error(message)
        raise ValueError(message)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _validate_coefficients(values: ArrayLike, n_primitive: int, name: str) -> NDArray[np.float64]:
    """Return one finite primitive-to-surface coefficient matrix."""
    coefficients = np.asarray(values, dtype=np.float64)
    if coefficients.ndim != 2 or coefficients.shape[0] != n_primitive or coefficients.shape[1] < 1:
        message = f"{name} must have shape ({n_primitive},n_surface), but got {coefficients.shape}"
        logger.error(message)
        raise ValueError(message)
    if not np.all(np.isfinite(coefficients)):
        message = f"{name} must contain finite real values"
        logger.error(message)
        raise ValueError(message)
    return coefficients


# ----------------------------------------------------------------------------------------
