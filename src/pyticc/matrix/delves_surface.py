import numpy as np
from loguru import logger
from numpy.typing import ArrayLike, NDArray

from pyticc.basis.delves import DelvesBasis, delves_angular_basis, delves_theta_basis
from pyticc.matrix.delves import asymptotic_potential, get_Vgrid_delves
from pyticc.matrix.delves_hamiltonian import _get_Hmat_delves_grid
from pyticc.matrix.delves_overlap import _get_HSmat_delves_source
from pyticc.pes.total import TotalPES


# ----------------------------------------------------------------------------------------
def get_surface_matrices_delves(
    basis: DelvesBasis,
    total_pes: TotalPES,
    rho: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Assemble the full multiple-arrangement primitive surface matrices.

    Matrix indices follow ``basis.angular_qns`` and then the consecutive sine
    index n=1,...,n_sine. Same-arrangement blocks use the direct primitive
    Hamiltonian. Each cross-arrangement block is evaluated in both coordinate
    directions and averaged, exactly as ABC symmetrizes the finite-quadrature
    matrices before canonical orthogonalization.

    Formula:
        For explicit arrangements a and b,

        H_aa = <a|H|a>,                 S_aa = I,

        H_ab = 1/2 [H_ab^(a-grid)+(H_ba^(b-grid))^T],

        S_ab = 1/2 [S_ab^(a-grid)+(S_ba^(b-grid))^T],

        H_ba = H_ab^T,                 S_ba = S_ab^T.

        ``get_HSmat_delves`` supplies each directed block, including

        H_ab^(a-grid) = H_ref^(a) S_ab + DeltaV_ab^(a) + C_ab^(a).

        When exchange_parity p=+/-1, only arrangements 1 and 2 are explicit.
        ABC's omitted exchange image is then added to arrangement 2:

        H_22 <- H_22 + 1/2(H_23+H_23^T),
        S_22 <- I    + 1/2(S_23+S_23^T),

        where the 2-to-3 integral contains the factor p(-1)^j for the target
        rotational state. No extra PES convention is introduced: ``total_pes``
        remains the total scalar adiabatic three-body potential in Hartree.

    Inputs:
        basis: DelvesBasis - resolved primitive multiple-arrangement basis
        total_pes: TotalPES - total PES using physical bonds in bohr and
            returning Hartree
        rho: float - positive hyperradius in bohr

    Returns:
        H: NDArray[np.float64] - real symmetric primitive surface Hamiltonian
            in Hartree, shape ``(basis.n_primitive,basis.n_primitive)``
        S: NDArray[np.float64] - real symmetric primitive overlap matrix with
            the same shape
    """
    if not np.isfinite(rho) or rho <= 0.0:
        message = f"rho must be finite and positive, but got {rho}"
        logger.error(message)
        raise ValueError(message)

    arrangements = tuple(dict.fromkeys(arrangement for arrangement, _, _ in basis.angular_qns))
    blocks = _arrangement_slices(basis, arrangements)
    H = np.zeros((basis.n_primitive, basis.n_primitive), dtype=np.float64)
    S = np.zeros_like(H)

    theta, theta_weights, sine_values = delves_theta_basis(basis, rho)
    cos_gamma, gamma_weights, angular_values = delves_angular_basis(basis)
    reference_potential = asymptotic_potential(total_pes, basis.mass)
    total_grids: dict[int, NDArray[np.float64]] = {}
    references: dict[int, NDArray[np.float64]] = {}

    for arrangement in arrangements:
        block = blocks[arrangement]
        qns = tuple((j, K) for value_arrangement, j, K in basis.angular_qns if value_arrangement == arrangement)
        total_grids[arrangement] = get_Vgrid_delves(total_pes, rho, arrangement, theta, cos_gamma, basis.mass)
        references[arrangement] = reference_potential(arrangement, rho * np.sin(theta))
        H[block, block] = _get_Hmat_delves_grid(
            basis,
            rho,
            qns,
            theta,
            theta_weights,
            sine_values,
            gamma_weights,
            angular_values,
            total_grids[arrangement],
            coriolis=True,
        )
        S[block, block] = np.eye(block.stop - block.start)

    for index_a, arrangement_a in enumerate(arrangements):
        for arrangement_b in arrangements[index_a + 1 :]:
            H_ab, S_ab = _get_HSmat_delves_source(
                basis,
                total_pes,
                rho,
                arrangement_a,
                arrangement_b,
                total_grid=total_grids[arrangement_a],
                reference=references[arrangement_a],
            )
            H_ba, S_ba = _get_HSmat_delves_source(
                basis,
                total_pes,
                rho,
                arrangement_b,
                arrangement_a,
                total_grid=total_grids[arrangement_b],
                reference=references[arrangement_b],
            )
            H_block = 0.5 * (H_ab + H_ba.T)
            S_block = 0.5 * (S_ab + S_ba.T)
            block_a = blocks[arrangement_a]
            block_b = blocks[arrangement_b]
            H[block_a, block_b] = H_block
            H[block_b, block_a] = H_block.T
            S[block_a, block_b] = S_block
            S[block_b, block_a] = S_block.T

    if basis.exchange_parity != 0:
        if arrangements != (1, 2):
            message = f"exchange-symmetric Delves basis must contain explicit arrangements (1,2), but got {arrangements}"
            logger.error(message)
            raise ValueError(message)
        H_23, S_23 = _get_HSmat_delves_source(
            basis,
            total_pes,
            rho,
            2,
            3,
            total_grid=total_grids[2],
            reference=references[2],
        )
        block_2 = blocks[2]
        H[block_2, block_2] += 0.5 * (H_23 + H_23.T)
        S[block_2, block_2] += 0.5 * (S_23 + S_23.T)

    return 0.5 * (H + H.T), 0.5 * (S + S.T)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def solve_surface_delves(
    H: ArrayLike,
    S: ArrayLike,
    *,
    overlap_cut: float = 1.0e-4,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Solve the Delves surface generalized eigenproblem by canonical orthogonalization.

    This implements ABC ``hcsevp`` without modifying the input matrices.
    Near-linear combinations of the multiple-arrangement primitive basis are
    removed using the same default absolute overlap threshold as ABC.

    Formula:
        For real symmetric H and S, diagonalize

        S U = U diag(sigma),       U^T U = I.

        Retain indices r for which sigma_r > overlap_cut and define

        X = U_r diag(sigma_r^-1/2).

        The orthonormal surface Hamiltonian and eigenproblem are

        H_orth = X^T H X,

        H_orth W = W diag(epsilon).

        Primitive-basis coefficients are

        C = X W,

        and obey

        H C = S C diag(epsilon),
        C^T S C = I.

        Eigenvalues and columns of C are returned in ascending energy order.
        ``sigma`` contains all overlap eigenvalues in ascending order so basis
        overcompleteness and the retained dimension can be inspected directly.

    Inputs:
        H: ArrayLike - real finite symmetric Hamiltonian in Hartree, shape (n,n)
        S: ArrayLike - real finite symmetric overlap matrix, shape (n,n)
        overlap_cut: float - positive absolute cutoff for retained overlap
            eigenvalues; ABC uses 1e-4

    Returns:
        energies: NDArray[np.float64] - retained surface energies in Hartree,
            shape (n_surface,)
        coefficients: NDArray[np.float64] - primitive-to-surface coefficients
            indexed as ``[primitive,surface]``, shape (n,n_surface)
        overlap_eigenvalues: NDArray[np.float64] - all eigenvalues of S in
            ascending order, shape (n,)
    """
    H_matrix = np.asarray(H, dtype=np.float64)
    S_matrix = np.asarray(S, dtype=np.float64)
    if H_matrix.ndim != 2 or H_matrix.shape[0] != H_matrix.shape[1]:
        message = f"H must be a square matrix, but got shape={H_matrix.shape}"
        logger.error(message)
        raise ValueError(message)
    if S_matrix.shape != H_matrix.shape:
        message = f"S must have the same square shape as H, but got H={H_matrix.shape}, S={S_matrix.shape}"
        logger.error(message)
        raise ValueError(message)
    if not np.all(np.isfinite(H_matrix)) or not np.all(np.isfinite(S_matrix)):
        message = "H and S must contain finite real values"
        logger.error(message)
        raise ValueError(message)
    if not np.allclose(H_matrix, H_matrix.T, rtol=1.0e-12, atol=1.0e-13):
        message = "H must be symmetric before canonical orthogonalization"
        logger.error(message)
        raise ValueError(message)
    if not np.allclose(S_matrix, S_matrix.T, rtol=1.0e-12, atol=1.0e-13):
        message = "S must be symmetric before canonical orthogonalization"
        logger.error(message)
        raise ValueError(message)
    if not np.isfinite(overlap_cut) or overlap_cut <= 0.0:
        message = f"overlap_cut must be finite and positive, but got {overlap_cut}"
        logger.error(message)
        raise ValueError(message)

    H_symmetric = 0.5 * (H_matrix + H_matrix.T)
    S_symmetric = 0.5 * (S_matrix + S_matrix.T)
    overlap_eigenvalues, overlap_vectors = np.linalg.eigh(S_symmetric)
    retained = overlap_eigenvalues > overlap_cut
    if not np.any(retained):
        message = f"No overlap eigenvalues exceed overlap_cut={overlap_cut}"
        logger.error(message)
        raise ValueError(message)

    transform = overlap_vectors[:, retained] / np.sqrt(overlap_eigenvalues[retained])[None, :]
    orthogonal_hamiltonian = transform.T @ H_symmetric @ transform
    energies, orthogonal_vectors = np.linalg.eigh(0.5 * (orthogonal_hamiltonian + orthogonal_hamiltonian.T))
    coefficients = transform @ orthogonal_vectors
    return energies, coefficients, overlap_eigenvalues


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _arrangement_slices(basis: DelvesBasis, arrangements: tuple[int, ...]) -> dict[int, slice]:
    """Return contiguous primitive slices for the stored arrangement ordering."""
    arrangement_labels = tuple(arrangement for arrangement, _, _ in basis.angular_qns)
    grouped_labels = tuple(arrangement for arrangement in arrangements for value in arrangement_labels if value == arrangement)
    if arrangement_labels != grouped_labels:
        message = "basis.angular_qns must be contiguous and ordered by arrangement"
        logger.error(message)
        raise ValueError(message)

    blocks: dict[int, slice] = {}
    start = 0
    for arrangement in arrangements:
        count = sum(value_arrangement == arrangement for value_arrangement, _, _ in basis.angular_qns) * basis.n_sine
        if count == 0:
            message = f"Delves basis contains no primitive functions for arrangement={arrangement}"
            logger.error(message)
            raise ValueError(message)
        blocks[arrangement] = slice(start, start + count)
        start += count
    if start != basis.n_primitive:
        message = "basis.angular_qns must be contiguous and ordered by arrangement"
        logger.error(message)
        raise ValueError(message)
    return blocks


# ----------------------------------------------------------------------------------------
