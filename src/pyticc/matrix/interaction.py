from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.angle import clebsch_gordan, norm_YjK
from pyticc.basis.channel import ChannelBasis
from pyticc.basis.podvr import RovibPODVR


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class VBasisBF:
    """
    Precomputed internal basis for a body-fixed interaction matrix.

    Members:
        n_channel: int - number of channels in the complete ChannelBasis
        grid_shape: tuple[int, ...] - tensor-product internal-grid shape
        channel_indices: dict[int, tuple[int, ...]] - complete-basis positions grouped by exact K
        B_real: dict[int, NDArray[np.float64]] - weighted real basis grouped by
            exact K; each value has shape (n_channel_K, n_grid)
        B_imag: dict[int, NDArray[np.float64]] | None - weighted imaginary basis;
            each value has shape (n_channel_K, n_grid)
        normalization: float - normalization applied after grid contraction
    """

    n_channel: int
    grid_shape: tuple[int, ...]
    channel_indices: dict[int, tuple[int, ...]]
    B_real: dict[int, NDArray[np.float64]]
    B_imag: dict[int, NDArray[np.float64]] | None
    normalization: float = 1.0


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def prepare_Vmat_BF_atom_diatom(
    basis: ChannelBasis,
    rovib: RovibPODVR,
    cos_theta: NDArray[np.float64],
    theta_weights: NDArray[np.float64],
) -> VBasisBF:
    r"""
    Prepare the PES-independent body-fixed interaction basis for an atom-diatom system.

    ``cos_theta`` and ``theta_weights`` are Gauss-Legendre grids and weights on [-1, 1].
    Grid flattening follows C order with axes ``(r, theta)``.

    Formula:
        For channel c = (v, j, K) and grid g = (p, l),

        B^K_{c g} = Phi_{v j}(r_p) sqrt(w_l) P_tilde^K_j(cos(theta_l)),

        where Phi is ``rovib.WF_vj`` and P_tilde is the normalized associated
        Legendre function returned by ``norm_YjK``.

    Inputs:
        basis: ChannelBasis - complete atom-diatom channel basis
        rovib: RovibPODVR - diatomic PODVR rovibrational basis
        cos_theta: NDArray[np.float64] - cos(theta) quadrature grids, shape
            (n_theta,)
        theta_weights: NDArray[np.float64] - Gauss-Legendre quadrature weights,
            shape (n_theta,)

    Returns:
        V_basis: VBasisBF - weighted bases whose arrays have shape
            (n_channel_K, n_r * n_theta)
    """
    grid_shape = (rovib.grids.size, cos_theta.size)
    n_grid = prod(grid_shape)
    sqrt_weight = np.sqrt(theta_weights)
    # dict[(j, K): Y_jK]
    angular: dict[tuple[int, int], NDArray[np.float64]] = {}
    channel_indices_by_K: dict[int, tuple[int, ...]] = {}
    B_real_by_K: dict[int, NDArray[np.float64]] = {}

    for K in sorted({channel.K for channel in basis}):
        channel_indices = tuple(index for index, channel in enumerate(basis) if channel.K == K)
        B_real = np.empty((len(channel_indices), n_grid), dtype=np.float64)

        for local_index, channel_index in enumerate(channel_indices):
            channel = basis[channel_index]
            diatom_state = channel.mis_X if channel.mis_X.v is not None else channel.mis_Y
            # cause v can be None if mis.X is atom
            v = cast(int, diatom_state.v)
            j = diatom_state.j
            key = (j, K)
            if key not in angular:
                angular[key] = sqrt_weight * np.asarray(norm_YjK(j, K, cos_theta), dtype=np.float64)

            radial = rovib.WF_vj[:, v, j]
            B_real[local_index] = np.multiply.outer(radial, angular[key]).reshape(-1)

        channel_indices_by_K[K] = channel_indices
        B_real_by_K[K] = np.ascontiguousarray(B_real)

    return VBasisBF(
        n_channel=basis.n_channel,
        grid_shape=grid_shape,
        channel_indices=channel_indices_by_K,
        B_real=B_real_by_K,
        B_imag=None,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _angular_diatom_diatom(
    j1: int,
    j2: int,
    j_couple: int,
    K: int,
    cos_theta_1: NDArray[np.float64],
    theta_weights_1: NDArray[np.float64],
    cos_theta_2: NDArray[np.float64],
    theta_weights_2: NDArray[np.float64],
    phi: NDArray[np.float64],
    phi_weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Build weighted real and imaginary angular factors for one coupled (j1, j2, j, K).

    Inputs:
        cos_theta_1: NDArray[np.float64] - first polar grid, shape (n_theta_1,)
        theta_weights_1: NDArray[np.float64] - first polar weights, shape
            (n_theta_1,)
        cos_theta_2: NDArray[np.float64] - second polar grid, shape (n_theta_2,)
        theta_weights_2: NDArray[np.float64] - second polar weights, shape
            (n_theta_2,)
        phi: NDArray[np.float64] - dihedral grid, shape (n_phi,)
        phi_weights: NDArray[np.float64] - dihedral weights, shape (n_phi,)

    Returns:
        angular_real: NDArray[np.float64] - cosine contribution, shape
            (n_theta_1, n_theta_2, n_phi)
        angular_imag: NDArray[np.float64] - sine contribution, shape
            (n_theta_1, n_theta_2, n_phi)
    """
    shape = (cos_theta_1.size, cos_theta_2.size, phi.size)
    angular_real = np.zeros(shape, dtype=np.float64)
    angular_imag = np.zeros(shape, dtype=np.float64)
    sqrt_weight_1 = np.sqrt(theta_weights_1)
    sqrt_weight_2 = np.sqrt(theta_weights_2)
    sqrt_weight_phi = np.sqrt(phi_weights)

    for omega_1 in range(-j1, j1 + 1):
        omega_2 = K - omega_1
        if abs(omega_2) > j2:
            continue

        coefficient = clebsch_gordan(j1, omega_1, j2, omega_2, j_couple)
        Y1 = sqrt_weight_1 * np.asarray(norm_YjK(j1, omega_1, cos_theta_1), dtype=np.float64)
        Y2 = sqrt_weight_2 * np.asarray(norm_YjK(j2, omega_2, cos_theta_2), dtype=np.float64)
        amplitude = coefficient * Y1[:, None, None] * Y2[None, :, None]
        angular_real += amplitude * (sqrt_weight_phi * np.cos(omega_1 * phi))[None, None, :]
        angular_imag += amplitude * (sqrt_weight_phi * np.sin(omega_1 * phi))[None, None, :]

    return angular_real, angular_imag


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def prepare_Vmat_BF_diatom_diatom(
    basis: ChannelBasis,
    rovib_X: RovibPODVR,
    rovib_Y: RovibPODVR,
    cos_theta_X: NDArray[np.float64],
    theta_weights_X: NDArray[np.float64],
    cos_theta_Y: NDArray[np.float64],
    theta_weights_Y: NDArray[np.float64],
    phi: NDArray[np.float64],
    phi_weights: NDArray[np.float64],
) -> VBasisBF:
    r"""
    Prepare the PES-independent body-fixed interaction basis for a diatom-diatom system.

    Polar quadratures are Gauss-Legendre grids in cos(theta). The phi quadrature is
    defined on [0, pi]. Grid flattening follows C order with axes
    ``(r_X, r_Y, theta_X, theta_Y, phi)``.

    Formula:
        For omega_Y = K - omega_X, the real and imaginary angular bases are

        A^R = sum_{omega_X} CG(j_X omega_X, j_Y omega_Y | j_couple K)
              sqrt(w_X w_Y w_phi) P_tilde^{omega_X}_{j_X}(cos(theta_X))
              P_tilde^{omega_Y}_{j_Y}(cos(theta_Y)) cos(omega_X phi),

        A^I = sum_{omega_X} CG(j_X omega_X, j_Y omega_Y | j_couple K)
              sqrt(w_X w_Y w_phi) P_tilde^{omega_X}_{j_X}(cos(theta_X))
              P_tilde^{omega_Y}_{j_Y}(cos(theta_Y)) sin(omega_X phi),

        B^{R/I}_{c g} = Phi^X_{v_X j_X}(r_X) Phi^Y_{v_Y j_Y}(r_Y) A^{R/I}.

    Inputs:
        basis: ChannelBasis - complete diatom-diatom channel basis
        rovib_X: RovibPODVR - first diatomic rovibrational basis
        rovib_Y: RovibPODVR - second diatomic rovibrational basis
        cos_theta_X: NDArray[np.float64] - cos(theta_X) quadrature grids, shape
            (n_theta_X,)
        theta_weights_X: NDArray[np.float64] - theta_X quadrature weights, shape
            (n_theta_X,)
        cos_theta_Y: NDArray[np.float64] - cos(theta_Y) quadrature grids, shape
            (n_theta_Y,)
        theta_weights_Y: NDArray[np.float64] - theta_Y quadrature weights, shape
            (n_theta_Y,)
        phi: NDArray[np.float64] - dihedral-angle quadrature grids on [0, pi],
            shape (n_phi,)
        phi_weights: NDArray[np.float64] - dihedral-angle quadrature weights,
            shape (n_phi,)

    Returns:
        V_basis: VBasisBF - weighted bases whose arrays have shape
            (n_channel_K, n_r_X * n_r_Y * n_theta_X * n_theta_Y * n_phi)
    """
    grid_shape = (rovib_X.grids.size, rovib_Y.grids.size, cos_theta_X.size, cos_theta_Y.size, phi.size)
    n_grid = prod(grid_shape)
    angular: dict[tuple[int, int, int, int], tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    channel_indices_by_K: dict[int, tuple[int, ...]] = {}
    B_real_by_K: dict[int, NDArray[np.float64]] = {}
    B_imag_by_K: dict[int, NDArray[np.float64]] = {}

    for K in sorted({channel.K for channel in basis}):
        channel_indices = tuple(index for index, channel in enumerate(basis) if channel.K == K)
        B_real = np.empty((len(channel_indices), n_grid), dtype=np.float64)
        B_imag = np.empty_like(B_real)

        for local_index, channel_index in enumerate(channel_indices):
            channel = basis[channel_index]
            v_X = cast(int, channel.mis_X.v)
            v_Y = cast(int, channel.mis_Y.v)
            j_X = channel.mis_X.j
            j_Y = channel.mis_Y.j
            key = (j_X, j_Y, channel.j_couple, K)
            if key not in angular:
                angular[key] = _angular_diatom_diatom(
                    j_X,
                    j_Y,
                    channel.j_couple,
                    K,
                    cos_theta_X,
                    theta_weights_X,
                    cos_theta_Y,
                    theta_weights_Y,
                    phi,
                    phi_weights,
                )

            angular_real, angular_imag = angular[key]
            radial = np.multiply.outer(rovib_X.WF_vj[:, v_X, j_X], rovib_Y.WF_vj[:, v_Y, j_Y])
            B_real[local_index] = (radial[:, :, None, None, None] * angular_real[None, None, :, :, :]).reshape(-1)
            B_imag[local_index] = (radial[:, :, None, None, None] * angular_imag[None, None, :, :, :]).reshape(-1)

        channel_indices_by_K[K] = channel_indices
        B_real_by_K[K] = np.ascontiguousarray(B_real)
        B_imag_by_K[K] = np.ascontiguousarray(B_imag)

    return VBasisBF(
        n_channel=basis.n_channel,
        grid_shape=grid_shape,
        channel_indices=channel_indices_by_K,
        B_real=B_real_by_K,
        B_imag=B_imag_by_K,
        normalization=1.0 / np.pi,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _contract_basis(
    B_real: NDArray[np.float64],
    B_imag: NDArray[np.float64] | None,
    normalization: float,
    potential_flat: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Contract one exact-K basis with a flattened potential grid.

    Inputs:
        B_real: NDArray[np.float64] - real weighted basis, shape
            (n_channel_K, n_grid)
        B_imag: NDArray[np.float64] | None - imaginary weighted basis, shape
            (n_channel_K, n_grid)
        normalization: float - final quadrature normalization
        potential_flat: NDArray[np.float64] - potential values, shape (n_grid,)

    Returns:
        Vmat: NDArray[np.float64] - symmetric interaction block, shape
            (n_channel_K, n_channel_K)
    """
    weighted = B_real * potential_flat[None, :]
    Vmat = weighted @ B_real.T
    if B_imag is not None:
        weighted = B_imag * potential_flat[None, :]
        Vmat += weighted @ B_imag.T
    Vmat *= normalization
    return 0.5 * (Vmat + Vmat.T)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def get_Vmat_BF(
    V_basis: VBasisBF,
    potential_flat: NDArray[np.float64],
    channel_indices: Sequence[int] | None = None,
) -> NDArray[np.float64]:
    r"""
    Get body-fixed interaction matrices from potential values on internal grids.

    The potential is diagonal in exact K. ``channel_indices`` can select a complete,
    CS, or NNCC propagation block while preserving the requested channel order.

    Formula:
        V_{cK,c'K'} = delta_{K,K'} N sum_g [
            B^R_{c g} V_g B^R_{c' g} + B^I_{c g} V_g B^I_{c' g}
        ],

        where N = 1 for atom-diatom and N = 1/pi for atom-triatom or
        diatom-diatom. The imaginary contribution is absent when
        ``V_basis.B_imag`` is None.

    Inputs:
        V_basis: VBasisBF - precomputed atom-diatom, atom-triatom, or
            diatom-diatom basis
        potential_flat: NDArray[np.float64] - one potential grid with shape
            grid_shape or (n_grid,), or a batch with shape (n_R, *grid_shape) or
            (n_R, n_grid)
        channel_indices: Sequence[int] | None - requested complete-basis positions,
            shape (n_selected_channel,)

    Returns:
        Vmat: NDArray[np.float64] - interaction matrix with shape
            (n_selected_channel, n_selected_channel), or batched matrices with shape
            (n_R, n_selected_channel, n_selected_channel)
    """
    if channel_indices is None:
        indices = tuple(range(V_basis.n_channel))
    else:
        indices = tuple(channel_indices)

    n_grid = prod(V_basis.grid_shape)
    potential = np.asarray(potential_flat, dtype=np.float64)
    if potential.shape == V_basis.grid_shape or potential.shape == (n_grid,):
        potential_batch = potential.reshape(1, n_grid)
        batched = False
    elif potential.ndim == len(V_basis.grid_shape) + 1 and potential.shape[1:] == V_basis.grid_shape:
        potential_batch = potential.reshape(potential.shape[0], n_grid)
        batched = True
    elif potential.ndim == 2 and potential.shape[1] == n_grid:
        potential_batch = potential.reshape(potential.shape[0], n_grid)
        batched = True
    else:
        message = f"Potential grid has shape {potential.shape}, but V basis requires {V_basis.grid_shape} with an optional leading R axis"
        logger.error(message)
        raise ValueError(message)

    global_to_local = {global_index: local_index for local_index, global_index in enumerate(indices)}
    Vmat = np.zeros((potential_batch.shape[0], len(indices), len(indices)), dtype=np.float64)

    for K, K_channel_indices in V_basis.channel_indices.items():
        selected = tuple(
            (block_index, global_to_local[global_index])
            for block_index, global_index in enumerate(K_channel_indices)
            if global_index in global_to_local
        )
        if not selected:
            continue

        block_positions, output_positions = zip(*selected, strict=True)
        B_real = V_basis.B_real[K][np.asarray(block_positions)]
        B_imag = None if V_basis.B_imag is None else V_basis.B_imag[K][np.asarray(block_positions)]
        for radial_index, potential_row in enumerate(potential_batch):
            block_Vmat = _contract_basis(B_real, B_imag, V_basis.normalization, potential_row)
            Vmat[radial_index][np.ix_(output_positions, output_positions)] = block_Vmat

    return Vmat if batched else Vmat[0]


# ----------------------------------------------------------------------------------------
