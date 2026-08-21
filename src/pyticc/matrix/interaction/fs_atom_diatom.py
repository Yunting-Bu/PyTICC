from dataclasses import dataclass
from math import prod

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.angle import norm_reduced_wigner_d_half
from pyticc.fine_structure.basis import FSState
from pyticc.fine_structure.channel import FSChannelBasis


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FSVBasis:
    """
    Precontracted quadrature kernels for V_sum and V_dif.

    Members:
        n_channel: int - number of fine-structure channels
        grid_shape: tuple[int,int] - PODVR and angular grid shape
        pair_rows: NDArray[np.int64] - lower-triangle row indices, shape (n_pair,)
        pair_columns: NDArray[np.int64] - lower-triangle column indices, shape
            (n_pair,)
        kernel: NDArray[np.float64] - flattened V_sum/V_dif lower-triangle
            contraction operator, shape (2*n_grid,n_pair)
    """

    n_channel: int
    grid_shape: tuple[int, int]
    pair_rows: NDArray[np.int64]
    pair_columns: NDArray[np.int64]
    kernel: NDArray[np.float64]


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def prepare(
    basis: FSChannelBasis,
    theta: NDArray[np.float64],
    theta_weights: NDArray[np.float64],
) -> FSVBasis:
    r"""
    Prepare Gaussian-quadrature kernels in the monomer fine-structure eigenbasis.

    Formula:
        For channel c=(v,j,tau,epsilon,K), its signed primitive amplitude is

        A_{c,a}(r_p,theta_q) = C_{a,tau} chi_v(r_p)
          sqrt(w_q) sqrt((2j+1)/2) d^j_{K,Omega_a}(theta_q).

        The two kernels are

        G_sum(c',c;g) = sum_{a',a} A_{c',a'} A_{c,a}
          delta_{Sigma'Sigma} delta_{Lambda'Lambda},

        G_dif(c',c;g) = sum_{a',a} A_{c',a'} A_{c,a}
          delta_{Sigma'Sigma} delta_{Lambda',-Lambda}.

        K is conserved by the interaction. Radial PODVR contraction is discrete
        and carries no additional quadrature weight.

    Inputs:
        basis: FSChannelBasis - parity-adapted molecular eigenchannels
        theta: NDArray[np.float64] - angular quadrature nodes in radians
        theta_weights: NDArray[np.float64] - weights for integration over cos(theta)

    Returns:
        V_basis: FSVBasis - reusable V_sum and V_dif kernels
    """
    angles = np.asarray(theta, dtype=np.float64)
    weights = np.asarray(theta_weights, dtype=np.float64)
    vib = basis.monomer.vib
    n_grid = vib.grids.size * angles.size
    amplitudes: list[list[tuple[FSState, NDArray[np.float64]]]] = []
    for channel in basis:
        block = basis.monomer.blocks[channel.block]
        signed_coefficients = block.transform @ block.coefficients[:, channel.tau]
        channel_amplitudes: list[tuple[FSState, NDArray[np.float64]]] = []
        for state, coefficient in zip(block.primitive_states, signed_coefficients, strict=True):
            if abs(coefficient) < 1.0e-15:
                continue
            angular = np.sqrt(weights) * np.asarray(norm_reduced_wigner_d_half(state.two_j, channel.two_K, state.two_omega, angles), dtype=np.float64)
            radial = vib.wavefunctions[:, state.v]
            channel_amplitudes.append((state, coefficient * np.multiply.outer(radial, angular).reshape(n_grid)))
        amplitudes.append(channel_amplitudes)

    kernel_sum = np.zeros((basis.n_channel, basis.n_channel, n_grid), dtype=np.float64)
    kernel_dif = np.zeros_like(kernel_sum)
    for row, bra_channel in enumerate(basis):
        for column in range(row + 1):
            ket_channel = basis[column]
            if bra_channel.two_K != ket_channel.two_K:
                continue
            for bra, bra_amplitude in amplitudes[row]:
                for ket, ket_amplitude in amplitudes[column]:
                    if bra.two_sigma != ket.two_sigma:
                        continue
                    product = bra_amplitude * ket_amplitude
                    if bra.two_lambda == ket.two_lambda:
                        kernel_sum[row, column] += product
                    if bra.two_lambda == -ket.two_lambda:
                        kernel_dif[row, column] += product
            kernel_sum[column, row] = kernel_sum[row, column]
            kernel_dif[column, row] = kernel_dif[row, column]
    pair_rows, pair_columns = np.tril_indices(basis.n_channel)
    pair_kernels = np.stack((kernel_sum[pair_rows, pair_columns], kernel_dif[pair_rows, pair_columns]), axis=-1)
    kernel = pair_kernels.reshape(pair_rows.size, 2 * n_grid).T.copy()
    return FSVBasis(basis.n_channel, (vib.grids.size, angles.size), pair_rows, pair_columns, kernel)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def contract(V_basis: FSVBasis, potential: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""
    Contract V_sum and V_dif grids into fine-structure channel matrices.

    Formula:
        V_{c'c}(R) = sum_g [G_sum(c',c;g)V_sum(R,g)
                            +G_dif(c',c;g)V_dif(R,g)].

    Inputs:
        V_basis: FSVBasis - prepared quadrature kernels
        potential: NDArray[np.float64] - shape (*grid_shape,2) or
            (n_R,*grid_shape,2)

    Returns:
        matrix: NDArray[np.float64] - channel matrix, optionally batched over R
    """
    values = np.asarray(potential, dtype=np.float64)
    if values.shape == (*V_basis.grid_shape, 2):
        batches = values.reshape(1, prod(V_basis.grid_shape), 2)
        batched = False
    elif values.ndim == 4 and values.shape[1:] == (*V_basis.grid_shape, 2):
        batches = values.reshape(values.shape[0], prod(V_basis.grid_shape), 2)
        batched = True
    else:
        message = f"FS PES grid has shape {values.shape}, expected {(*V_basis.grid_shape, 2)} with optional leading R axis"
        logger.error(message)
        raise ValueError(message)
    potential_vectors = batches.reshape(batches.shape[0], -1)
    contracted = potential_vectors @ V_basis.kernel
    matrix = np.empty((batches.shape[0], V_basis.n_channel, V_basis.n_channel), dtype=np.float64)
    matrix[:, V_basis.pair_rows, V_basis.pair_columns] = contracted
    matrix[:, V_basis.pair_columns, V_basis.pair_rows] = contracted
    return matrix if batched else matrix[0]


# ----------------------------------------------------------------------------------------
