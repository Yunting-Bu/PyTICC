"""Diatom-diatom interaction-matrix basis."""

from math import prod
from typing import cast

import numpy as np
from numpy.typing import NDArray

from pyticc.basis.angle import clebsch_gordan, norm_YjK
from pyticc.basis.channel import ChannelBasis
from pyticc.basis.rovib import RovibBasis
from pyticc.matrix.interaction import VBasisBF


# ----------------------------------------------------------------------------------------
def _angular(
    j_X: int,
    j_Y: int,
    j_couple: int,
    K: int,
    cos_theta_X: NDArray[np.float64],
    theta_weights_X: NDArray[np.float64],
    cos_theta_Y: NDArray[np.float64],
    theta_weights_Y: NDArray[np.float64],
    phi: NDArray[np.float64],
    phi_weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build weighted real and imaginary angular factors for one channel."""
    shape = (cos_theta_X.size, cos_theta_Y.size, phi.size)
    real = np.zeros(shape, dtype=np.float64)
    imag = np.zeros(shape, dtype=np.float64)
    sqrt_weight_X = np.sqrt(theta_weights_X)
    sqrt_weight_Y = np.sqrt(theta_weights_Y)
    sqrt_weight_phi = np.sqrt(phi_weights)
    for omega_X in range(-j_X, j_X + 1):
        omega_Y = K - omega_X
        if abs(omega_Y) > j_Y:
            continue
        coefficient = clebsch_gordan(j_X, omega_X, j_Y, omega_Y, j_couple)
        Y_X = sqrt_weight_X * np.asarray(norm_YjK(j_X, omega_X, cos_theta_X), dtype=np.float64)
        Y_Y = sqrt_weight_Y * np.asarray(norm_YjK(j_Y, omega_Y, cos_theta_Y), dtype=np.float64)
        amplitude = coefficient * Y_X[:, None, None] * Y_Y[None, :, None]
        real += amplitude * (sqrt_weight_phi * np.cos(omega_X * phi))[None, None, :]
        imag += amplitude * (sqrt_weight_phi * np.sin(omega_X * phi))[None, None, :]
    return real, imag


# ----------------------------------------------------------------------------------------
def prepare(
    basis: ChannelBasis,
    rovib_X: RovibBasis,
    rovib_Y: RovibBasis,
    cos_theta_X: NDArray[np.float64],
    theta_weights_X: NDArray[np.float64],
    cos_theta_Y: NDArray[np.float64],
    theta_weights_Y: NDArray[np.float64],
    phi: NDArray[np.float64],
    phi_weights: NDArray[np.float64],
) -> VBasisBF:
    r"""Prepare weighted BF rows, including optional molecule exchange.

    Formula:
        B_eta = T_eta.T B, separately for real and imaginary weighted rows.
        V_eta = (B_eta.real diag(V) B_eta.real.T
                 + B_eta.imag diag(V) B_eta.imag.T)/pi = T_eta.T V_BF T_eta.
        T_eta is the dimensionless, at-most-two-term channel expansion stored
        in basis.exchange. Exchange leaves K unchanged. Quadrature weights
        and radial wavefunctions are already included in B.

    Inputs:
        basis: ChannelBasis - parity-adapted channels, optionally exchange-adapted
        rovib_X, rovib_Y: RovibBasis - radial grids in bohr and wavefunctions
        cos_theta_X, cos_theta_Y: NDArray[np.float64] - polar cosine nodes
        theta_weights_X, theta_weights_Y: NDArray[np.float64] - polar weights
        phi, phi_weights: NDArray[np.float64] - torsional nodes on [0,pi] in
            radians and their quadrature weights

    Returns:
        V_basis: VBasisBF - weighted rows indexed by adapted channel and grid,
            grouped by K; contraction returns Hartree when V is in Hartree
    """
    grid_shape = (rovib_X.grids.size, rovib_Y.grids.size, cos_theta_X.size, cos_theta_Y.size, phi.size)
    n_grid = prod(grid_shape)
    angular: dict[tuple[int, int, int, int], tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    channel_indices: dict[int, tuple[int, ...]] = {}
    B_real: dict[int, NDArray[np.float64]] = {}
    B_imag: dict[int, NDArray[np.float64]] = {}
    source = tuple(basis) if basis.exchange is None else basis.exchange.source_channels

    for K in sorted({channel.K for channel in basis}):
        indices = tuple(index for index, channel in enumerate(basis) if channel.K == K)
        source_indices = tuple(index for index, channel in enumerate(source) if channel.K == K)
        real_rows = np.empty((len(source_indices), n_grid), dtype=np.float64)
        imag_rows = np.empty_like(real_rows)
        for local_index, channel_index in enumerate(source_indices):
            channel = source[channel_index]
            v_X = cast(int, channel.mis_X.v)
            v_Y = cast(int, channel.mis_Y.v)
            key = (channel.mis_X.j, channel.mis_Y.j, channel.j_couple, K)
            if key not in angular:
                angular[key] = _angular(
                    channel.mis_X.j,
                    channel.mis_Y.j,
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
            radial = np.multiply.outer(rovib_X.WF_vj[:, v_X, channel.mis_X.j], rovib_Y.WF_vj[:, v_Y, channel.mis_Y.j])
            real_rows[local_index] = (radial[:, :, None, None, None] * angular_real[None, None]).reshape(-1)
            imag_rows[local_index] = (radial[:, :, None, None, None] * angular_imag[None, None]).reshape(-1)

        if basis.exchange is not None:
            source_to_local = {index: local for local, index in enumerate(source_indices)}
            expansion = basis.exchange.source_indices[np.asarray(indices)]
            local = np.asarray([[source_to_local[int(i)] for i in pair] for pair in expansion], dtype=np.int64)
            coefficients = basis.exchange.coefficients[np.asarray(indices)]
            real_rows = np.sum(coefficients[:, :, None] * real_rows[local], axis=1)
            imag_rows = np.sum(coefficients[:, :, None] * imag_rows[local], axis=1)

        channel_indices[K] = indices
        B_real[K] = np.ascontiguousarray(real_rows)
        B_imag[K] = np.ascontiguousarray(imag_rows)

    return VBasisBF(
        n_channel=basis.n_channel,
        grid_shape=grid_shape,
        channel_indices=channel_indices,
        B_real=B_real,
        B_imag=B_imag,
        normalization=1.0 / np.pi,
    )


# ----------------------------------------------------------------------------------------
