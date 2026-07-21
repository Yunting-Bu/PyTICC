"""Atom-triatom interaction-matrix basis."""

from math import prod
from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.angle import norm_reduced_wigner_d, norm_YjK
from pyticc.basis.channel import ChannelBasis
from pyticc.basis.triatom import TriatomBasis, get_triatom_expansion
from pyticc.matrix.interaction import VBasisBF


# ----------------------------------------------------------------------------------------
def _triatom_grid_wavefunction(
    triatom: TriatomBasis,
    j: int,
    K: int,
    t: int,
    cos_theta_1: NDArray[np.float64],
    theta_weights_1: NDArray[np.float64],
    cos_theta_2: NDArray[np.float64],
    theta_weights_2: NDArray[np.float64],
    phi: NDArray[np.float64],
    phi_weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Evaluate one contracted triatomic channel function on its five-dimensional grid.

    The grid axes are ``(r1, r2, theta1, theta2, phi)``. Both returned arrays
    include square-root quadrature weights and have that five-dimensional shape.
    """
    if triatom.radial_1 is None or triatom.radial_2 is None:
        message = "Atom-triatom interaction preparation requires radial PODVR data in TriatomBasis"
        logger.error(message)
        raise ValueError(message)

    qn, coefficients = get_triatom_expansion(triatom, j, K, t)
    sqrt_weight_1 = np.sqrt(theta_weights_1)
    sqrt_weight_2 = np.sqrt(theta_weights_2)
    sqrt_weight_phi = np.sqrt(phi_weights)
    theta_2 = np.arccos(cos_theta_2)
    omega_values = tuple(sorted(int(value) for value in np.unique(qn[:, 1])))
    omega_positions = {omega: index for index, omega in enumerate(omega_values)}
    j1max = int(np.max(qn[:, 0]))
    n_v1 = triatom.radial_1.wavefunctions.shape[1]
    n_v2 = triatom.radial_2.wavefunctions.shape[1]

    coefficient_tensor = np.zeros((len(omega_values), j1max + 1, n_v1, n_v2), dtype=np.float64)
    for (j1, omega, v1, v2), coefficient in zip(qn, coefficients, strict=True):
        coefficient_tensor[omega_positions[int(omega)], int(j1), int(v1), int(v2)] = coefficient

    angular_1 = np.zeros((len(omega_values), cos_theta_1.size, j1max + 1), dtype=np.float64)
    angular_2 = np.empty((len(omega_values), cos_theta_2.size), dtype=np.float64)
    phi_real = np.empty((len(omega_values), phi.size), dtype=np.float64)
    phi_imag = np.empty_like(phi_real)
    for omega_index, omega in enumerate(omega_values):
        for j1 in range(abs(omega), j1max + 1):
            angular_1[omega_index, :, j1] = sqrt_weight_1 * np.asarray(norm_YjK(j1, omega, cos_theta_1), dtype=np.float64)
        angular_2[omega_index] = sqrt_weight_2 * np.asarray(
            norm_reduced_wigner_d(j, K, omega, theta_2),
            dtype=np.float64,
        )
        phi_real[omega_index] = sqrt_weight_phi * np.cos(omega * phi)
        phi_imag[omega_index] = sqrt_weight_phi * np.sin(omega * phi)

    internal = np.einsum(
        "ojvw,av,bw,ocj->oabc",
        coefficient_tensor,
        triatom.radial_1.wavefunctions,
        triatom.radial_2.wavefunctions,
        angular_1,
        optimize=True,
    )
    real = np.einsum("oabc,od,oe->abcde", internal, angular_2, phi_real, optimize=True)
    imag = np.einsum("oabc,od,oe->abcde", internal, angular_2, phi_imag, optimize=True)
    return real, imag


# ----------------------------------------------------------------------------------------
def prepare(
    basis: ChannelBasis,
    triatom: TriatomBasis,
    cos_theta_1: NDArray[np.float64],
    theta_weights_1: NDArray[np.float64],
    cos_theta_2: NDArray[np.float64],
    theta_weights_2: NDArray[np.float64],
    phi: NDArray[np.float64],
    phi_weights: NDArray[np.float64],
) -> VBasisBF:
    r"""
    Prepare the PES-independent body-fixed interaction basis for an atom-triatom system.

    Grid flattening follows C order with axes
    ``(r1, r2, theta1, theta2, phi)``. ``theta1`` is the triatomic bend,
    ``theta2`` orients its body axis relative to R, and ``phi`` is the dihedral
    angle on [0, pi].

    Formula:
        B^{R/I}_{t j K}(g) = sum_{j1,omega,v1,v2}
            T^{jK}_{j1 omega v1 v2,t} Phi_v1(r1) Phi_v2(r2)
            P_tilde^{omega}_{j1}(cos(theta1))
            d_tilde^{j}_{K,omega}(theta2)
            {cos,sin}(omega phi) sqrt(w1 w2 wphi),

        V_{cK,c'K'} = delta_{K,K'} / pi sum_g [
            B^R_c(g) V(g) B^R_c'(g) + B^I_c(g) V(g) B^I_c'(g)].

    Inputs:
        basis: ChannelBasis - complete atom-triatom channel basis
        triatom: TriatomBasis - contracted triatomic eigenstates and radial PODVR data
        cos_theta_1: NDArray[np.float64] - triatomic bend quadrature grids, shape
            (n_theta_1,)
        theta_weights_1: NDArray[np.float64] - bend quadrature weights, shape
            (n_theta_1,)
        cos_theta_2: NDArray[np.float64] - external polar quadrature grids, shape
            (n_theta_2,)
        theta_weights_2: NDArray[np.float64] - external polar weights, shape
            (n_theta_2,)
        phi: NDArray[np.float64] - dihedral quadrature grids on [0, pi], shape
            (n_phi,)
        phi_weights: NDArray[np.float64] - dihedral quadrature weights, shape
            (n_phi,)

    Returns:
        V_basis: VBasisBF - weighted bases whose arrays have shape
            (n_channel_K, n_r1 * n_r2 * n_theta_1 * n_theta_2 * n_phi)
    """
    if triatom.radial_1 is None or triatom.radial_2 is None:
        message = "Atom-triatom interaction preparation requires radial PODVR data in TriatomBasis"
        logger.error(message)
        raise ValueError(message)

    grid_shape = (
        triatom.radial_1.grids.size,
        triatom.radial_2.grids.size,
        cos_theta_1.size,
        cos_theta_2.size,
        phi.size,
    )
    n_grid = prod(grid_shape)
    channel_indices_by_K: dict[int, tuple[int, ...]] = {}
    B_real_by_K: dict[int, NDArray[np.float64]] = {}
    B_imag_by_K: dict[int, NDArray[np.float64]] = {}

    for K in sorted({channel.K for channel in basis}):
        channel_indices = tuple(index for index, channel in enumerate(basis) if channel.K == K)
        B_real = np.empty((len(channel_indices), n_grid), dtype=np.float64)
        B_imag = np.empty_like(B_real)

        for local_index, channel_index in enumerate(channel_indices):
            channel = basis[channel_index]
            triatom_state = channel.mis_X if channel.mis_X.t is not None else channel.mis_Y
            t = cast(int, triatom_state.t)
            real, imag = _triatom_grid_wavefunction(
                triatom,
                triatom_state.j,
                K,
                t,
                cos_theta_1,
                theta_weights_1,
                cos_theta_2,
                theta_weights_2,
                phi,
                phi_weights,
            )
            B_real[local_index] = real.reshape(-1)
            B_imag[local_index] = imag.reshape(-1)

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
