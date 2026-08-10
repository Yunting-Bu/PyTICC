"""Atom-diatom interaction-matrix basis."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.angle import norm_YjK
from pyticc.basis.channel import ChannelBasis, ChannelBasisElectricSF
from pyticc.basis.monomer.diatom_electric import DiatomElectricBasis, diatom_electric_amplitude
from pyticc.basis.podvr import RovibPODVR
from pyticc.matrix.interaction import VBasisBF


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AtomDiatomVBasisElectricSF:
    r"""
    Weighted SF basis and Jacobi-angle grid for an electric-field atom-diatom interaction.

    Formula:
        For a channel eta = (alpha,m,l,m_l) with M = m + m_l, define the
        tensor-grid index g = (p,q_r,q_R,q_delta) and

        Q_{eta,g}
          = A_{alpha m}(p,x_{r,q_r})
            P_tilde_l^{m_l}(x_{R,q_R})
            sqrt(w_{r,q_r} w_{R,q_R} w_{delta,q_delta}/(2 pi)).

        The two real weighted components are

        B^cos_{eta,g} = Q_{eta,g} cos(m delta_{q_delta}),

        B^sin_{eta,g} = Q_{eta,g} sin(m delta_{q_delta}).

        The PODVR radial index p is already a discrete orthonormal
        representation and therefore carries no additional quadrature weight.
        The Jacobi angle sampled by the PES is

        cos(gamma)
          = x_r x_R
            + sqrt(1-x_r^2) sqrt(1-x_R^2) cos(delta).

    Members:
        n_channel: int - number of SF channels
        grid_shape: tuple[int, int, int, int] - tensor-grid shape
            (n_podvr,n_theta_r,n_theta_R,n_delta)
        r: NDArray[np.float64] - PODVR bond-length nodes in bohr, shape
            (n_podvr,)
        gamma: NDArray[np.float64] - Jacobi angles in radians, shape
            (n_theta_r,n_theta_R,n_delta)
        B_cos: NDArray[np.float64] - cosine components proportional to
            cos(m delta), shape
            (n_channel,prod(grid_shape))
        B_sin: NDArray[np.float64] - sine components proportional to
            sin(m delta), shape
            (n_channel,prod(grid_shape))
    """

    n_channel: int
    grid_shape: tuple[int, int, int, int]
    r: NDArray[np.float64]
    gamma: NDArray[np.float64]
    B_cos: NDArray[np.float64]
    B_sin: NDArray[np.float64]


def build_AtomDiatomVBasisElectricSF(
    basis: ChannelBasisElectricSF,
    monomer_basis: DiatomElectricBasis,
    cos_theta_r: NDArray[np.float64],
    theta_r_weights: NDArray[np.float64],
    cos_theta_R: NDArray[np.float64],
    theta_R_weights: NDArray[np.float64],
    delta: NDArray[np.float64],
    delta_weights: NDArray[np.float64],
) -> AtomDiatomVBasisElectricSF:
    r"""
    Build the weighted SF interaction basis for an electric-field atom-diatom system.

    Formula:
        With x_r = cos(theta_r), x_R = cos(theta_R), and
        delta = phi_r - phi_R, the common azimuth Phi = phi_R integrates to
        2 pi because every channel satisfies M = m + m_l. Including the four
        spherical-harmonic normalization factors gives

        V_{eta' eta}^M(R)
          = (1/(2 pi)) sum_p
            integral_{-1}^{1} dx_r
            integral_{-1}^{1} dx_R
            integral_{0}^{2 pi} d delta
            A_{alpha'm'}(p,x_r) A_{alpha m}(p,x_r)
            P_tilde_{l'}^{m_l'}(x_R) P_tilde_l^{m_l}(x_R)
            cos[(m-m')delta]
            Delta V(R,r_p,gamma).

        The returned cosine and sine arrays use

        cos[(m-m')delta]
          = cos(m delta) cos(m' delta)
            + sin(m delta) sin(m' delta),

        and contain the square roots of all three angular quadrature weights
        and of 1/(2 pi). A half-interval delta rule is valid only when its
        weights represent the complete [0,2 pi) integral, for example by
        doubling paired weights.

    Inputs:
        basis: ChannelBasisElectricSF - complete fixed-M electric-field SF
            channel basis
        monomer_basis: DiatomElectricBasis - fixed-m dressed monomer states
        cos_theta_r: NDArray[np.float64] - x_r nodes in [-1,1], shape
            (n_theta_r,)
        theta_r_weights: NDArray[np.float64] - x_r quadrature weights, shape
            (n_theta_r,)
        cos_theta_R: NDArray[np.float64] - x_R nodes in [-1,1], shape
            (n_theta_R,)
        theta_R_weights: NDArray[np.float64] - x_R quadrature weights, shape
            (n_theta_R,)
        delta: NDArray[np.float64] - relative azimuth nodes in [0,2 pi],
            shape (n_delta,)
        delta_weights: NDArray[np.float64] - weights representing integration
            over the complete [0,2 pi) interval, shape (n_delta,)

    Returns:
        V_basis: AtomDiatomVBasisElectricSF - weighted basis and PES geometry grid
    """
    x_r = np.asarray(cos_theta_r, dtype=np.float64)
    w_r = np.asarray(theta_r_weights, dtype=np.float64)
    x_R = np.asarray(cos_theta_R, dtype=np.float64)
    w_R = np.asarray(theta_R_weights, dtype=np.float64)
    delta_values = np.asarray(delta, dtype=np.float64)
    w_delta = np.asarray(delta_weights, dtype=np.float64)

    amplitudes: dict[int, NDArray[np.float64]] = {}
    angular_R: dict[tuple[int, int], NDArray[np.float64]] = {}
    azimuthal: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    sqrt_w_R = np.sqrt(w_R)
    sqrt_w_delta = np.sqrt(w_delta / (2.0 * np.pi))
    grid_shape = (monomer_basis.grids.size, x_r.size, x_R.size, delta_values.size)
    n_grid = prod(grid_shape)
    B_cos = np.empty((basis.n_channel, n_grid), dtype=np.float64)
    B_sin = np.empty_like(B_cos)

    for channel in basis:
        block = monomer_basis.block(channel.m)

        if channel.m not in amplitudes:
            amplitudes[channel.m] = diatom_electric_amplitude(block, x_r, w_r)
        angular_key = (channel.l, channel.m_l)
        if angular_key not in angular_R:
            angular_R[angular_key] = sqrt_w_R * np.asarray(norm_YjK(channel.l, channel.m_l, x_R), dtype=np.float64)
        if channel.m not in azimuthal:
            azimuthal[channel.m] = (
                sqrt_w_delta * np.cos(channel.m * delta_values),
                sqrt_w_delta * np.sin(channel.m * delta_values),
            )

        radial_angular = amplitudes[channel.m][channel.alpha, :, :, None, None] * angular_R[angular_key][None, None, :, None]
        phase_cos, phase_sin = azimuthal[channel.m]
        B_cos[channel.index] = (radial_angular * phase_cos[None, None, None, :]).reshape(-1)
        B_sin[channel.index] = (radial_angular * phase_sin[None, None, None, :]).reshape(-1)

    sin_theta_r = np.sqrt(np.clip(1.0 - x_r**2, 0.0, None))
    sin_theta_R = np.sqrt(np.clip(1.0 - x_R**2, 0.0, None))
    cos_gamma = (
        x_r[:, None, None] * x_R[None, :, None] + sin_theta_r[:, None, None] * sin_theta_R[None, :, None] * np.cos(delta_values)[None, None, :]
    )
    gamma = np.arccos(np.clip(cos_gamma, -1.0, 1.0))
    return AtomDiatomVBasisElectricSF(
        n_channel=basis.n_channel,
        grid_shape=grid_shape,
        r=np.asarray(monomer_basis.grids, dtype=np.float64),
        gamma=gamma,
        B_cos=np.ascontiguousarray(B_cos),
        B_sin=np.ascontiguousarray(B_sin),
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def contract_electric_sf(
    V_basis: AtomDiatomVBasisElectricSF,
    potential: NDArray[np.float64],
    channel_indices: Sequence[int] | None = None,
) -> NDArray[np.float64]:
    r"""
    Contract a scalar interaction grid into the electric-field SF channel basis.

    Formula:
        For real Delta V(R,g), the SF matrix is evaluated entirely with real
        arrays:

        V_{eta' eta}(R)
          = sum_g Delta V(R,g)
            [B^cos_{eta',g} B^cos_{eta,g}
             + B^sin_{eta',g} B^sin_{eta,g}],

        using

        cos[(m-m')delta]
          = cos(m delta) cos(m' delta)
            + sin(m delta) sin(m' delta).

        Because Delta V depends on delta only through cos(gamma), the exact
        imaginary part vanishes under the complete relative-azimuth integral.

    Inputs:
        V_basis: AtomDiatomVBasisElectricSF - prepared weighted electric-field
            SF basis
        potential: NDArray[np.float64] - potential with shape grid_shape or
            (n_R,*grid_shape)
        channel_indices: Sequence[int] | None - optional unique complete-basis
            positions in the requested output order

    Returns:
        Vmat: NDArray[np.float64] - symmetric matrix with shape
            (n_selected,n_selected), optionally preceded by n_R
    """
    values = np.asarray(potential, dtype=np.float64)
    n_grid = prod(V_basis.grid_shape)
    if values.shape == V_basis.grid_shape:
        batches = values.reshape(1, n_grid)
        batched = False
    elif values.ndim == len(V_basis.grid_shape) + 1 and values.shape[1:] == V_basis.grid_shape:
        batches = values.reshape(values.shape[0], n_grid)
        batched = True
    else:
        message = f"Potential grid has shape {values.shape}, but SF basis requires {V_basis.grid_shape} with an optional leading R axis"
        logger.error(message)
        raise ValueError(message)
    indices = tuple(range(V_basis.n_channel)) if channel_indices is None else tuple(channel_indices)
    positions = np.asarray(indices, dtype=np.int64)
    B_cos = V_basis.B_cos[positions]
    B_sin = V_basis.B_sin[positions]
    Vmat = np.empty((batches.shape[0], len(indices), len(indices)), dtype=np.float64)
    for radial_index, values_at_R in enumerate(batches):
        matrix = (B_cos * values_at_R[None, :]) @ B_cos.T
        matrix += (B_sin * values_at_R[None, :]) @ B_sin.T
        Vmat[radial_index] = 0.5 * (matrix + matrix.T)
    return Vmat if batched else Vmat[0]


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def prepare(
    basis: ChannelBasis,
    rovib: RovibPODVR,
    cos_theta: NDArray[np.float64],
    theta_weights: NDArray[np.float64],
) -> VBasisBF:
    """Prepare the body-fixed interaction basis for an atom-diatom system."""
    grid_shape = (rovib.grids.size, cos_theta.size)
    n_grid = prod(grid_shape)
    sqrt_weight = np.sqrt(theta_weights)
    angular: dict[tuple[int, int], NDArray[np.float64]] = {}
    channel_indices: dict[int, tuple[int, ...]] = {}
    B_real: dict[int, NDArray[np.float64]] = {}

    for K in sorted({channel.K for channel in basis}):
        indices = tuple(index for index, channel in enumerate(basis) if channel.K == K)
        rows = np.empty((len(indices), n_grid), dtype=np.float64)
        for local_index, channel_index in enumerate(indices):
            channel = basis[channel_index]
            state = channel.mis_X if channel.mis_X.v is not None else channel.mis_Y
            v = cast(int, state.v)
            key = (state.j, K)
            if key not in angular:
                angular[key] = sqrt_weight * np.asarray(norm_YjK(state.j, K, cos_theta), dtype=np.float64)
            rows[local_index] = np.multiply.outer(rovib.WF_vj[:, v, state.j], angular[key]).reshape(-1)

        channel_indices[K] = indices
        B_real[K] = np.ascontiguousarray(rows)

    return VBasisBF(
        n_channel=basis.n_channel,
        grid_shape=grid_shape,
        channel_indices=channel_indices,
        B_real=B_real,
        B_imag=None,
    )


# ----------------------------------------------------------------------------------------
