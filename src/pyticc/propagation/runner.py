from collections.abc import Callable, Sequence
from typing import Literal, cast

import jax
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.channel import ChannelBasis
from pyticc.energy import EnergyInput, get_Etot
from pyticc.matrix.centrifugal import get_Umat_BF
from pyticc.matrix.radial import get_Wmat
from pyticc.propagation.grid import build_radial_sectors
from pyticc.propagation.logd import initialize_logD_capture, initialize_logD_inelastic, propagate_logD


# ----------------------------------------------------------------------------------------
def propagate_BF(
    basis: ChannelBasis,
    Vmat: Callable[[float], NDArray[np.float64]] | Callable[[NDArray[np.float64]], NDArray[np.float64]],
    Etot: EnergyInput,
    reduced_mass: float,
    radial_boundaries: Sequence[float],
    radial_half_steps: Sequence[float],
    mode: Literal["inelastic", "capture"] = "inelastic",
    channel_indices: Sequence[int] | None = None,
    batch_Vmat: bool = False,
) -> jax.Array:
    r"""
    Propagate the body-fixed log-derivative matrix with the LDMD method.

    The interaction callback is evaluated once at each distinct radial point, or once
    for the complete radial array when ``batch_Vmat`` is true.
    Radial sectors, centrifugal matrices, energy-dependent radial matrices, and
    the initial log-derivative matrices are constructed internally.

    Formula:
        W(R; Etot) = U / R**2
                     + 2 * reduced_mass * [V(R) + diag(E_int - Etot)].

    Inputs:
        basis: ChannelBasis - complete field-free channel basis
        Vmat: Callable - scalar or batched interaction matrix evaluated at R
        Etot: EnergyInput - total-energy array or one-column text file in atomic units
        reduced_mass: float - collision reduced mass in atomic units
        radial_boundaries: Sequence[float] - increasing radial interval boundaries in atomic units
        radial_half_steps: Sequence[float] - nominal LDMD half-step for each radial interval
        mode: Literal["inelastic", "capture"] - inner-boundary condition
        channel_indices: Sequence[int] | None - complete-basis positions for one propagation block
        batch_Vmat: bool - evaluate all distinct radial interaction matrices in one call

    Returns:
        Y_final: jax.Array - final log-derivative matrices with shape (n_energy, n_channel, n_channel)
    """
    if mode not in ("inelastic", "capture"):
        message = f"mode must be 'inelastic' or 'capture', but got {mode!r}"
        logger.error(message)
        raise ValueError(message)

    energies = get_Etot(Etot)
    sectors = build_radial_sectors(radial_boundaries, radial_half_steps)
    indices = tuple(range(basis.n_channel)) if channel_indices is None else tuple(channel_indices)
    if not indices:
        message = "At least one channel is required for propagation"
        logger.error(message)
        raise ValueError(message)

    E_int = basis.E_int[np.asarray(indices)]
    Umat = get_Umat_BF(basis, indices)
    radial_starts = np.asarray([sector.radial_start for sector in sectors], dtype=np.float64)
    radial_mids = np.asarray([sector.radial_mid for sector in sectors], dtype=np.float64)
    radial_ends = np.asarray([sector.radial_end for sector in sectors], dtype=np.float64)
    sector_half_steps = np.asarray([sector.radial_half_step for sector in sectors], dtype=np.float64)

    W_base: dict[float, NDArray[np.float64]] = {}
    radial_points = np.unique(np.concatenate((radial_starts, radial_mids, radial_ends)))
    if batch_Vmat:
        batched_callback = cast(Callable[[NDArray[np.float64]], NDArray[np.float64]], Vmat)
        interactions = np.asarray(batched_callback(radial_points), dtype=np.float64)
        expected_shape = (radial_points.size, len(indices), len(indices))
        if interactions.shape != expected_shape:
            message = f"Batched Vmat returned shape {interactions.shape}, but expected {expected_shape}"
            logger.error(message)
            raise ValueError(message)
    else:
        scalar_callback = cast(Callable[[float], NDArray[np.float64]], Vmat)
        interactions = np.stack([np.asarray(scalar_callback(float(radial_point)), dtype=np.float64) for radial_point in radial_points])

    for radial_point, interaction in zip(radial_points, interactions, strict=True):
        radial_value = float(radial_point)
        W_base[radial_value] = get_Wmat(radial_value, 0.0, reduced_mass, E_int, Umat, interaction)

    W_base_start = np.stack([W_base[float(radial_value)] for radial_value in radial_starts])
    W_base_mid = np.stack([W_base[float(radial_value)] for radial_value in radial_mids])
    W_base_end = np.stack([W_base[float(radial_value)] for radial_value in radial_ends])

    n_channel = len(indices)
    identity = np.eye(n_channel, dtype=np.float64)
    W_initial = W_base_start[0][None, :, :] - 2.0 * reduced_mass * energies[:, None, None] * identity
    if mode == "inelastic":
        Y_initial = initialize_logD_inelastic(W_initial)
    else:
        Y_initial = initialize_logD_capture(W_initial)

    return propagate_logD(
        Y_initial,
        energies,
        reduced_mass,
        sector_half_steps,
        W_base_start,
        W_base_mid,
        W_base_end,
    )


# ----------------------------------------------------------------------------------------
