from __future__ import annotations

from collections.abc import Callable, Sequence
from math import ceil
from time import perf_counter
from typing import TYPE_CHECKING, Literal, cast

import jax
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.channel import ChannelBasis
from pyticc.energy import EnergyInput, get_Etot
from pyticc.matrix.centrifugal import get_Umat_BF
from pyticc.matrix.radial import get_Wmat
from pyticc.propagation.device import resolve_device
from pyticc.propagation.grid import build_radial_sectors, iter_radial_windows
from pyticc.propagation.logd import initialize_logD_capture, initialize_logD_inelastic, propagate_logD

if TYPE_CHECKING:
    from pyticc.scattering.hamiltonian import ScattHamiltonian

VmatCallback = Callable[[float], NDArray[np.float64]] | Callable[[NDArray[np.float64]], NDArray[np.float64]]
InteractionProvider = Callable[[NDArray[np.float64], tuple[tuple[int, ...], ...]], tuple[NDArray[np.float64], ...]]


# ----------------------------------------------------------------------------------------
def propagate(
    hamiltonian: ScattHamiltonian,
    Etot: EnergyInput,
    radial_boundaries: Sequence[float],
    radial_half_steps: Sequence[float],
    mode: Literal["inelastic", "capture"] = "inelastic",
    memory_limit_mb: float = 512.0,
    print_verbose: bool = False,
    device: str = "auto",
) -> jax.Array:
    """Propagate one scattering Hamiltonian to the final radial boundary.

    Inputs:
        hamiltonian: ScattHamiltonian - projected channel Hamiltonian
        Etot: EnergyInput - total energies in atomic units
        radial_boundaries: Sequence[float] - radial interval boundaries
        radial_half_steps: Sequence[float] - LDMD half-step for every interval
        mode: Literal["inelastic", "capture"] - inner-boundary condition
        memory_limit_mb: float - target transient-memory limit in MiB
        print_verbose: bool - whether to emit INFO-level propagation progress

    Returns:
        Y_final: jax.Array - final body-fixed log-derivative matrices
    """
    return propagate_BF(
        basis=hamiltonian.basis,
        Vmat=hamiltonian.interaction,
        Etot=Etot,
        reduced_mass=hamiltonian.reduced_mass,
        radial_boundaries=radial_boundaries,
        radial_half_steps=radial_half_steps,
        mode=mode,
        batch_Vmat=hamiltonian.batched,
        memory_limit_mb=memory_limit_mb,
        potential_grid_size=hamiltonian.potential_grid_size,
        print_verbose=print_verbose,
        device=device,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def propagate_BF_blocks(
    basis: ChannelBasis,
    channel_blocks: Sequence[Sequence[int]],
    interaction_provider: InteractionProvider,
    energies: NDArray[np.float64],
    reduced_mass: float,
    radial_boundaries: Sequence[float],
    radial_half_steps: Sequence[float],
    mode: Literal["inelastic", "capture"],
    memory_limit_mb: float,
    potential_grid_size: int,
    print_verbose: bool = False,
    device: str = "auto",
) -> tuple[jax.Array, ...]:
    """
    Propagate one or more body-fixed channel blocks through shared radial windows.

    The provider evaluates only new radial points. The interaction matrix at each
    window boundary is retained and reused by the next window. All blocks therefore
    share the same radial traversal while keeping independent log derivatives.

    Inputs:
        basis: ChannelBasis - complete field-free channel basis
        channel_blocks: Sequence[Sequence[int]] - complete-basis positions for each
            propagation block, each with shape (n_channel_block,)
        interaction_provider: InteractionProvider - maps new radial points with
            shape (n_new_R,) and channel blocks to one interaction batch per block;
            each batch has shape (n_new_R, n_channel_block, n_channel_block)
        energies: NDArray[np.float64] - total energies, shape (n_energy,)
        reduced_mass: float - collision reduced mass in atomic units
        radial_boundaries: Sequence[float] - radial interval boundaries, shape
            (n_interval + 1,)
        radial_half_steps: Sequence[float] - nominal LDMD half-steps, shape
            (n_interval,)
        mode: Literal["inelastic", "capture"] - inner-boundary condition
        memory_limit_mb: float - target transient-memory limit in MiB
        potential_grid_size: int - internal PES grid points per R used by the
            memory estimator
        print_verbose: bool - whether to emit INFO-level propagation progress
        device: str - auto, CPU, or GPU propagation-device request

    Returns:
        Y_blocks: tuple[jax.Array, ...] - final log derivatives; each array has
            shape (n_energy, n_channel_block, n_channel_block)
    """
    if mode not in ("inelastic", "capture"):
        message = f"mode must be 'inelastic' or 'capture', but got {mode!r}"
        logger.error(message)
        raise ValueError(message)
    selected_device = resolve_device(device)
    blocks = tuple(tuple(indices) for indices in channel_blocks)
    if not blocks or any(not indices for indices in blocks):
        message = "At least one non-empty channel block is required for propagation"
        logger.error(message)
        raise ValueError(message)

    E_int_blocks: list[NDArray[np.float64]] = []
    Umat_blocks: list[NDArray[np.float64]] = []
    for indices in blocks:
        positions = np.asarray(indices, dtype=np.int64)
        E_int_blocks.append(basis.E_int[positions])
        Umat_blocks.append(get_Umat_BF(basis, indices))

    sectors = build_radial_sectors(radial_boundaries, radial_half_steps)
    block_sizes = tuple(len(indices) for indices in blocks)
    windows = iter_radial_windows(
        sectors,
        n_grid=potential_grid_size,
        n_channel=max(block_sizes),
        n_energy=energies.size,
        memory_limit_mb=memory_limit_mb,
        state_matrix_elements=sum(size**2 for size in block_sizes),
    )
    n_sector = len(sectors)
    progress_interval = max(1, ceil(n_sector / 10))
    next_progress = progress_interval
    completed_sectors = 0
    propagation_start = perf_counter()
    logger.info(f"Propagation device: {selected_device.label}, x64={jax.config.jax_enable_x64}")
    if print_verbose:
        logger.info(f"Propagation started: blocks={len(blocks)}, sectors={n_sector}, channels={max(block_sizes)}, energies={energies.size}")

    Y_states: list[jax.Array | None] = [None] * len(blocks)
    cached_R: float | None = None
    cached_interactions: list[NDArray[np.float64] | None] = [None] * len(blocks)

    for window_index, (window, radial_points) in enumerate(windows):
        reuse_endpoint = cached_R is not None and np.isclose(radial_points[0], cached_R, rtol=0.0, atol=1.0e-12)
        new_points = radial_points[1:] if reuse_endpoint else radial_points
        new_interactions = interaction_provider(new_points, blocks)
        if len(new_interactions) != len(blocks):
            message = f"Interaction provider returned {len(new_interactions)} blocks, but expected {len(blocks)}"
            logger.error(message)
            raise ValueError(message)

        for block_index, (E_int, Umat, new_interaction) in enumerate(zip(E_int_blocks, Umat_blocks, new_interactions, strict=True)):
            n_channel = block_sizes[block_index]
            expected_shape = (new_points.size, n_channel, n_channel)
            if new_interaction.shape != expected_shape:
                message = f"Interaction block {block_index} has shape {new_interaction.shape}, but expected {expected_shape}"
                logger.error(message)
                raise ValueError(message)

            cached_interaction = cached_interactions[block_index]
            if reuse_endpoint and cached_interaction is not None:
                interactions = np.concatenate((cached_interaction[None, :, :], new_interaction), axis=0)
            else:
                interactions = new_interaction

            W_points = np.stack(
                [
                    get_Wmat(float(RR), 0.0, reduced_mass, E_int, Umat, interaction)
                    for RR, interaction in zip(radial_points, interactions, strict=True)
                ]
            )
            W_base_start = W_points[0:-1:2]
            W_base_mid = W_points[1::2]
            W_base_end = W_points[2::2]
            sector_half_steps = np.asarray([sector.radial_half_step for sector in window], dtype=np.float64)

            Y_current = Y_states[block_index]
            if Y_current is None:
                identity = np.eye(n_channel, dtype=np.float64)
                W_initial = W_base_start[0][None, :, :] - 2.0 * reduced_mass * energies[:, None, None] * identity
                if mode == "inelastic":
                    Y_current = initialize_logD_inelastic(jax.device_put(W_initial, selected_device.device))
                else:
                    Y_current = initialize_logD_capture(jax.device_put(W_initial, selected_device.device))

            Y_states[block_index] = propagate_logD(
                Y_current,
                energies,
                reduced_mass,
                sector_half_steps,
                W_base_start,
                W_base_mid,
                W_base_end,
                device=selected_device.device,
            )
            cached_interactions[block_index] = interactions[-1].copy()

        cached_R = float(radial_points[-1])
        completed_sectors += len(window)
        if print_verbose and (completed_sectors == n_sector or completed_sectors >= next_progress):
            for Y_state in Y_states:
                if Y_state is not None:
                    Y_state.block_until_ready()
            logger.info(
                f"Propagation: {completed_sectors}/{n_sector} sectors, R={cached_R:.6f} bohr, wall={perf_counter() - propagation_start:.3f} s"
            )
            while next_progress <= completed_sectors:
                next_progress += progress_interval
        logger.trace(f"Completed radial window {window_index + 1}")

    if any(Y_state is None for Y_state in Y_states):
        message = "Propagation produced no radial windows"
        logger.error(message)
        raise RuntimeError(message)
    return tuple(cast(jax.Array, Y_state) for Y_state in Y_states)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def propagate_BF(
    basis: ChannelBasis,
    Vmat: VmatCallback,
    Etot: EnergyInput,
    reduced_mass: float,
    radial_boundaries: Sequence[float],
    radial_half_steps: Sequence[float],
    mode: Literal["inelastic", "capture"] = "inelastic",
    channel_indices: Sequence[int] | None = None,
    batch_Vmat: bool = False,
    memory_limit_mb: float = 512.0,
    potential_grid_size: int = 0,
    print_verbose: bool = False,
    device: str = "auto",
) -> jax.Array:
    r"""
    Propagate one body-fixed log-derivative block with the LDMD method.

    Formula:
        W(R; Etot) = U / R**2
                     + 2 * reduced_mass * [V(R) + diag(E_int - Etot)].

    Inputs:
        basis: ChannelBasis - complete field-free channel basis
        Vmat: Callable - scalar callback returning shape (n_channel, n_channel), or
            batched callback mapping R with shape (n_R,) to matrices with shape
            (n_R, n_channel, n_channel)
        Etot: EnergyInput - total-energy array with shape (n_energy,), or a
            one-column text file in atomic units
        reduced_mass: float - collision reduced mass in atomic units
        radial_boundaries: Sequence[float] - increasing radial interval boundaries
            with shape (n_interval + 1,) in atomic units
        radial_half_steps: Sequence[float] - nominal LDMD half-step for each radial
            interval, shape (n_interval,)
        mode: Literal["inelastic", "capture"] - inner-boundary condition
        channel_indices: Sequence[int] | None - complete-basis positions for one
            propagation block, shape (n_channel,)
        batch_Vmat: bool - evaluate each window's new radial points in one call
        memory_limit_mb: float - target transient-memory limit in MiB
        potential_grid_size: int - internal PES grid points per R used by the memory
            estimator
        print_verbose: bool - whether to emit INFO-level propagation progress
        device: str - auto, CPU, or GPU propagation-device request

    Returns:
        Y_final: jax.Array - final log-derivative matrices with shape
            (n_energy, n_channel, n_channel)
    """
    energies = get_Etot(Etot)
    indices = tuple(range(basis.n_channel)) if channel_indices is None else tuple(channel_indices)

    def interaction_provider(radial_points: NDArray[np.float64], blocks: tuple[tuple[int, ...], ...]) -> tuple[NDArray[np.float64], ...]:
        """Evaluate one interaction block at new R points, returning shape (n_R, n_channel, n_channel)."""
        if batch_Vmat:
            callback = cast(Callable[[NDArray[np.float64]], NDArray[np.float64]], Vmat)
            interactions = np.asarray(callback(radial_points), dtype=np.float64)
        else:
            callback = cast(Callable[[float], NDArray[np.float64]], Vmat)
            interactions = np.stack([np.asarray(callback(float(radial_point)), dtype=np.float64) for radial_point in radial_points])
        return (interactions,)

    return propagate_BF_blocks(
        basis,
        (indices,),
        interaction_provider,
        energies,
        reduced_mass,
        radial_boundaries,
        radial_half_steps,
        mode,
        memory_limit_mb,
        potential_grid_size,
        print_verbose=print_verbose,
        device=device,
    )[0]


# ----------------------------------------------------------------------------------------
