from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from time import perf_counter
from typing import TYPE_CHECKING, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc._typing import JaxDevice
from pyticc.basis.channel import ChannelBasis
from pyticc.energy import EnergyInput, get_Etot
from pyticc.matrix.centrifugal import get_Umat_BF
from pyticc.matrix.radial import get_Wmat
from pyticc.propagation.config import Propagation
from pyticc.propagation.device import resolve_device
from pyticc.propagation.grid import RadialSector, iter_radial_windows
from pyticc.propagation.logd import initialize_logD_capture, initialize_logD_inelastic, propagate_logD

if TYPE_CHECKING:
    from pyticc.scattering.hamiltonian import ScattHamiltonian

InteractionMatrix: TypeAlias = NDArray[np.float64] | NDArray[np.complex128] | jax.Array
InteractionProvider = Callable[[NDArray[np.float64], tuple[tuple[int, ...], ...], JaxDevice], tuple[InteractionMatrix, ...]]
RadialWindow: TypeAlias = tuple[tuple[RadialSector, ...], NDArray[np.float64]]
InteractionWindow: TypeAlias = tuple[
    tuple[RadialSector, ...],
    NDArray[np.float64],
    NDArray[np.float64],
    bool,
    tuple[InteractionMatrix, ...],
]


# ----------------------------------------------------------------------------------------
def _evaluate_interaction_windows(
    windows: Iterator[RadialWindow],
    interaction_provider: InteractionProvider,
    blocks: tuple[tuple[int, ...], ...],
    device: JaxDevice,
    *,
    prefetch: bool,
) -> Iterator[InteractionWindow]:
    """Evaluate radial interactions sequentially or with one-window lookahead.

    GPU propagation uses a single background thread to prepare and contract the
    next radial window while the current log-derivative propagation is in
    flight. CPU propagation remains sequential to avoid competing with its own
    JAX and BLAS threads.

    Inputs:
        windows: Iterator[RadialWindow] - memory-bounded radial windows
        interaction_provider: InteractionProvider - PES and contraction callback
        blocks: tuple[tuple[int, ...], ...] - channel positions for each block
        device: JaxDevice - selected contraction and propagation device
        prefetch: bool - whether to evaluate one future window concurrently

    Yields:
        window: tuple[RadialSector, ...] - current propagation sectors
        radial_points: NDArray[np.float64] - all start, midpoint, and end points
        new_points: NDArray[np.float64] - points requiring new PES evaluation
        reuse_endpoint: bool - whether the first interaction is cached
        interactions: tuple[InteractionMatrix, ...] - new matrices per block
    """
    try:
        current_window, current_points = next(windows)
    except StopIteration:
        return

    current_reuse = False
    current_new_points = current_points
    current_interactions = interaction_provider(current_new_points, blocks, device)
    if not prefetch:
        yield current_window, current_points, current_new_points, current_reuse, current_interactions
        previous_R = float(current_points[-1])
        for current_window, current_points in windows:
            current_reuse = np.isclose(current_points[0], previous_R, rtol=0.0, atol=1.0e-12)
            current_new_points = current_points[1:] if current_reuse else current_points
            current_interactions = interaction_provider(current_new_points, blocks, device)
            yield current_window, current_points, current_new_points, current_reuse, current_interactions
            previous_R = float(current_points[-1])
        return

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pyticc-pes-prefetch") as executor:
        while True:
            try:
                next_window, next_points = next(windows)
            except StopIteration:
                yield current_window, current_points, current_new_points, current_reuse, current_interactions
                return

            next_reuse = np.isclose(next_points[0], current_points[-1], rtol=0.0, atol=1.0e-12)
            next_new_points = next_points[1:] if next_reuse else next_points
            next_interactions = executor.submit(interaction_provider, next_new_points, blocks, device)
            yield current_window, current_points, current_new_points, current_reuse, current_interactions
            current_window = next_window
            current_points = next_points
            current_new_points = next_new_points
            current_reuse = next_reuse
            current_interactions = next_interactions.result()


# ----------------------------------------------------------------------------------------
def propagate(
    hamiltonian: ScattHamiltonian,
    Etot: EnergyInput,
    radial_sectors: Sequence[RadialSector],
    config: Propagation,
) -> jax.Array:
    r"""
    Propagate one exact scattering Hamiltonian to the final radial boundary.

    Formula:
        In the Hamiltonian's channel representation,

        W(R;E_tot)
          = U/R^2
            + 2 mu_R [V(R)+diag(E_int-E_tot)].

        U is supplied by ScattHamiltonian and is therefore the BF Coriolis
        matrix for a field-free basis or diag[l(l+1)] for an Electric-SF basis.

    Inputs:
        hamiltonian: ScattHamiltonian - projected channel Hamiltonian
        Etot: EnergyInput - total energies in atomic units
        radial_sectors: Sequence[RadialSector] - ordered propagation sectors
        config: Propagation - boundary condition, memory limit, logging, and
            device settings

    Returns:
        Y_final: jax.Array - final log-derivative matrices in the propagated
            channel representation
    """
    energies = get_Etot(Etot)
    indices = tuple(range(hamiltonian.basis.n_channel))

    def interaction_provider(
        radial_points: NDArray[np.float64], blocks: tuple[tuple[int, ...], ...], device: JaxDevice
    ) -> tuple[InteractionMatrix, ...]:
        """Evaluate the complete exact interaction matrix at new R points."""
        return hamiltonian.V_blocks(radial_points, blocks, device)

    return _propagate_blocks(
        (indices,),
        (hamiltonian.E_int,),
        (hamiltonian.U,),
        interaction_provider,
        energies,
        hamiltonian.reduced_mass,
        radial_sectors,
        config,
        hamiltonian.potential_grid_size,
    )[0]


# ----------------------------------------------------------------------------------------
def _propagate_blocks(
    blocks: tuple[tuple[int, ...], ...],
    E_int_blocks: tuple[NDArray[np.float64], ...],
    Umat_blocks: tuple[NDArray[np.float64], ...],
    interaction_provider: InteractionProvider,
    energies: NDArray[np.float64],
    reduced_mass: float,
    radial_sectors: Sequence[RadialSector],
    config: Propagation,
    potential_grid_size: int,
) -> tuple[jax.Array, ...]:
    """
    Propagate prepared channel blocks through shared radial windows.

    The provider evaluates only new radial points. The interaction matrix at each
    window boundary is retained and reused by the next window. All blocks therefore
    share the same radial traversal while keeping independent log derivatives.

    Inputs:
        blocks: tuple[tuple[int, ...], ...] - complete-basis positions for each
            channel block
        E_int_blocks: tuple[NDArray[np.float64], ...] - channel thresholds for
            every block
        Umat_blocks: tuple[NDArray[np.float64], ...] - dimensionless
            centrifugal matrix for every block
        interaction_provider: InteractionProvider - maps new radial points with
            shape (n_new_R,) and channel blocks to one interaction batch per block;
            each batch has shape (n_new_R, n_channel_block, n_channel_block)
        energies: NDArray[np.float64] - total energies, shape (n_energy,)
        reduced_mass: float - collision reduced mass in atomic units
        radial_sectors: Sequence[RadialSector] - ordered propagation sectors
        config: Propagation - boundary condition, memory limit, logging, and
            device settings
        potential_grid_size: int - internal PES grid points per R used by the
            memory estimator

    Returns:
        Y_blocks: tuple[jax.Array, ...] - final log derivatives; each array has
            shape (n_energy, n_channel_block, n_channel_block)
    """
    selected_device = resolve_device(config.device)
    if not blocks or any(not indices for indices in blocks):
        message = "At least one non-empty channel block is required for propagation"
        logger.error(message)
        raise ValueError(message)

    block_sizes = tuple(len(indices) for indices in blocks)
    windows = iter_radial_windows(
        radial_sectors,
        n_grid=potential_grid_size,
        n_channel=max(block_sizes),
        n_energy=energies.size,
        memory_limit_mb=config.memory_mb,
        state_matrix_elements=sum(size**2 for size in block_sizes),
    )
    n_sector = len(radial_sectors)
    progress_interval = max(1, ceil(n_sector / 10))
    next_progress = progress_interval
    completed_sectors = 0
    propagation_start = perf_counter()
    prefetch_interactions = selected_device.device.platform == "gpu"
    logger.info(f"Propagation device: {selected_device.label}, x64={jax.config.read('jax_enable_x64')}")
    if config.print_verbose:
        logger.info(
            f"Propagation started: blocks={len(blocks)}, sectors={n_sector}, channels={max(block_sizes)}, "
            f"energies={energies.size}, pes_prefetch={prefetch_interactions}"
        )

    Y_states: list[jax.Array | None] = [None] * len(blocks)
    cached_R: float | None = None
    cached_interactions: list[InteractionMatrix | None] = [None] * len(blocks)

    interaction_windows = _evaluate_interaction_windows(
        windows,
        interaction_provider,
        blocks,
        selected_device.device,
        prefetch=prefetch_interactions,
    )
    for window_index, (window, radial_points, new_points, reuse_endpoint, new_interactions) in enumerate(interaction_windows):
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
                concatenate = jnp.concatenate if isinstance(new_interaction, jax.Array) else np.concatenate
                interactions = concatenate((cached_interaction[None, :, :], new_interaction), axis=0)
            else:
                interactions = new_interaction

            if isinstance(interactions, jax.Array):
                radial_device = jax.device_put(radial_points, selected_device.device)
                E_int_device = jax.device_put(E_int, selected_device.device)
                Umat_device = jax.device_put(Umat, selected_device.device)
                W_points = Umat_device[None, :, :] / radial_device[:, None, None] ** 2 + 2.0 * reduced_mass * interactions
                diagonal = jnp.diag_indices(n_channel)
                W_points = W_points.at[:, diagonal[0], diagonal[1]].add(2.0 * reduced_mass * E_int_device[None, :])
            else:
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
                if config.mode == "inelastic":
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
        if config.print_verbose and (completed_sectors == n_sector or completed_sectors >= next_progress):
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
def propagate_blocks(
    hamiltonian: ScattHamiltonian,
    channel_blocks: Sequence[Sequence[int]],
    Etot: EnergyInput,
    radial_sectors: Sequence[RadialSector],
    config: Propagation,
) -> tuple[jax.Array, ...]:
    r"""
    Propagate one or more field-free BF channel blocks.

    Formula:
        For block b,

        W_b(R;E_tot)
          = U_b^BF/R^2
            + 2 mu_R [V_b(R)+diag(E_int,b-E_tot)].

    Inputs:
        hamiltonian: ScattHamiltonian - projected field-free Hamiltonian
        channel_blocks: Sequence[Sequence[int]] - complete-basis positions for
            each propagation block
        Etot: EnergyInput - total energies in atomic units
        radial_sectors: Sequence[RadialSector] - ordered propagation sectors
        config: Propagation - boundary condition and runtime settings

    Returns:
        Y_blocks: tuple[jax.Array, ...] - final BF log derivatives, one array
            per block
    """
    if not isinstance(hamiltonian.basis, ChannelBasis):
        message = "Block propagation requires a field-free ChannelBasis"
        logger.error(message)
        raise TypeError(message)
    basis = hamiltonian.basis
    energies = get_Etot(Etot)
    blocks = tuple(tuple(indices) for indices in channel_blocks)
    E_int_blocks = tuple(basis.E_int[np.asarray(indices, dtype=np.int64)] for indices in blocks)
    Umat_blocks = tuple(get_Umat_BF(basis, indices) for indices in blocks)
    return _propagate_blocks(
        blocks,
        E_int_blocks,
        Umat_blocks,
        lambda radial_points, selected_blocks, device: hamiltonian.V_blocks(radial_points, selected_blocks, device),
        energies,
        hamiltonian.reduced_mass,
        radial_sectors,
        config,
        hamiltonian.potential_grid_size,
    )


# ----------------------------------------------------------------------------------------
