from collections.abc import Callable, Sequence
from math import prod
from typing import Literal

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.channel import ChannelBasis
from pyticc.basis.kblock import build_cs_blocks, build_nncc_blocks
from pyticc.energy import EnergyInput, get_Etot
from pyticc.match.finalize import finalize_K_block
from pyticc.matrix.interaction import VBasisBF, get_Vmat_BF
from pyticc.propagation.runner import propagate_BF_blocks
from pyticc.result import CoupledStatesResult, KBlockResult
from pyticc.system import Approx


# ----------------------------------------------------------------------------------------
def run_coupled_states_BF(
    basis: ChannelBasis,
    V_basis: VBasisBF,
    Vgrid: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    Etot: EnergyInput,
    reduced_mass: float,
    radial_boundaries: Sequence[float],
    radial_half_steps: Sequence[float],
    approx: Approx,
    K_delta: int = 1,
    mode: Literal["inelastic", "capture"] = "inelastic",
    memory_limit_mb: float = 512.0,
) -> CoupledStatesResult:
    """
    Propagate and match all CS or NNCC K blocks.

    Every radial window's raw PES grid is evaluated once and contracted for all K
    blocks before the shared propagation engine advances their log derivatives.

    Inputs:
        basis: ChannelBasis - complete field-free channel basis
        V_basis: VBasisBF - precomputed internal interaction basis
        Vgrid: Callable - callback mapping radial points with shape (n_R,) to a PES
            grid with shape (n_R, *grid_shape)
        Etot: EnergyInput - total energies with shape (n_energy,) in atomic units,
            or a one-column text file
        reduced_mass: float - collision reduced mass in atomic units
        radial_boundaries: Sequence[float] - radial interval boundaries, shape
            (n_interval + 1,)
        radial_half_steps: Sequence[float] - nominal LDMD half-steps, shape
            (n_interval,)
        approx: Approx - CS or NNCC approximation
        K_delta: int - neighboring K range retained on each side in NNCC
        mode: Literal["inelastic", "capture"] - inner-boundary condition
        memory_limit_mb: float - target transient-memory limit in MiB

    Returns:
        result: CoupledStatesResult - independently matched K blocks whose block
            log-derivative arrays have shape
            (n_energy, n_channel_block, n_channel_block)
    """
    if approx is Approx.CS:
        blocks = build_cs_blocks(basis)
    elif approx is Approx.NNCC:
        blocks = build_nncc_blocks(basis, K_delta)
    else:
        message = f"Coupled-states propagation requires approx='cs' or 'nncc', but got {approx.value!r}"
        logger.error(message)
        raise ValueError(message)

    energies = get_Etot(Etot)
    channel_blocks = tuple(block.channel_indices for block in blocks)

    def interaction_provider(radial_points: NDArray[np.float64], indices_by_block: tuple[tuple[int, ...], ...]) -> tuple[NDArray[np.float64], ...]:
        """Evaluate one PES grid batch and contract all K blocks at new R points."""
        potential_grid = np.asarray(Vgrid(radial_points), dtype=np.float64)
        expected_shape = (radial_points.size, *V_basis.grid_shape)
        if potential_grid.shape != expected_shape:
            message = f"Vgrid returned shape {potential_grid.shape}, but expected {expected_shape}"
            logger.error(message)
            raise ValueError(message)
        return tuple(get_Vmat_BF(V_basis, potential_grid, indices) for indices in indices_by_block)

    Y_states = propagate_BF_blocks(
        basis,
        channel_blocks,
        interaction_provider,
        energies,
        reduced_mass,
        radial_boundaries,
        radial_half_steps,
        mode,
        memory_limit_mb,
        prod(V_basis.grid_shape),
    )

    block_results: list[KBlockResult] = []
    for block, Y_BF in zip(blocks, Y_states, strict=True):
        block_results.append(
            finalize_K_block(
                basis,
                block,
                np.asarray(Y_BF),
                energies,
                reduced_mass,
                float(radial_boundaries[-1]),
            )
        )

    return CoupledStatesResult(
        basis=basis,
        Etot=energies,
        open_closed=basis.open_closed(energies),
        approx=approx,
        blocks=tuple(block_results),
    )
