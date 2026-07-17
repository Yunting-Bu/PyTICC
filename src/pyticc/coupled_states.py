from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.channel import ChannelBasis
from pyticc.basis.kblock import build_cs_blocks, build_nncc_blocks
from pyticc.energy import EnergyInput, get_Etot
from pyticc.matrix.interaction import VBasisBF, get_Vmat_BF
from pyticc.propagation.runner import propagate_BF
from pyticc.result import CoupledStatesResult, KBlockResult, _build_K_block_result
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
) -> CoupledStatesResult:
    """
    Propagate and match all CS or NNCC K blocks.

    The PES grid is cached for the common radial propagation grid. Each K block then
    contracts only its own interaction-matrix channels, so overlapping NNCC blocks do
    not trigger repeated PES evaluations.

    Inputs:
        basis: ChannelBasis - complete field-free channel basis
        V_basis: VBasisBF - precomputed internal interaction basis
        Vgrid: Callable - batched PES-grid callback evaluated at radial points
        Etot: EnergyInput - total energies in atomic units
        reduced_mass: float - collision reduced mass in atomic units
        radial_boundaries: Sequence[float] - radial interval boundaries
        radial_half_steps: Sequence[float] - nominal LDMD half-steps
        approx: Approx - CS or NNCC approximation
        K_delta: int - neighboring K range retained on each side in NNCC
        mode: Literal["inelastic", "capture"] - inner-boundary condition

    Returns:
        result: CoupledStatesResult - independently matched K blocks
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
    cached_radial_points: NDArray[np.float64] | None = None
    cached_potential_grid: NDArray[np.float64] | None = None

    def potential_grid(radial_points: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal cached_radial_points, cached_potential_grid
        points = np.asarray(radial_points, dtype=np.float64)
        if cached_potential_grid is None or cached_radial_points is None or not np.array_equal(points, cached_radial_points):
            cached_radial_points = points.copy()
            cached_potential_grid = np.asarray(Vgrid(points), dtype=np.float64)
        return cached_potential_grid

    block_results: list[KBlockResult] = []
    for block in blocks:
        message = (
            f"Propagating {approx.value.upper()} block {block.index + 1}/{len(blocks)} "
            f"with K={block.K_values}, channels={len(block.channel_indices)}"
        )
        logger.info(message)

        def Vmat(radial_points: NDArray[np.float64], indices: tuple[int, ...] = block.channel_indices) -> NDArray[np.float64]:
            return get_Vmat_BF(V_basis, potential_grid(radial_points), indices)

        Y_BF = propagate_BF(
            basis=basis,
            Vmat=Vmat,
            Etot=energies,
            reduced_mass=reduced_mass,
            radial_boundaries=radial_boundaries,
            radial_half_steps=radial_half_steps,
            mode=mode,
            channel_indices=block.channel_indices,
            batch_Vmat=True,
        )
        block_results.append(
            _build_K_block_result(
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
