from dataclasses import replace
from time import perf_counter, process_time
from typing import cast

import numpy as np
from loguru import logger

from pyticc.basis.channel import ChannelBasis, ChannelBasisElectricSF
from pyticc.basis.kblock import KBlock, build_cs_blocks, build_nncc_blocks
from pyticc.energy import EnergyInput, get_Etot
from pyticc.match.finalize import finalize_K_block, finalize_scattering
from pyticc.propagation.config import Propagation
from pyticc.propagation.runner import propagate, propagate_BF_blocks
from pyticc.result import CoupledStatesResult, ScatteringResult, Timing
from pyticc.scattering.hamiltonian import ScattHamiltonian
from pyticc.system import Approx


# ----------------------------------------------------------------------------------------
def solve(
    hamiltonian: ScattHamiltonian,
    Etot: EnergyInput,
    propagation: Propagation,
) -> ScatteringResult | CoupledStatesResult:
    """Propagate and match one scattering Hamiltonian.

    Inputs:
        hamiltonian: ScattHamiltonian - projected channel Hamiltonian
        Etot: EnergyInput - total energies in atomic units
        propagation: Propagation - radial grid and runtime settings

    Returns:
        result: ScatteringResult | CoupledStatesResult - matched exact or
            coupled-states result
    """
    energies = get_Etot(Etot)
    basis = hamiltonian.basis
    electric_sf = isinstance(basis, ChannelBasisElectricSF)
    approximation = Approx.EXACT if electric_sf else hamiltonian.system.approx
    block_label = f"M={basis.M}" if electric_sf else f"Jtot={hamiltonian.system.Jtot}, system_parity={hamiltonian.system.system_parity:+d}"
    wall_start = perf_counter()
    cpu_start = process_time()
    logger.info(f"Solving {block_label}, approx={approximation.value}, channels={basis.n_channel}, energies={energies.size}")

    if approximation is Approx.EXACT:
        Y_propagated = propagate(
            hamiltonian,
            energies,
            propagation.boundaries,
            propagation.half_steps,
            mode=propagation.mode,
            memory_limit_mb=propagation.memory_mb,
            print_verbose=propagation.print_verbose,
            device=propagation.device,
        )
        result: ScatteringResult | CoupledStatesResult = finalize_scattering(
            basis,
            np.asarray(Y_propagated),
            energies,
            hamiltonian.reduced_mass,
            propagation.Rmatch,
        )

    else:
        basis_bf = cast(ChannelBasis, basis)
        blocks = build_k_blocks(hamiltonian)
        channel_blocks = tuple(block.channel_indices for block in blocks)
        Y_blocks = propagate_BF_blocks(
            basis_bf,
            channel_blocks,
            hamiltonian.V_blocks,
            energies,
            hamiltonian.reduced_mass,
            propagation.boundaries,
            propagation.half_steps,
            propagation.mode,
            propagation.memory_mb,
            hamiltonian.potential_grid_size,
            print_verbose=propagation.print_verbose,
            device=propagation.device,
        )
        results = tuple(
            finalize_K_block(
                basis_bf,
                block,
                np.asarray(Y_BF),
                energies,
                hamiltonian.reduced_mass,
                propagation.Rmatch,
            )
            for block, Y_BF in zip(blocks, Y_blocks, strict=True)
        )
        result = CoupledStatesResult(
            basis=basis_bf,
            Etot=energies,
            open_closed=basis_bf.open_closed(energies),
            approx=approximation,
            blocks=results,
        )

    timing = Timing(wall_seconds=perf_counter() - wall_start, cpu_seconds=process_time() - cpu_start)
    logger.info(f"Solver complete: {timing}")
    return replace(result, timing=timing)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_k_blocks(hamiltonian: ScattHamiltonian) -> tuple[KBlock, ...]:
    """Build CS or NNCC propagation blocks for one Hamiltonian."""
    approximation = hamiltonian.system.approx
    basis = cast(ChannelBasis, hamiltonian.basis)
    if approximation is Approx.CS:
        return build_cs_blocks(basis)
    if approximation is Approx.NNCC:
        return build_nncc_blocks(basis, hamiltonian.system.K_delta)

    message = f"Coupled-states blocks require approx='cs' or 'nncc', but got {approximation.value!r}"
    logger.error(message)
    raise ValueError(message)


# ----------------------------------------------------------------------------------------
