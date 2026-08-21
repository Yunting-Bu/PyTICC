from dataclasses import replace
from time import perf_counter, process_time
from typing import cast, overload

import numpy as np
from loguru import logger

from pyticc.basis.channel import ChannelBasis, ChannelBasisElectricSF
from pyticc.basis.kblock import KBlock, build_cs_blocks, build_nncc_blocks
from pyticc.energy import EnergyInput, get_Etot
from pyticc.fine_structure.channel import FSChannelBasis
from pyticc.match.finalize import finalize_K_block, finalize_reactive_scattering, finalize_scattering
from pyticc.propagation.config import Propagation
from pyticc.propagation.delves import propagate_delves
from pyticc.propagation.runner import propagate, propagate_blocks
from pyticc.result import CoupledStatesResult, ReactiveScatteringResult, ScatteringResult, Timing
from pyticc.scattering.hamiltonian import ScattHamiltonian
from pyticc.scattering.reactive.delves import DelvesHamiltonian
from pyticc.system import Approx


# ----------------------------------------------------------------------------------------
@overload
def solve(
    hamiltonian: DelvesHamiltonian,
    Etot: EnergyInput,
    propagation: Propagation,
) -> ReactiveScatteringResult: ...


@overload
def solve(
    hamiltonian: ScattHamiltonian,
    Etot: EnergyInput,
    propagation: Propagation,
) -> ScatteringResult | CoupledStatesResult: ...


def solve(
    hamiltonian: ScattHamiltonian | DelvesHamiltonian,
    Etot: EnergyInput,
    propagation: Propagation,
) -> ScatteringResult | CoupledStatesResult | ReactiveScatteringResult:
    """Propagate and match one scattering Hamiltonian.

    Inputs:
        hamiltonian: ScattHamiltonian | DelvesHamiltonian - fixed-channel or
            reactive surface Hamiltonian
        Etot: EnergyInput - total energies in atomic units
        propagation: Propagation - radial grid and runtime settings

    Returns:
        result: ScatteringResult | CoupledStatesResult |
            ReactiveScatteringResult - matched fixed-arrangement,
            coupled-states, or reactive result
    """
    energies = get_Etot(Etot)
    if isinstance(hamiltonian, DelvesHamiltonian):
        block_label = (
            f"Jtot={hamiltonian.basis.Jtot}, system_parity={hamiltonian.basis.system_parity:+d}, "
            f"exchange_parity={hamiltonian.basis.exchange_parity:+d}"
        )
        wall_start = perf_counter()
        cpu_start = process_time()
        logger.info(f"Solving reactive {block_label}, primitive={hamiltonian.basis.n_primitive}, energies={energies.size}")
        propagated = propagate_delves(
            hamiltonian,
            energies,
            propagation,
        )
        reactive_result = finalize_reactive_scattering(
            hamiltonian.basis,
            hamiltonian.total_potential,
            propagated,
            energies,
            energy_zero=hamiltonian.energy_zero,
        )
        timing = Timing(wall_seconds=perf_counter() - wall_start, cpu_seconds=process_time() - cpu_start)
        logger.info(f"Solver complete: {timing}")
        return replace(reactive_result, timing=timing)

    basis = hamiltonian.basis
    electric_sf = isinstance(basis, ChannelBasisElectricSF)
    approximation = Approx.EXACT if electric_sf else hamiltonian.approx
    block_label = f"M={basis.M}" if electric_sf else f"Jtot={basis.Jtot}, system_parity={basis.system_parity:+d}"
    wall_start = perf_counter()
    cpu_start = process_time()
    logger.info(f"Solving {block_label}, approx={approximation.value}, channels={basis.n_channel}, energies={energies.size}")

    if approximation is Approx.EXACT:
        Y_propagated = propagate(hamiltonian, energies, propagation)
        result: ScatteringResult | CoupledStatesResult = finalize_scattering(
            basis,
            np.asarray(Y_propagated),
            energies,
            hamiltonian.reduced_mass,
            propagation.Rmatch,
        )

    else:
        if isinstance(basis, FSChannelBasis):
            message = "Fine-structure channels currently support exact coupled channels; CS/NNCC will be added after exact validation"
            logger.error(message)
            raise NotImplementedError(message)
        basis_bf = cast(ChannelBasis, basis)
        blocks = build_k_blocks(hamiltonian)
        channel_blocks = tuple(block.channel_indices for block in blocks)
        Y_blocks = propagate_blocks(hamiltonian, channel_blocks, energies, propagation)
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
    approximation = hamiltonian.approx
    basis = cast(ChannelBasis, hamiltonian.basis)
    if approximation is Approx.CS:
        return build_cs_blocks(basis)
    if approximation is Approx.NNCC:
        return build_nncc_blocks(basis, hamiltonian.K_delta)

    message = f"Coupled-states blocks require approx='cs' or 'nncc', but got {approximation.value!r}"
    logger.error(message)
    raise ValueError(message)


# ----------------------------------------------------------------------------------------
