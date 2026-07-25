from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import prod
from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.angle import norm_YjK
from pyticc.basis.channel import Channel, ChannelBasis
from pyticc.basis.monomer.diabatic import DiabaticDiatomBasis
from pyticc.pes.diabatic import DiabaticPESWrapper, RadialInput, get_diabatic_potential_grid_atom_diatom
from pyticc.system import MolInnerState

ElectronicK = tuple[int, int]
_MAX_CONTRACTION_WORKERS = 3
_MAX_CONTRACTION_WORKSPACE_BYTES = 64 * 1024**2


@dataclass(frozen=True)
class DiabaticVBasisBF:
    """State-aware atom-diatom bases for diagonal and off-diagonal diabatic-potential contractions."""

    n_channel: int
    n_state: int
    theta: NDArray[np.float64]
    diagonal_grids: tuple[NDArray[np.float64], ...]
    coupling_grid: NDArray[np.float64]
    channel_indices: dict[ElectronicK, tuple[int, ...]]
    B_diagonal: dict[ElectronicK, NDArray[np.float64]]
    B_coupling: dict[ElectronicK, NDArray[np.float64]]


@dataclass(frozen=True)
class DiabaticVGridBF:
    """Diabatic-potential values sampled on state-specific diagonal and shared coupling grids."""

    diagonal: tuple[NDArray[np.float64], ...]
    coupling: NDArray[np.float64]


@dataclass(frozen=True)
class _ContractionTask:
    """One independent electronic-K block contraction."""

    left: NDArray[np.float64]
    weights: NDArray[np.float64]
    right: NDArray[np.float64]
    row_positions: NDArray[np.int64]
    column_positions: NDArray[np.int64]
    symmetric: bool


def _diatomic_inner_state(channel: Channel) -> MolInnerState:
    """Return the unique diatomic inner state from an atom-diatom channel."""
    candidates = tuple(state for state in (channel.mis_X, channel.mis_Y) if state.v is not None)
    if len(candidates) != 1 or candidates[0].electronic_state is None:
        message = "Diabatic atom-diatom channels require exactly one electronically labelled diatomic state"
        logger.error(message)
        raise ValueError(message)
    return candidates[0]


def prepare(
    basis: ChannelBasis,
    diabatic_basis: DiabaticDiatomBasis,
    cos_theta: NDArray[np.float64],
    theta_weights: NDArray[np.float64],
) -> DiabaticVBasisBF:
    r"""Prepare state-specific PODVR bases and shared primitive-DVR coupling bases.

    Diagonal blocks use state-specific PODVR grids; couplings use the common
    primitive DVR grid, matching the TransP/TransD separation in ABCdia.

    Inputs:
        basis: ChannelBasis - electronically labelled atom-diatom channels
        diabatic_basis: DiabaticDiatomBasis - state-specific and shared radial bases
        cos_theta: NDArray[np.float64] - Gauss-Legendre nodes on [-1, 1]
        theta_weights: NDArray[np.float64] - corresponding quadrature weights

    Returns:
        V_basis: DiabaticVBasisBF - precomputed bases and their radial grids
    """
    cos_values = np.asarray(cos_theta, dtype=np.float64)
    weights = np.asarray(theta_weights, dtype=np.float64)
    if cos_values.ndim != 1 or weights.shape != cos_values.shape:
        message = f"Angular nodes and weights must be matching one-dimensional arrays, but got {cos_values.shape} and {weights.shape}"
        logger.error(message)
        raise ValueError(message)
    if np.any(weights < 0.0) or np.any(np.abs(cos_values) > 1.0) or not np.all(np.isfinite(cos_values)) or not np.all(np.isfinite(weights)):
        message = "Angular nodes must be finite and lie in [-1, 1]; weights must be finite and non-negative"
        logger.error(message)
        raise ValueError(message)

    channel_indices: dict[ElectronicK, tuple[int, ...]] = {}
    B_diagonal: dict[ElectronicK, NDArray[np.float64]] = {}
    B_coupling: dict[ElectronicK, NDArray[np.float64]] = {}
    sqrt_weights = np.sqrt(weights)
    angular: dict[tuple[int, int], NDArray[np.float64]] = {}

    for electronic_state in range(diabatic_basis.n_state):
        state_basis = diabatic_basis.state(electronic_state)
        for K in sorted({channel.K for channel in basis}):
            indices = tuple(
                index for index, channel in enumerate(basis) if channel.K == K and _diatomic_inner_state(channel).electronic_state == electronic_state
            )
            if not indices:
                continue

            diagonal_rows = np.empty((len(indices), state_basis.rovib.grids.size * cos_values.size), dtype=np.float64)
            coupling_rows = np.empty((len(indices), state_basis.dvr.grids.size * cos_values.size), dtype=np.float64)
            for local_index, channel_index in enumerate(indices):
                channel = basis[channel_index]
                inner_state = _diatomic_inner_state(channel)
                v = cast(int, inner_state.v)
                j = inner_state.j
                angular_key = (j, K)
                if angular_key not in angular:
                    angular[angular_key] = sqrt_weights * np.asarray(norm_YjK(j, K, cos_values), dtype=np.float64)

                angular_values = angular[angular_key]
                diagonal_rows[local_index] = np.multiply.outer(state_basis.rovib.WF_vj[:, v, j], angular_values).reshape(-1)
                coupling_rows[local_index] = np.multiply.outer(state_basis.rovib_dvr.WF_vj[:, v, j], angular_values).reshape(-1)

            key = (electronic_state, K)
            channel_indices[key] = indices
            B_diagonal[key] = np.ascontiguousarray(diagonal_rows)
            B_coupling[key] = np.ascontiguousarray(coupling_rows)

    return DiabaticVBasisBF(
        n_channel=basis.n_channel,
        n_state=diabatic_basis.n_state,
        theta=np.arccos(cos_values),
        diagonal_grids=tuple(state.rovib.grids for state in diabatic_basis.states),
        coupling_grid=diabatic_basis.states[0].dvr.grids,
        channel_indices=channel_indices,
        B_diagonal=B_diagonal,
        B_coupling=B_coupling,
    )


def sample(
    pes: DiabaticPESWrapper,
    R: RadialInput,
    V_basis: DiabaticVBasisBF,
) -> DiabaticVGridBF:
    """Sample diagonal and coupling diabatic-potential elements on their respective radial grids."""
    if pes.n_state != V_basis.n_state:
        message = f"PES has {pes.n_state} electronic states, but the interaction basis has {V_basis.n_state}"
        logger.error(message)
        raise ValueError(message)

    radial_grids = (*V_basis.diagonal_grids, V_basis.coupling_grid)
    radial_sizes = tuple(grid.size for grid in radial_grids)
    combined_grid = np.concatenate(radial_grids)
    combined_potential = get_diabatic_potential_grid_atom_diatom(pes, R, combined_grid, V_basis.theta)
    radial_axis = 1 if np.asarray(R).ndim == 1 else 0
    split_points = np.cumsum(radial_sizes[:-1])
    sampled_grids = np.split(combined_potential, split_points, axis=radial_axis)
    diagonal = tuple(sampled_grids[state][..., state, state] for state in range(V_basis.n_state))
    coupling = sampled_grids[-1]
    return DiabaticVGridBF(diagonal=diagonal, coupling=coupling)


def _as_grid_batch(
    values: NDArray[np.float64],
    grid_shape: tuple[int, ...],
    trailing_shape: tuple[int, ...],
    name: str,
) -> tuple[NDArray[np.float64], bool]:
    """Validate one scalar or radial-batched potential grid and expose a batch axis."""
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        message = f"{name} must contain real values"
        logger.error(message)
        raise ValueError(message)
    array = np.asarray(raw, dtype=np.float64)
    expected = (*grid_shape, *trailing_shape)
    if array.shape == expected:
        batch = array.reshape((1, *expected))
        batched = False
    elif array.ndim == len(expected) + 1 and array.shape[1:] == expected:
        batch = array
        batched = True
    else:
        message = f"{name} has shape {array.shape}, but expected {expected} with an optional leading R axis"
        logger.error(message)
        raise ValueError(message)
    if not np.all(np.isfinite(batch)):
        message = f"{name} contains non-finite values"
        logger.error(message)
        raise ValueError(message)
    return batch, batched


def _selected_group(
    group_indices: tuple[int, ...],
    selected_indices: dict[int, int],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return basis-row and output positions retained from one electronic-K group."""
    selected = tuple((row, selected_indices[index]) for row, index in enumerate(group_indices) if index in selected_indices)
    return (
        np.asarray([row for row, _ in selected], dtype=np.int64),
        np.asarray([position for _, position in selected], dtype=np.int64),
    )


def _contract_weighted_basis(
    left: NDArray[np.float64],
    weights: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Contract radial batches as memory-bounded large BLAS matrix products."""
    result = np.empty((weights.shape[0], left.shape[0], right.shape[0]), dtype=np.float64)
    right_transpose = right.T
    bytes_per_batch = left.size * left.itemsize
    chunk_size = max(1, _MAX_CONTRACTION_WORKSPACE_BYTES // bytes_per_batch)

    for start in range(0, weights.shape[0], chunk_size):
        stop = min(start + chunk_size, weights.shape[0])
        weighted_left = np.multiply(left[None, :, :], weights[start:stop, None, :])
        weighted_rows = weighted_left.reshape(-1, left.shape[1])
        result_rows = result[start:stop].reshape(-1, right.shape[0])
        np.matmul(weighted_rows, right_transpose, out=result_rows)
    return result


def _evaluate_contraction_task(task: _ContractionTask) -> NDArray[np.float64]:
    """Evaluate one electronic-K contraction and enforce diagonal-block symmetry."""
    block = _contract_weighted_basis(task.left, task.weights, task.right)
    return 0.5 * (block + np.swapaxes(block, -2, -1)) if task.symmetric else block


def _evaluate_contraction_tasks(tasks: Sequence[_ContractionTask], batched: bool) -> tuple[NDArray[np.float64], ...]:
    """Evaluate independent electronic-K blocks concurrently for radial batches."""
    if not batched or len(tasks) < 2:
        return tuple(_evaluate_contraction_task(task) for task in tasks)

    worker_count = min(_MAX_CONTRACTION_WORKERS, len(tasks))
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="pyticc-vmat") as executor:
        return tuple(executor.map(_evaluate_contraction_task, tasks))


def contract(
    V_basis: DiabaticVBasisBF,
    potential: DiabaticVGridBF,
    channel_indices: Sequence[int] | None = None,
) -> NDArray[np.float64]:
    r"""Contract a state-resolved atom-diatom diabatic potential into a body-fixed channel matrix.

    Formula:
        V[c_i,c'_j] = delta[K,K'] sum_g B[c_i,g] V[i,j,g] B[c'_j,g].

    Inputs:
        V_basis: DiabaticVBasisBF - state-aware diagonal and coupling bases
        potential: DiabaticVGridBF - sampled diagonal and full diabatic coupling grids
        channel_indices: Sequence[int] | None - optional complete/CS/NNCC selection

    Returns:
        Vmat: NDArray[np.float64] - symmetric channel matrix, optionally preceded by R
    """
    if len(potential.diagonal) != V_basis.n_state:
        message = f"Expected {V_basis.n_state} diagonal potential grids, but got {len(potential.diagonal)}"
        logger.error(message)
        raise ValueError(message)

    diagonal_batches: list[NDArray[np.float64]] = []
    batched_flags: list[bool] = []
    for electronic_state, (values, radial_grid) in enumerate(zip(potential.diagonal, V_basis.diagonal_grids, strict=True)):
        batch, batched = _as_grid_batch(values, (radial_grid.size, V_basis.theta.size), (), f"State {electronic_state} diagonal potential")
        diagonal_batches.append(batch.reshape(batch.shape[0], -1))
        batched_flags.append(batched)

    coupling_batch, coupling_batched = _as_grid_batch(
        potential.coupling,
        (V_basis.coupling_grid.size, V_basis.theta.size),
        (V_basis.n_state, V_basis.n_state),
        "Diabatic coupling potential",
    )
    batched_flags.append(coupling_batched)
    batch_sizes = {batch.shape[0] for batch in (*diagonal_batches, coupling_batch)}
    if len(set(batched_flags)) != 1 or len(batch_sizes) != 1:
        message = "Diagonal and coupling potential grids must use the same scalar or radial-batch shape"
        logger.error(message)
        raise ValueError(message)
    if not np.allclose(coupling_batch, np.swapaxes(coupling_batch, -2, -1), rtol=1.0e-12, atol=1.0e-12):
        message = "Diabatic coupling potential must be symmetric"
        logger.error(message)
        raise ValueError(message)

    indices = tuple(range(V_basis.n_channel)) if channel_indices is None else tuple(channel_indices)
    if len(set(indices)) != len(indices) or any(index < 0 or index >= V_basis.n_channel for index in indices):
        message = "channel_indices must be unique complete-basis positions"
        logger.error(message)
        raise ValueError(message)
    selected_indices = {global_index: local_index for local_index, global_index in enumerate(indices)}
    n_batch = next(iter(batch_sizes))
    Vmat = np.zeros((n_batch, len(indices), len(indices)), dtype=np.float64)
    tasks: list[_ContractionTask] = []

    for key, group_indices in V_basis.channel_indices.items():
        electronic_state, _ = key
        rows, positions = _selected_group(group_indices, selected_indices)
        if rows.size == 0:
            continue
        B = V_basis.B_diagonal[key][rows]
        tasks.append(
            _ContractionTask(
                left=B,
                weights=diagonal_batches[electronic_state],
                right=B,
                row_positions=positions,
                column_positions=positions,
                symmetric=True,
            )
        )

    K_values = sorted({K for _, K in V_basis.channel_indices})
    coupling_flat = coupling_batch.reshape(n_batch, prod(coupling_batch.shape[1:-2]), V_basis.n_state, V_basis.n_state)
    for K in K_values:
        for state_a in range(V_basis.n_state):
            key_a = (state_a, K)
            if key_a not in V_basis.channel_indices:
                continue
            rows_a, positions_a = _selected_group(V_basis.channel_indices[key_a], selected_indices)
            if rows_a.size == 0:
                continue
            for state_b in range(state_a + 1, V_basis.n_state):
                key_b = (state_b, K)
                if key_b not in V_basis.channel_indices:
                    continue
                rows_b, positions_b = _selected_group(V_basis.channel_indices[key_b], selected_indices)
                if rows_b.size == 0:
                    continue
                B_a = V_basis.B_coupling[key_a][rows_a]
                B_b = V_basis.B_coupling[key_b][rows_b]
                tasks.append(
                    _ContractionTask(
                        left=B_a,
                        weights=coupling_flat[:, :, state_a, state_b],
                        right=B_b,
                        row_positions=positions_a,
                        column_positions=positions_b,
                        symmetric=False,
                    )
                )

    blocks = _evaluate_contraction_tasks(tasks, batched=batched_flags[0])
    for task, block in zip(tasks, blocks, strict=True):
        rows = task.row_positions
        columns = task.column_positions
        Vmat[:, rows[:, None], columns[None, :]] = block
        if not task.symmetric:
            Vmat[:, columns[:, None], rows[None, :]] = np.swapaxes(block, -2, -1)

    return Vmat if batched_flags[0] else Vmat[0]
