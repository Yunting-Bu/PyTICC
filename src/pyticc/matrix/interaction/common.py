"""Shared basis and contraction machinery for scalar interaction potentials."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

import numpy as np
from loguru import logger
from numpy.typing import NDArray


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class VBasisBF:
    """PES-independent body-fixed basis used to contract a scalar interaction."""

    n_channel: int
    grid_shape: tuple[int, ...]
    channel_indices: dict[int, tuple[int, ...]]
    B_real: dict[int, NDArray[np.float64]]
    B_imag: dict[int, NDArray[np.float64]] | None
    normalization: float = 1.0


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _contract_block(
    B_real: NDArray[np.float64],
    B_imag: NDArray[np.float64] | None,
    normalization: float,
    potential: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Contract one exact-K basis block with a flattened potential grid."""
    weighted = B_real * potential[None, :]
    Vmat = weighted @ B_real.T
    if B_imag is not None:
        weighted = B_imag * potential[None, :]
        Vmat += weighted @ B_imag.T
    Vmat *= normalization
    return 0.5 * (Vmat + Vmat.T)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def contract(
    basis: VBasisBF,
    potential: NDArray[np.float64],
    channel_indices: Sequence[int] | None = None,
) -> NDArray[np.float64]:
    """Contract a scalar potential grid into body-fixed channel matrices.

    Inputs:
        basis: VBasisBF - precomputed geometry-specific interaction basis
        potential: NDArray[np.float64] - scalar or radial-batched potential grid
        channel_indices: Sequence[int] | None - optional complete-basis positions

    Returns:
        Vmat: NDArray[np.float64] - scalar or radial-batched channel matrices
    """
    indices = tuple(range(basis.n_channel)) if channel_indices is None else tuple(channel_indices)
    n_grid = prod(basis.grid_shape)
    values = np.asarray(potential, dtype=np.float64)
    if values.shape == basis.grid_shape or values.shape == (n_grid,):
        batches = values.reshape(1, n_grid)
        batched = False
    elif values.ndim == len(basis.grid_shape) + 1 and values.shape[1:] == basis.grid_shape:
        batches = values.reshape(values.shape[0], n_grid)
        batched = True
    elif values.ndim == 2 and values.shape[1] == n_grid:
        batches = values
        batched = True
    else:
        message = f"Potential grid has shape {values.shape}, but V basis requires {basis.grid_shape} with an optional leading R axis"
        logger.error(message)
        raise ValueError(message)

    global_to_local = {global_index: local_index for local_index, global_index in enumerate(indices)}
    Vmat = np.zeros((batches.shape[0], len(indices), len(indices)), dtype=np.float64)
    for K, group_indices in basis.channel_indices.items():
        selected = tuple(
            (basis_index, global_to_local[global_index]) for basis_index, global_index in enumerate(group_indices) if global_index in global_to_local
        )
        if not selected:
            continue

        basis_positions, output_positions = zip(*selected, strict=True)
        B_real = basis.B_real[K][np.asarray(basis_positions)]
        B_imag = None if basis.B_imag is None else basis.B_imag[K][np.asarray(basis_positions)]
        for radial_index in range(batches.shape[0]):
            values_at_R = np.asarray(batches[radial_index], dtype=np.float64)
            block = _contract_block(B_real, B_imag, basis.normalization, values_at_R)
            Vmat[radial_index][np.ix_(output_positions, output_positions)] = block

    return Vmat if batched else Vmat[0]


# ----------------------------------------------------------------------------------------
