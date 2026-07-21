"""Atom-diatom interaction-matrix basis."""

from math import prod
from typing import cast

import numpy as np
from numpy.typing import NDArray

from pyticc.basis.angle import norm_YjK
from pyticc.basis.channel import ChannelBasis
from pyticc.basis.podvr import RovibPODVR
from pyticc.matrix.interaction import VBasisBF


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
