from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from pyticc.basis.angle import lambda_plus
from pyticc.basis.channel import Channel, ChannelBasis

CentrifugalKey = tuple[int | None, int | None, int, int | None, int | None, int, int]


# ----------------------------------------------------------------------------------------
def _centrifugal_key(channel: Channel) -> CentrifugalKey:
    """Return the quantum numbers that identify one Coriolis-coupled K ladder."""
    return (
        channel.mis_X.v,
        channel.mis_X.t,
        channel.mis_X.j,
        channel.mis_Y.v,
        channel.mis_Y.t,
        channel.mis_Y.j,
        channel.j_couple,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def get_Umat_BF(
    basis: ChannelBasis,
    channel_indices: Sequence[int] | None = None,
    *,
    coriolis: bool = True,
) -> NDArray[np.float64]:
    r"""
    Get the dimensionless centrifugal matrix in the body-fixed representation.

    The matrix contains the diagonal centrifugal terms and, when coriolis is True,
    the nearest-neighbor K-to-K+1 Coriolis couplings. Its row and column order follows
    channel_indices, or the complete channel basis when channel_indices is None.

    Inputs:
        basis: ChannelBasis - complete field-free channel basis
        channel_indices: Sequence[int] | None - selected complete-basis positions,
            shape (n_selected_channel,)
        coriolis: bool - whether to include nearest-neighbor Coriolis couplings

    Returns:
        Umat: NDArray[np.float64] - dimensionless body-fixed centrifugal matrix,
            shape (n_selected_channel, n_selected_channel)
    """
    if channel_indices is None:
        indices = tuple(range(basis.n_channel))
    else:
        indices = tuple(channel_indices)

    channels = tuple(basis[index] for index in indices)
    Umat = np.zeros((len(channels), len(channels)), dtype=np.float64)

    groups: dict[CentrifugalKey, dict[int, int]] = {}
    for local_index, channel in enumerate(channels):
        K_to_index = groups.setdefault(_centrifugal_key(channel), {})
        K_to_index[channel.K] = local_index

        Umat[local_index, local_index] = channel.Jtot * (channel.Jtot + 1) + channel.j_couple * (channel.j_couple + 1) - 2 * channel.K**2

    if not coriolis:
        return Umat

    for K_to_index in groups.values():
        for K, local_index in K_to_index.items():
            next_index = K_to_index.get(K + 1)
            if next_index is None:
                continue

            channel = channels[local_index]
            boundary_factor = np.sqrt(2.0) if K == 0 else 1.0
            coupling = -boundary_factor * lambda_plus(channel.Jtot, K) * lambda_plus(channel.j_couple, K)
            Umat[local_index, next_index] = coupling
            Umat[next_index, local_index] = coupling

    return Umat


# ----------------------------------------------------------------------------------------
