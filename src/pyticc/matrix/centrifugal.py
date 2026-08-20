from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from pyticc.basis.angle import lambda_plus
from pyticc.basis.channel import Channel, ChannelBasis, ChannelBasisElectricSF
from pyticc.system import MolInnerState

CentrifugalKey = tuple[MolInnerState, MolInnerState, int]


# ----------------------------------------------------------------------------------------
def _centrifugal_key(channel: Channel) -> CentrifugalKey:
    """Return the quantum numbers that identify one Coriolis-coupled K ladder."""
    return channel.mis_X, channel.mis_Y, channel.j_couple


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

        Umat[local_index, local_index] = basis.Jtot * (basis.Jtot + 1) + channel.j_couple * (channel.j_couple + 1) - 2 * channel.K**2

    if not coriolis:
        return Umat

    for K_to_index in groups.values():
        for K, local_index in K_to_index.items():
            next_index = K_to_index.get(K + 1)
            if next_index is None:
                continue

            channel = channels[local_index]
            boundary_factor = np.sqrt(2.0) if K == 0 else 1.0
            coupling = -boundary_factor * lambda_plus(basis.Jtot, K) * lambda_plus(channel.j_couple, K)
            Umat[local_index, next_index] = coupling
            Umat[next_index, local_index] = coupling

    return Umat


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def get_Umat_ElectricSF(
    basis: ChannelBasisElectricSF,
    channel_indices: Sequence[int] | None = None,
) -> NDArray[np.float64]:
    r"""
    Get the dimensionless centrifugal matrix in the space-fixed representation.

    Formula:
        For the SF channel

        |eta; M> = |phi_{alpha m}> |l m_l>,

        the end-over-end operator is diagonal:

        U_{eta' eta}^{SF}
          = <eta'|L^2|eta>
          = l(l+1) delta_{eta' eta}.

        There are no Coriolis off-diagonal elements in this uncoupled SF basis.
        Row and column order follows channel_indices, or the complete channel
        basis when channel_indices is None.

    Inputs:
        basis: ChannelBasisElectricSF - complete electric-field SF channel basis
        channel_indices: Sequence[int] | None - selected complete-basis
            positions, shape (n_selected_channel,)

    Returns:
        Umat: NDArray[np.float64] - dimensionless diagonal SF centrifugal
            matrix, shape (n_selected_channel, n_selected_channel)
    """
    if channel_indices is None:
        indices = tuple(range(basis.n_channel))
    else:
        indices = tuple(channel_indices)

    diagonal = np.asarray([basis[index].l * (basis[index].l + 1) for index in indices], dtype=np.float64)
    return np.diag(diagonal)


# ----------------------------------------------------------------------------------------
