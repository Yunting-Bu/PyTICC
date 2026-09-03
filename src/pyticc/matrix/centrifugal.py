from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from pyticc.basis.angle import lambda_plus
from pyticc.basis.channel import Channel, ChannelBasis, ChannelBasisElectricSF
from pyticc.fine_structure.channel import FSChannelBasis
from pyticc.fine_structure.diatom_diatom import FSDiatomDiatomBasis
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

    For molecule-exchange-adapted AB+CD channels this same expression is
    T_eta.T U T_eta: exchange preserves each canonical state-pair/j12 ladder,
    and the expansion coefficients are independent of K. No additional
    exchange normalization is applied to the retained ladder matrix elements.

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
def get_Umat_FS_BF(
    basis: FSChannelBasis,
    channel_indices: Sequence[int] | None = None,
    *,
    coriolis: bool = True,
) -> NDArray[np.float64]:
    r"""
    Return the BF centrifugal matrix for integer or half-integer open-shell channels.

    Formula:
        U_KK = J(J+1)+j(j+1)-2K^2,

        U_{K+1,K} = -b_K sqrt[J(J+1)-K(K+1)]
          sqrt[j(j+1)-K(K+1)],

        where b_K=sqrt(2) at the parity-adapted K=0 boundary and one otherwise.
        For half-integer J and j, the primitive Coriolis coupling between
        K=+1/2 and K=-1/2 becomes the diagonal boundary term

        Delta U_(1/2,1/2) = -s (J+1/2)(j+1/2),
        s = P epsilon (-1)^(j-J).

    Inputs:
        basis: FSChannelBasis - fixed-(J,P) fine-structure channels
        channel_indices: Sequence[int] | None - selected complete-basis positions
        coriolis: bool - whether to retain nearest-neighbor K coupling

    Returns:
        matrix: NDArray[np.float64] - dimensionless centrifugal matrix
    """
    indices = tuple(range(basis.n_channel)) if channel_indices is None else tuple(channel_indices)
    channels = tuple(basis[index] for index in indices)
    matrix = np.zeros((len(channels), len(channels)), dtype=np.float64)
    J = basis.two_J / 2.0
    groups: dict[tuple[int, int], dict[int, int]] = {}
    for local_index, channel in enumerate(channels):
        block = basis.monomer.blocks[channel.block]
        j = block.two_j / 2.0
        K = channel.two_K / 2.0
        matrix[local_index, local_index] = J * (J + 1.0) + j * (j + 1.0) - 2.0 * K**2
        if channel.two_K == 1:
            exponent = (block.two_j - basis.two_J) // 2
            parity_phase = basis.system_parity * block.parity * (-1) ** exponent
            matrix[local_index, local_index] -= parity_phase * (J + 0.5) * (j + 0.5)
        groups.setdefault((channel.block, channel.tau), {})[channel.two_K] = local_index
    if not coriolis:
        return matrix
    for (block_index, _), K_to_index in groups.items():
        j = basis.monomer.blocks[block_index].two_j / 2.0
        for two_K, local_index in K_to_index.items():
            next_index = K_to_index.get(two_K + 2)
            if next_index is None:
                continue
            K = two_K / 2.0
            boundary = np.sqrt(2.0) if two_K == 0 else 1.0
            coupling = -boundary * np.sqrt(J * (J + 1.0) - K * (K + 1.0)) * np.sqrt(j * (j + 1.0) - K * (K + 1.0))
            matrix[local_index, next_index] = coupling
            matrix[next_index, local_index] = coupling
    return matrix


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def get_Umat_FS_DiatomDiatom_BF(
    basis: FSDiatomDiatomBasis,
    channel_indices: Sequence[int] | None = None,
    *,
    coriolis: bool = True,
) -> NDArray[np.float64]:
    r"""
    Return the BF centrifugal matrix for two fine-structure diatoms.

    Exchange adaptation preserves each canonical internal-state/j12 ladder;
    its coefficients are independent of K. The same expressions therefore
    give T_eta.T U T_eta on an exchange-adapted basis without extra factors.

    Formula:
        With ``j_12=j_X+j_Y`` and nonnegative parity-adapted helicity K,

        U_KK = J(J+1)+j_12(j_12+1)-2K^2,

        U_(K+1,K) = -b_K
          sqrt[J(J+1)-K(K+1)]
          sqrt[j_12(j_12+1)-K(K+1)],

        where ``b_K=sqrt(2)`` at the integer K=0 boundary and one otherwise.
        For half-integer J and j_12, folding the primitive K=-1/2 channel into
        the parity-adapted K=+1/2 channel adds

        Delta U_(1/2,1/2)
          = -s (J+1/2)(j_12+1/2),

        s = P epsilon_X epsilon_Y (-1)^(j_12-J).

        The matrix is dimensionless. Its row and column order follows
        ``channel_indices`` or the complete basis order. Coriolis coupling is
        restricted to channels with identical monomer eigenlevels and j_12.

    Inputs:
        basis: FSDiatomDiatomBasis - fixed-(J,P) two-diatom channels
        channel_indices: Sequence[int] | None - selected complete-basis positions
        coriolis: bool - whether to retain nearest-neighbor K coupling

    Returns:
        matrix: NDArray[np.float64] - dimensionless BF centrifugal matrix,
            shape ``(n_selected,n_selected)``
    """
    indices = tuple(range(basis.n_channel)) if channel_indices is None else tuple(channel_indices)
    channels = tuple(basis[index] for index in indices)
    matrix = np.zeros((len(channels), len(channels)), dtype=np.float64)
    J = basis.two_J / 2.0
    groups: dict[tuple[int, int, int, int, int], dict[int, int]] = {}

    for local_index, channel in enumerate(channels):
        block_X = basis.monomer_X.blocks[channel.block_X]
        block_Y = basis.monomer_Y.blocks[channel.block_Y]
        j12 = channel.two_j12 / 2.0
        K = channel.two_K / 2.0
        matrix[local_index, local_index] = J * (J + 1.0) + j12 * (j12 + 1.0) - 2.0 * K**2
        if channel.two_K == 1:
            exponent = (channel.two_j12 - basis.two_J) // 2
            parity_phase = basis.system_parity * block_X.parity * block_Y.parity * (-1) ** exponent
            matrix[local_index, local_index] -= parity_phase * (J + 0.5) * (j12 + 0.5)
        key = (channel.block_X, channel.tau_X, channel.block_Y, channel.tau_Y, channel.two_j12)
        groups.setdefault(key, {})[channel.two_K] = local_index

    if not coriolis:
        return matrix

    for key, K_to_index in groups.items():
        j12 = key[-1] / 2.0
        for two_K, local_index in K_to_index.items():
            next_index = K_to_index.get(two_K + 2)
            if next_index is None:
                continue
            K = two_K / 2.0
            boundary = np.sqrt(2.0) if two_K == 0 else 1.0
            coupling = -boundary * np.sqrt(J * (J + 1.0) - K * (K + 1.0)) * np.sqrt(j12 * (j12 + 1.0) - K * (K + 1.0))
            matrix[local_index, next_index] = coupling
            matrix[next_index, local_index] = coupling
    return matrix


# ----------------------------------------------------------------------------------------
