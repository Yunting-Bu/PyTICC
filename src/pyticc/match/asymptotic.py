from collections.abc import Sequence

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike, NDArray

from pyticc.basis.channel import ChannelBasis
from pyticc.fine_structure.channel import FSChannelBasis
from pyticc.fine_structure.diatom_diatom import FSDiatomDiatomBasis
from pyticc.matrix.centrifugal import get_Umat_BF, get_Umat_FS_BF, get_Umat_FS_DiatomDiatom_BF


# ----------------------------------------------------------------------------------------
def get_Bmat_BF_to_SF(
    basis: ChannelBasis,
    channel_indices: Sequence[int] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Get the body-fixed to space-fixed asymptotic transformation.

    Channels are grouped by identical monomer states and coupled angular momentum.
    Each corresponding centrifugal submatrix is diagonalized independently. This
    avoids mixing degenerate eigenvectors from different asymptotic internal states.

    Formula:
        B.T @ U_BF @ B = diag(L * (L + 1)),
        L = sqrt(eigenvalue + 1 / 4) - 1 / 2.

    Inputs:
        basis: ChannelBasis - complete field-free channel basis
        channel_indices: Sequence[int] | None - complete-basis positions for one
            propagation block, shape (n_channel,)

    Returns:
        Bmat: NDArray[np.float64] - BF-to-SF orthogonal transformation matrix,
            shape (n_channel, n_channel)
        L: NDArray[np.float64] - orbital angular momenta in SF channel order, shape
            (n_channel,)
    """
    indices = tuple(range(basis.n_channel)) if channel_indices is None else tuple(channel_indices)
    if not indices:
        message = "At least one channel is required for the BF-to-SF transformation"
        logger.error(message)
        raise ValueError(message)

    channels = tuple(basis[index] for index in indices)
    Umat = get_Umat_BF(basis, indices)
    n_channel = len(indices)
    Bmat = np.zeros((n_channel, n_channel), dtype=np.float64)
    L = np.empty(n_channel, dtype=np.float64)

    groups: dict[tuple[object, object, int], list[int]] = {}
    for local_index, channel in enumerate(channels):
        groups.setdefault((channel.mis_X, channel.mis_Y, channel.j_couple), []).append(local_index)

    for positions_list in groups.values():
        positions = np.asarray(positions_list, dtype=np.int64)
        eigenvalues, eigenvectors = np.linalg.eigh(Umat[np.ix_(positions, positions)])
        if np.min(eigenvalues) < -1.0e-10:
            message = f"Centrifugal matrix has a negative eigenvalue {np.min(eigenvalues)}"
            logger.error(message)
            raise ValueError(message)

        channel = channels[positions_list[0]]
        phase = (-1) ** (channel.j_couple + min(basis.Jtot, channel.j_couple))
        signs = np.where(eigenvectors[-1] * phase < 0.0, -1.0, 1.0)
        eigenvectors *= signs

        Bmat[np.ix_(positions, positions)] = eigenvectors
        eigenvalues = np.maximum(eigenvalues, 0.0)
        L[positions] = np.sqrt(eigenvalues + 0.25) - 0.5

    return Bmat, L


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def transform_logD_BF_to_SF(Ymat: ArrayLike, Bmat: ArrayLike) -> NDArray[np.float64] | NDArray[np.complex128]:
    r"""
    Transform one or more log-derivative matrices from BF to SF representation.

    Formula:
        Y_SF = B.T @ Y_BF @ B.

    Inputs:
        Ymat: ArrayLike - BF log-derivative matrix or batch, shape
            (..., n_channel, n_channel)
        Bmat: ArrayLike - BF-to-SF transformation matrix, shape
            (n_channel, n_channel)

    Returns:
        Y_SF: NDArray[np.float64] | NDArray[np.complex128] - transformed matrices,
            shape (..., n_channel, n_channel)
    """
    Ymat_array = np.asarray(Ymat)
    Bmat_array = np.asarray(Bmat, dtype=np.float64)
    if Bmat_array.ndim != 2 or Bmat_array.shape[0] != Bmat_array.shape[1]:
        message = f"Bmat must be square, but got shape={Bmat_array.shape}"
        logger.error(message)
        raise ValueError(message)
    if Ymat_array.ndim < 2 or Ymat_array.shape[-2:] != Bmat_array.shape:
        message = f"Ymat must end with matrix shape {Bmat_array.shape}, but got shape={Ymat_array.shape}"
        logger.error(message)
        raise ValueError(message)

    return np.einsum("ia,...ij,jb->...ab", Bmat_array, Ymat_array, Bmat_array, optimize=True)


# ----------------------------------------------------------------------------------------
def get_Bmat_FS_BF_to_SF(
    basis: FSChannelBasis,
    channel_indices: Sequence[int] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Diagonalize open-shell BF centrifugal ladders into asymptotic orbital L.

    Formula:
        B.T U_BF B = diag[L(L+1)],
        L = sqrt(lambda+1/4)-1/2.

        Channels are diagonalized independently for every molecular eigenstate
        (block,tau), so degenerate thresholds never mix accidentally.

    Inputs:
        basis: FSChannelBasis - fine-structure BF channels
        channel_indices: Sequence[int] | None - selected complete-basis positions

    Returns:
        Bmat: NDArray[np.float64] - orthogonal BF-to-SF transformation
        L: NDArray[np.float64] - asymptotic end-over-end angular momenta
    """
    indices = tuple(range(basis.n_channel)) if channel_indices is None else tuple(channel_indices)
    if not indices:
        raise ValueError("At least one fine-structure channel is required")
    channels = tuple(basis[index] for index in indices)
    Umat = get_Umat_FS_BF(basis, indices)
    Bmat = np.zeros((len(indices), len(indices)), dtype=np.float64)
    L = np.empty(len(indices), dtype=np.float64)
    groups: dict[tuple[int, int], list[int]] = {}
    for local_index, channel in enumerate(channels):
        groups.setdefault((channel.block, channel.tau), []).append(local_index)
    for positions_list in groups.values():
        positions = np.asarray(positions_list, dtype=np.int64)
        eigenvalues, eigenvectors = np.linalg.eigh(Umat[np.ix_(positions, positions)])
        signs = np.where(eigenvectors[-1] < 0.0, -1.0, 1.0)
        eigenvectors *= signs
        Bmat[np.ix_(positions, positions)] = eigenvectors
        L[positions] = np.sqrt(np.maximum(eigenvalues, 0.0) + 0.25) - 0.5
    return Bmat, L


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def get_Bmat_FS_DiatomDiatom_BF_to_SF(
    basis: FSDiatomDiatomBasis,
    channel_indices: Sequence[int] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Transform two-fine-structure-diatom BF channels to asymptotic orbital L.

    Formula:
        For every fixed internal ladder

        q = (block_X,tau_X,block_Y,tau_Y,j_12),

        independently diagonalize the dimensionless end-over-end operator,

        B_q.T U_q B_q = diag[L(L+1)],
        L = sqrt(lambda+1/4)-1/2.

        Here ``U_q`` contains the parity-adapted integer- or half-integer K
        ladder returned by ``get_Umat_FS_DiatomDiatom_BF``. The monomer
        eigenstates are orthonormal, and B is real orthogonal. Each eigenvector
        phase is fixed by making its last K component nonnegative. Row order is
        BF channel order; column order is increasing centrifugal eigenvalue
        within each internal ladder.

    Inputs:
        basis: FSDiatomDiatomBasis - complete fine-structure BF channels
        channel_indices: Sequence[int] | None - selected complete-basis positions

    Returns:
        Bmat: NDArray[np.float64] - BF-to-SF orthogonal transformation,
            shape ``(n_selected,n_selected)``
        L: NDArray[np.float64] - asymptotic end-over-end angular momenta,
            shape ``(n_selected,)``
    """
    indices = tuple(range(basis.n_channel)) if channel_indices is None else tuple(channel_indices)
    if not indices:
        message = "At least one two-diatom fine-structure channel is required"
        logger.error(message)
        raise ValueError(message)

    channels = tuple(basis[index] for index in indices)
    Umat = get_Umat_FS_DiatomDiatom_BF(basis, indices)
    Bmat = np.zeros((len(indices), len(indices)), dtype=np.float64)
    L = np.empty(len(indices), dtype=np.float64)
    groups: dict[tuple[int, int, int, int, int], list[int]] = {}
    for local_index, channel in enumerate(channels):
        key = (channel.block_X, channel.tau_X, channel.block_Y, channel.tau_Y, channel.two_j12)
        groups.setdefault(key, []).append(local_index)

    for positions_list in groups.values():
        positions = np.asarray(positions_list, dtype=np.int64)
        eigenvalues, eigenvectors = np.linalg.eigh(Umat[np.ix_(positions, positions)])
        if np.min(eigenvalues) < -1.0e-10:
            message = f"Two-diatom centrifugal matrix has a negative eigenvalue {np.min(eigenvalues)}"
            logger.error(message)
            raise ValueError(message)
        signs = np.where(eigenvectors[-1] < 0.0, -1.0, 1.0)
        eigenvectors *= signs
        Bmat[np.ix_(positions, positions)] = eigenvectors
        L[positions] = np.sqrt(np.maximum(eigenvalues, 0.0) + 0.25) - 0.5

    return Bmat, L


# ----------------------------------------------------------------------------------------
