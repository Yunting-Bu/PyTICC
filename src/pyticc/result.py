from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyticc.basis.channel import ChannelBasis, OpenClosedChannels
from pyticc.basis.kblock import KBlock
from pyticc.system import Approx

LogDArray = NDArray[np.float64] | NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class Timing:
    """Elapsed wall-clock and process CPU times in seconds."""

    wall_seconds: float
    cpu_seconds: float

    def __str__(self) -> str:
        return f"wall={self.wall_seconds:.3f} s, CPU={self.cpu_seconds:.3f} s"


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ScatteringResult:
    """
    Field-free scattering result for one Jtot and system-parity block.

    Members:
        basis: ChannelBasis - complete body-fixed channel basis
        Etot: NDArray[np.float64] - total energies in atomic units, shape
            (n_energy,)
        open_closed: OpenClosedChannels - open and closed channels at each energy
        Y_BF: NDArray - final body-fixed log-derivative matrices, shape
            (n_energy, n_channel, n_channel)
        Bmat: NDArray[np.float64] - body-fixed to space-fixed transformation, shape
            (n_channel, n_channel)
        L: NDArray[np.float64] - space-fixed orbital angular momenta, shape
            (n_channel,)
        Y_SF: NDArray - final space-fixed log-derivative matrices, shape
            (n_energy, n_channel, n_channel)
        Smat: tuple[NDArray[np.complex128], ...] - one matrix per energy; element i
            has shape (n_open[i], n_open[i])
        timing: Timing | None - elapsed solver or end-to-end run time
    """

    basis: ChannelBasis
    Etot: NDArray[np.float64]
    open_closed: OpenClosedChannels
    Y_BF: LogDArray
    Bmat: NDArray[np.float64]
    L: NDArray[np.float64]
    Y_SF: LogDArray
    Smat: tuple[NDArray[np.complex128], ...]
    timing: Timing | None = None

    @property
    def open_channel_indices(self) -> tuple[NDArray[np.int64], ...]:
        """Return one index array with shape (n_open[i],) for each energy."""
        return tuple(np.asarray(np.flatnonzero(mask), dtype=np.int64) for mask in self.open_closed.open_mask)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class KBlockResult:
    """
    Matched result for one CS or NNCC propagation block.

    Members:
        block: KBlock - propagated K window and ownership information
        open_channel_indices: tuple[NDArray[np.int64], ...] - one global index array
            with shape (n_open_block[i],) for each energy
        Y_BF: NDArray - final block log-derivative matrices in the BF basis, shape
            (n_energy, n_channel_block, n_channel_block)
        Bmat: NDArray[np.float64] - block BF-to-asymptotic transformation, shape
            (n_channel_block, n_channel_block)
        L: NDArray[np.float64] - effective orbital angular momenta in the block,
            shape (n_channel_block,)
        Y_asymptotic: NDArray - final block log derivatives in the asymptotic basis,
            shape (n_energy, n_channel_block, n_channel_block)
        Smat_asymptotic: tuple[NDArray[np.complex128], ...] - one matched block
            matrix per energy; element i has shape
            (n_open_block[i], n_open_block[i])
    """

    block: KBlock
    open_channel_indices: tuple[NDArray[np.int64], ...]
    Y_BF: LogDArray
    Bmat: NDArray[np.float64]
    L: NDArray[np.float64]
    Y_asymptotic: LogDArray
    Smat_asymptotic: tuple[NDArray[np.complex128], ...]


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CoupledStatesResult:
    """
    Field-free CS or NNCC result for one Jtot and system-parity block.

    NNCC block scattering matrices are not one global unitary scattering matrix, so
    they remain separated in ``blocks`` until observables are implemented.

    Members:
        basis: ChannelBasis - complete body-fixed channel basis
        Etot: NDArray[np.float64] - total energies in atomic units, shape
            (n_energy,)
        open_closed: OpenClosedChannels - full-basis open and closed channels
        approx: Approx - CS or NNCC approximation
        blocks: tuple[KBlockResult, ...] - independently propagated and matched blocks
        timing: Timing | None - elapsed solver or end-to-end run time
    """

    basis: ChannelBasis
    Etot: NDArray[np.float64]
    open_closed: OpenClosedChannels
    approx: Approx
    blocks: tuple[KBlockResult, ...]
    timing: Timing | None = None


# ----------------------------------------------------------------------------------------
