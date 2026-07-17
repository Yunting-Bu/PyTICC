from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyticc.basis.channel import ChannelBasis, OpenClosedChannels
from pyticc.basis.kblock import KBlock
from pyticc.constants import AU2CM
from pyticc.energy import EnergyInput, get_Etot
from pyticc.match import get_Bmat_BF_to_SF, get_Smat, transform_logD_BF_to_SF
from pyticc.system import Approx


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ScatteringResult:
    """
    Field-free scattering result for one J and parity block.

    Members:
        basis: ChannelBasis - complete body-fixed channel basis
        Etot: NDArray[np.float64] - total energies in atomic units
        open_closed: OpenClosedChannels - open and closed channels at each energy
        Y_BF: NDArray - final body-fixed log-derivative matrices
        Bmat: NDArray[np.float64] - body-fixed to space-fixed transformation
        L: NDArray[np.float64] - space-fixed orbital angular momenta
        Y_SF: NDArray - final space-fixed log-derivative matrices
        Smat: tuple[NDArray[np.complex128], ...] - open-channel scattering matrices
    """

    basis: ChannelBasis
    Etot: NDArray[np.float64]
    open_closed: OpenClosedChannels
    Y_BF: NDArray[np.float64] | NDArray[np.complex128]
    Bmat: NDArray[np.float64]
    L: NDArray[np.float64]
    Y_SF: NDArray[np.float64] | NDArray[np.complex128]
    Smat: tuple[NDArray[np.complex128], ...]

    @property
    def open_channel_indices(self) -> tuple[NDArray[np.int64], ...]:
        """Return complete-basis positions represented by each scattering matrix."""
        return tuple(np.asarray(np.flatnonzero(mask), dtype=np.int64) for mask in self.open_closed.open_mask)

    def print_summary(self) -> None:
        """Print channel counts and open-channel counts over the energy grid."""
        print(f"Channels: {self.basis.n_channel}")
        for energy, n_open in zip(self.Etot, self.open_closed.n_open, strict=True):
            print(f"Etot = {energy * AU2CM:12.6f} cm-1   open channels = {n_open}")


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class KBlockResult:
    """
    Matched result for one CS or NNCC propagation block.

    Members:
        block: KBlock - propagated K window and ownership information
        open_channel_indices: tuple[NDArray[np.int64], ...] - global open-channel positions at each energy
        Y_BF: NDArray - final block log-derivative matrices in the BF basis
        Bmat: NDArray[np.float64] - block BF-to-asymptotic transformation
        L: NDArray[np.float64] - effective orbital angular momenta in the block
        Y_asymptotic: NDArray - final block log derivatives in the asymptotic basis
        Smat_asymptotic: tuple[NDArray[np.complex128], ...] - matched block scattering matrices
    """

    block: KBlock
    open_channel_indices: tuple[NDArray[np.int64], ...]
    Y_BF: NDArray[np.float64] | NDArray[np.complex128]
    Bmat: NDArray[np.float64]
    L: NDArray[np.float64]
    Y_asymptotic: NDArray[np.float64] | NDArray[np.complex128]
    Smat_asymptotic: tuple[NDArray[np.complex128], ...]


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CoupledStatesResult:
    """
    Field-free CS or NNCC result for one J and parity block.

    NNCC block scattering matrices are not one global unitary scattering matrix, so
    they remain separated in ``blocks`` until observables are implemented.

    Members:
        basis: ChannelBasis - complete body-fixed channel basis
        Etot: NDArray[np.float64] - total energies in atomic units
        open_closed: OpenClosedChannels - full-basis open and closed channels
        approx: Approx - CS or NNCC approximation
        blocks: tuple[KBlockResult, ...] - independently propagated and matched blocks
    """

    basis: ChannelBasis
    Etot: NDArray[np.float64]
    open_closed: OpenClosedChannels
    approx: Approx
    blocks: tuple[KBlockResult, ...]

    def print_summary(self) -> None:
        """Print approximation, block sizes, and open-channel counts."""
        print(f"Approximation: {self.approx.value}")
        print(f"Channels: {self.basis.n_channel}   K blocks: {len(self.blocks)}")
        for block_result in self.blocks:
            print(block_result.block)
        for energy, n_open in zip(self.Etot, self.open_closed.n_open, strict=True):
            print(f"Etot = {energy * AU2CM:12.6f} cm-1   open channels = {n_open}")


# ----------------------------------------------------------------------------------------
def _build_result(
    basis: ChannelBasis,
    Y_BF: NDArray[np.float64] | NDArray[np.complex128],
    Etot: EnergyInput,
    reduced_mass: float,
    Rmatch: float,
) -> ScatteringResult:
    energies = get_Etot(Etot)
    Bmat, L = get_Bmat_BF_to_SF(basis)
    Y_BF_array = np.asarray(Y_BF)
    Y_SF = transform_logD_BF_to_SF(Y_BF_array, Bmat)
    Smat = get_Smat(Y_SF, Rmatch, energies, reduced_mass, basis.E_int, L)
    return ScatteringResult(
        basis=basis,
        Etot=energies,
        open_closed=basis.open_closed(energies),
        Y_BF=Y_BF_array,
        Bmat=Bmat,
        L=L,
        Y_SF=Y_SF,
        Smat=Smat,
    )


# ----------------------------------------------------------------------------------------
def _build_K_block_result(
    basis: ChannelBasis,
    block: KBlock,
    Y_BF: NDArray[np.float64] | NDArray[np.complex128],
    Etot: EnergyInput,
    reduced_mass: float,
    Rmatch: float,
) -> KBlockResult:
    """Match one propagated CS or NNCC K block in its asymptotic basis."""
    energies = get_Etot(Etot)
    indices = np.asarray(block.channel_indices, dtype=np.int64)
    E_int = basis.E_int[indices]
    Bmat, L = get_Bmat_BF_to_SF(basis, block.channel_indices)
    Y_BF_array = np.asarray(Y_BF)
    Y_asymptotic = transform_logD_BF_to_SF(Y_BF_array, Bmat)
    Smat_asymptotic = get_Smat(Y_asymptotic, Rmatch, energies, reduced_mass, E_int, L)

    global_open_indices: list[NDArray[np.int64]] = []
    for energy in energies:
        local_open = np.asarray(np.flatnonzero(E_int < energy), dtype=np.int64)
        global_open_indices.append(indices[local_open])

    return KBlockResult(
        block=block,
        open_channel_indices=tuple(global_open_indices),
        Y_BF=Y_BF_array,
        Bmat=Bmat,
        L=L,
        Y_asymptotic=Y_asymptotic,
        Smat_asymptotic=Smat_asymptotic,
    )
