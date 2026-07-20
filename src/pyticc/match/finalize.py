from typing import cast

import numpy as np
from numpy.typing import NDArray

from pyticc.basis.channel import ChannelBasis
from pyticc.basis.kblock import KBlock
from pyticc.energy import EnergyInput, get_Etot
from pyticc.match.asymptotic import get_Bmat_BF_to_SF, transform_logD_BF_to_SF
from pyticc.match.smatrix import get_Smat
from pyticc.result import KBlockResult, LogDArray, ScatteringResult


# ----------------------------------------------------------------------------------------
def finalize_scattering(
    basis: ChannelBasis,
    Y_BF: LogDArray,
    Etot: EnergyInput,
    reduced_mass: float,
    Rmatch: float,
) -> ScatteringResult:
    """
    Transform, asymptotically match, and package one exact-CC result.

    Inputs:
        basis: ChannelBasis - complete body-fixed channel basis
        Y_BF: LogDArray - final body-fixed log derivatives, shape
            (n_energy, n_channel, n_channel)
        Etot: EnergyInput - total energies with shape (n_energy,), or a one-column
            text file
        reduced_mass: float - collision reduced mass in atomic units
        Rmatch: float - asymptotic matching distance in atomic units

    Returns:
        result: ScatteringResult - matched body-fixed, space-fixed, and S matrices
    """
    energies = get_Etot(Etot)
    Bmat, L = get_Bmat_BF_to_SF(basis)
    Y_BF_array = cast(LogDArray, np.asarray(Y_BF))
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


# ----------------------------------------------------------------------------------------
def finalize_K_block(
    basis: ChannelBasis,
    block: KBlock,
    Y_BF: LogDArray,
    Etot: EnergyInput,
    reduced_mass: float,
    Rmatch: float,
) -> KBlockResult:
    """
    Transform, asymptotically match, and package one CS or NNCC K block.

    Inputs:
        basis: ChannelBasis - complete body-fixed channel basis
        block: KBlock - propagated channel subset and result ownership
        Y_BF: LogDArray - final block log derivatives, shape
            (n_energy, n_channel_block, n_channel_block)
        Etot: EnergyInput - total energies with shape (n_energy,), or a one-column
            text file
        reduced_mass: float - collision reduced mass in atomic units
        Rmatch: float - asymptotic matching distance in atomic units

    Returns:
        result: KBlockResult - matched log derivatives and S matrices for this block
    """
    energies = get_Etot(Etot)
    indices = np.asarray(block.channel_indices, dtype=np.int64)
    E_int = basis.E_int[indices]
    Bmat, L = get_Bmat_BF_to_SF(basis, block.channel_indices)
    Y_BF_array = cast(LogDArray, np.asarray(Y_BF))
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


# ----------------------------------------------------------------------------------------
