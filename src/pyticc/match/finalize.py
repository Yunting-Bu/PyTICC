from typing import cast

import numpy as np
from numpy.typing import NDArray

from pyticc.basis.channel import ChannelBasis, ChannelBasisElectricSF
from pyticc.basis.delves import DelvesBasis
from pyticc.basis.kblock import KBlock
from pyticc.energy import EnergyInput, get_Etot
from pyticc.match.asymptotic import get_Bmat_BF_to_SF, transform_logD_BF_to_SF
from pyticc.match.delves import build_delves_asymptotic_basis, transform_logD_to_delves_channels
from pyticc.match.delves_bessel import get_delves_Smat
from pyticc.match.smatrix import get_Smat
from pyticc.pes.total import TotalPES
from pyticc.propagation.delves import DelvesPropagationResult
from pyticc.result import KBlockResult, LogDArray, ReactiveScatteringResult, ScatteringResult


# ----------------------------------------------------------------------------------------
def finalize_scattering(
    basis: ChannelBasis | ChannelBasisElectricSF,
    Y_propagated: LogDArray,
    Etot: EnergyInput,
    reduced_mass: float,
    Rmatch: float,
) -> ScatteringResult:
    r"""
    Transform, asymptotically match, and package one exact result.

    Formula:
        For a field-free BF basis,

        Y_asym = B.T Y_BF B,

        where B diagonalizes the centrifugal matrix and produces the asymptotic
        orbital angular momenta L.

        For an Electric-SF channel eta=(alpha,m,l,m_l), the propagated basis is
        already asymptotic:

        B = I,    L_eta = l_eta,    Y_asym = Y_propagated.

    Inputs:
        basis: ChannelBasis | ChannelBasisElectricSF - complete exact channel
            basis
        Y_propagated: LogDArray - final log derivatives in the propagated
            representation, shape
            (n_energy, n_channel, n_channel)
        Etot: EnergyInput - total energies with shape (n_energy,), or a one-column
            text file
        reduced_mass: float - collision reduced mass in atomic units
        Rmatch: float - asymptotic matching distance in atomic units

    Returns:
        result: ScatteringResult - propagated, asymptotic, and S matrices
    """
    energies = get_Etot(Etot)
    Y_array = cast(LogDArray, np.asarray(Y_propagated))
    if isinstance(basis, ChannelBasisElectricSF):
        asymptotic_transform = np.eye(basis.n_channel, dtype=np.float64)
        L = np.asarray([channel.l for channel in basis], dtype=np.float64)
        Y_asymptotic = Y_array
    else:
        asymptotic_transform, L = get_Bmat_BF_to_SF(basis)
        Y_asymptotic = transform_logD_BF_to_SF(Y_array, asymptotic_transform)
    Smat = get_Smat(Y_asymptotic, Rmatch, energies, reduced_mass, basis.E_int, L)
    return ScatteringResult(
        basis=basis,
        Etot=energies,
        Y_propagated=Y_array,
        asymptotic_transform=asymptotic_transform,
        L=L,
        Smat=Smat,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def finalize_reactive_scattering(
    basis: DelvesBasis,
    total_potential: TotalPES,
    propagation: DelvesPropagationResult,
    Etot: EnergyInput,
    *,
    energy_zero: float = 0.0,
) -> ReactiveScatteringResult:
    r"""
    Transform, asymptotically match, and package one Delves reactive result.

    Formula:
        At the physical matching boundary ``rho_f``, construct fixed Jacobi
        channels ``Q=(a,v,j,K)`` and their finite-rho Delves coefficients
        ``C_channel(rho_f)``. If the last propagated surface lies at ``rho_s``,

        T = C_surface(rho_s)^T P(rho_s,rho_f) C_channel(rho_f),

        Y_channel(E) = T^T Y_surface(E) T.

        Pack--Parker projection of open and closed Jacobi reference functions
        then gives the reactive scattering matrices ``S(E)``.

    Inputs:
        basis: DelvesBasis - primitive Delves basis used during propagation
        total_potential: TotalPES - scalar total three-body PES
        propagation: DelvesPropagationResult - final fixed-grid surface LogD
            and sector endpoints
        Etot: EnergyInput - total energies in Hartree
        energy_zero: float - native-PES energy subtracted by the Hamiltonian,
            in Hartree

    Returns:
        result: ReactiveScatteringResult - arrangement channels, propagated and
            asymptotic LogD matrices, S matrices, and sector endpoints
    """
    energies = get_Etot(Etot)
    channels = build_delves_asymptotic_basis(basis, total_potential, propagation.rho_final)
    Y_asymptotic = transform_logD_to_delves_channels(
        basis,
        propagation.surface_rho,
        propagation.surface_coefficients,
        propagation.Y_final,
        channels,
    )
    Smat = get_delves_Smat(Y_asymptotic, energies, basis, channels)
    return ReactiveScatteringResult(
        basis=channels,
        Etot=energies,
        Y_propagated=np.asarray(propagation.Y_final),
        Y_asymptotic=Y_asymptotic,
        Smat=Smat,
        rho_final=propagation.rho_final,
        surface_rho=propagation.surface_rho,
        radial_points=propagation.radial_points,
        energy_zero=float(energy_zero),
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
        Smat_asymptotic=Smat_asymptotic,
    )


# ----------------------------------------------------------------------------------------
