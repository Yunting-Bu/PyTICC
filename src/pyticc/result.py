from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyticc.basis.channel import ChannelBasis, ChannelBasisElectricSF, OpenClosedChannels
from pyticc.basis.kblock import KBlock
from pyticc.match.delves import DelvesAsymptoticBasis
from pyticc.system import Approx

LogDArray = NDArray[np.float64] | NDArray[np.complex128]
ScatteringBasis = ChannelBasis | ChannelBasisElectricSF


def _transform_logD(Ymat: LogDArray, transform: NDArray[np.float64]) -> LogDArray:
    """Transform a log-derivative batch without retaining a second matrix copy."""
    return np.einsum("ia,...ij,jb->...ab", transform, Ymat, transform, optimize=True)


@dataclass(frozen=True, slots=True)
class Timing:
    """Elapsed solver wall-clock and process CPU times in seconds."""

    wall_seconds: float
    cpu_seconds: float

    def __str__(self) -> str:
        return f"wall={self.wall_seconds:.3f} s, CPU={self.cpu_seconds:.3f} s"


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ScatteringResult:
    """
    Exact scattering result in one conserved-quantity block.

    Members:
        basis: ChannelBasis | ChannelBasisElectricSF - propagated channel basis
        Etot: NDArray[np.float64] - total energies in atomic units, shape
            (n_energy,)
        Y_propagated: NDArray - final log-derivative matrices in the propagated
            channel representation, shape
            (n_energy, n_channel, n_channel)
        asymptotic_transform: NDArray[np.float64] - propagated-to-asymptotic
            orthogonal transformation, shape (n_channel,n_channel)
        L: NDArray[np.float64] - asymptotic orbital angular momenta, shape
            (n_channel,)
        Smat: tuple[NDArray[np.complex128], ...] - one matrix per energy; element i
            has shape (n_open[i], n_open[i])
        timing: Timing | None - elapsed solver time
    """

    basis: ScatteringBasis
    Etot: NDArray[np.float64]
    Y_propagated: LogDArray
    asymptotic_transform: NDArray[np.float64]
    L: NDArray[np.float64]
    Smat: tuple[NDArray[np.complex128], ...]
    timing: Timing | None = None

    @property
    def open_closed(self) -> OpenClosedChannels:
        """Classify channels at the stored total energies."""
        return self.basis.open_closed(self.Etot)

    @property
    def Y_asymptotic(self) -> LogDArray:
        """Return log derivatives transformed to the asymptotic representation."""
        return _transform_logD(self.Y_propagated, self.asymptotic_transform)

    @property
    def open_channel_indices(self) -> tuple[NDArray[np.int64], ...]:
        """Return one index array with shape (n_open[i],) for each energy."""
        return tuple(np.asarray(np.flatnonzero(mask), dtype=np.int64) for mask in self.open_closed.open_mask)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ReactiveScatteringResult:
    """
    Exact Delves reactive-scattering result in one conserved-quantity block.

    The propagated adiabatic surface dimension may differ from the final
    arrangement-channel dimension. ``Y_propagated`` and ``Y_asymptotic`` are
    therefore stored explicitly instead of assuming a square orthogonal
    transformation between equal-sized bases.

    Members:
        basis: DelvesAsymptoticBasis - final channels labeled ``(a,v,j,K)``
        Etot: NDArray[np.float64] - total energies in Hartree, shape
            ``(n_energy,)``
        Y_propagated: LogDArray - final LogD matrices in the last adiabatic
            surface basis, shape ``(n_energy,n_surface,n_surface)``
        Y_asymptotic: LogDArray - LogD matrices transformed to arrangement
            channels, shape ``(n_energy,n_channel,n_channel)``
        Smat: tuple[NDArray[np.complex128], ...] - one open-channel reactive
            scattering matrix per total energy
        rho_final: float - physical matching hyperradius in bohr
        surface_rho: float - hyperradius of the last adiabatic surface basis in
            bohr
        radial_points: NDArray[np.float64] - fixed sector endpoints generated
            from ``boundaries`` and ``half_steps``, shape ``(n_sector+1,)``
        timing: Timing | None - elapsed solver time
        energy_zero: float - native-PES energy subtracted by the Hamiltonian,
            in Hartree; add this value to stored energies to recover the native
            PES convention
    """

    basis: DelvesAsymptoticBasis
    Etot: NDArray[np.float64]
    Y_propagated: LogDArray
    Y_asymptotic: LogDArray
    Smat: tuple[NDArray[np.complex128], ...]
    rho_final: float
    surface_rho: float
    radial_points: NDArray[np.float64]
    timing: Timing | None = None
    energy_zero: float = 0.0

    @property
    def open_channel_indices(self) -> tuple[NDArray[np.int64], ...]:
        """Return asymptotic channel positions open at each total energy."""
        return tuple(np.asarray(np.flatnonzero(self.basis.energies < energy), dtype=np.int64) for energy in self.Etot)


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
        Smat_asymptotic: tuple[NDArray[np.complex128], ...] - one matched block
            matrix per energy; element i has shape
            (n_open_block[i], n_open_block[i])
    """

    block: KBlock
    open_channel_indices: tuple[NDArray[np.int64], ...]
    Y_BF: LogDArray
    Bmat: NDArray[np.float64]
    L: NDArray[np.float64]
    Smat_asymptotic: tuple[NDArray[np.complex128], ...]

    @property
    def Y_asymptotic(self) -> LogDArray:
        """Return block log derivatives transformed to the asymptotic basis."""
        return _transform_logD(self.Y_BF, self.Bmat)

    @property
    def Smat_BF(self) -> tuple[NDArray[np.complex128], ...]:
        r"""
        Return open-channel scattering matrices in the BF helicity basis.

        Formula:
            S_BF(E) = B_open(E) S_asym(E) B_open(E).T,

            where B_open is the open-channel submatrix of the orthogonal BF-to-SF
            transformation B. Rows and columns of S_BF follow the corresponding
            entry of ``open_channel_indices`` and are therefore labeled by the
            original complete-basis K channels.

        Returns:
            matrices: tuple[NDArray[np.complex128],...] - one K-labeled open-channel
                S matrix per energy, with shape (n_open_block[i],n_open_block[i])
        """
        block_positions = {global_index: local_index for local_index, global_index in enumerate(self.block.channel_indices)}
        matrices: list[NDArray[np.complex128]] = []
        for open_indices, Smat in zip(self.open_channel_indices, self.Smat_asymptotic, strict=True):
            local_open = np.asarray([block_positions[int(index)] for index in open_indices], dtype=np.int64)
            B_open = self.Bmat[np.ix_(local_open, local_open)]
            matrices.append(np.asarray(B_open @ Smat @ B_open.T, dtype=np.complex128))
        return tuple(matrices)


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
        approx: Approx - CS or NNCC approximation
        blocks: tuple[KBlockResult, ...] - independently propagated and matched blocks
        timing: Timing | None - elapsed solver time
    """

    basis: ChannelBasis
    Etot: NDArray[np.float64]
    approx: Approx
    blocks: tuple[KBlockResult, ...]
    timing: Timing | None = None

    @property
    def open_closed(self) -> OpenClosedChannels:
        """Classify complete-basis channels at the stored total energies."""
        return self.basis.open_closed(self.Etot)


# ----------------------------------------------------------------------------------------
