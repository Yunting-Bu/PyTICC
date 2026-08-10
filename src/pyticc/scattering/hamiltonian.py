from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from pyticc.basis.channel import ChannelBasis, ChannelBasisElectricSF
from pyticc.matrix.centrifugal import get_Umat_BF, get_Umat_ElectricSF
from pyticc.system import ScattSystem

Interaction = Callable[[float | NDArray[np.float64]], NDArray[np.float64]]
BlockInteraction = Callable[[NDArray[np.float64], tuple[tuple[int, ...], ...]], tuple[NDArray[np.float64], ...]]
ScatteringBasis = ChannelBasis | ChannelBasisElectricSF


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ScattHamiltonian:
    r"""
    Scattering Hamiltonian projected onto one channel basis.

    Formula:
        For either a field-free BF basis or a field SF basis,

        H(R) = diag(E_int) + U/(2 mu_R R^2) + V(R),

        W(R;E_tot) = 2 mu_R [H(R)-E_tot I].

        The conserved quantities and interaction PES belong to ScattSystem.
        The channel basis determines the representation-specific centrifugal
        matrix U.

    Members:
        system: ScattSystem - physical system, conserved quantities, PES, and
            collision reduced mass
        basis: ChannelBasis | ChannelBasisElectricSF - complete channel basis
        interaction: Interaction - callback returning V(R) for one point or a
            radial batch
        block_interaction: BlockInteraction | None - optimized interaction
            provider for field-free CS and NNCC channel blocks
        batched: bool - whether interaction accepts a radial array
        potential_grid_size: int - internal PES grid points per radial point
    """

    system: ScattSystem
    basis: ScatteringBasis
    interaction: Interaction
    block_interaction: BlockInteraction | None = None
    batched: bool = True
    potential_grid_size: int = 0

    @property
    def E_int(self) -> NDArray[np.float64]:
        """Return channel internal energies with shape (n_channel,)."""
        return self.basis.E_int

    @property
    def reduced_mass(self) -> float:
        """Return the collision reduced mass in atomic units."""
        return cast(float, self.system.reduced_mass)

    def centrifugal(self, channel_indices: Sequence[int] | None = None) -> NDArray[np.float64]:
        """
        Return the representation-specific dimensionless centrifugal matrix.

        Inputs:
            channel_indices: Sequence[int] | None - optional complete-basis
                positions in the requested order

        Returns:
            Umat: NDArray[np.float64] - dimensionless centrifugal matrix,
                shape (n_selected,n_selected)
        """
        if isinstance(self.basis, ChannelBasisElectricSF):
            return get_Umat_ElectricSF(self.basis, channel_indices)
        return get_Umat_BF(self.basis, channel_indices)

    @property
    def U(self) -> NDArray[np.float64]:
        """Return the complete dimensionless centrifugal matrix."""
        return self.centrifugal()

    def V(self, R: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the channel interaction matrix at one or more radial points."""
        return np.asarray(self.interaction(R), dtype=np.float64)

    def V_blocks(
        self,
        radial_points: NDArray[np.float64],
        channel_blocks: tuple[tuple[int, ...], ...],
    ) -> tuple[NDArray[np.float64], ...]:
        """Evaluate interaction matrices for one or more channel blocks."""
        if self.block_interaction is not None:
            return self.block_interaction(radial_points, channel_blocks)

        matrices = self.V(radial_points)
        return tuple(matrices[:, indices, :][:, :, indices] for indices in channel_blocks)

    def H(self, R: float, channel_indices: Sequence[int] | None = None) -> NDArray[np.float64]:
        r"""
        Evaluate the channel Hamiltonian.

        Formula:
            H(R) = diag(E_int) + U/(2 mu_R R^2) + V(R).

        Inputs:
            R: float - intermolecular separation in atomic units
            channel_indices: Sequence[int] | None - optional complete-basis
                positions in the requested order

        Returns:
            Hmat: NDArray[np.float64] - channel Hamiltonian in atomic units,
                shape (n_selected,n_selected)
        """
        indices = tuple(range(self.basis.n_channel)) if channel_indices is None else tuple(channel_indices)
        positions = np.asarray(indices, dtype=np.int64)
        matrix = self.V(R)[np.ix_(positions, positions)].copy()
        matrix += self.centrifugal(indices) / (2.0 * self.reduced_mass * R**2)
        diagonal = np.diag_indices(len(indices))
        matrix[diagonal] += self.E_int[positions]
        return matrix

    def W(self, R: float, Etot: float, channel_indices: Sequence[int] | None = None) -> NDArray[np.float64]:
        r"""
        Evaluate the radial coupled-equation matrix.

        Formula:
            W(R;E_tot) = 2 mu_R [H(R)-E_tot I].

        Inputs:
            R: float - intermolecular separation in atomic units
            Etot: float - total energy in atomic units
            channel_indices: Sequence[int] | None - optional complete-basis
                positions in the requested order

        Returns:
            Wmat: NDArray[np.float64] - radial equation matrix in inverse bohr
                squared, shape (n_selected,n_selected)
        """
        matrix = 2.0 * self.reduced_mass * self.H(R, channel_indices)
        diagonal = np.diag_indices(matrix.shape[0])
        matrix[diagonal] -= 2.0 * self.reduced_mass * Etot
        return matrix


# ----------------------------------------------------------------------------------------
