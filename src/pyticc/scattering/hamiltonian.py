from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc._typing import JaxDevice
from pyticc.basis.channel import ChannelBasis, ChannelBasisElectricSF
from pyticc.fine_structure.channel import FSChannelBasis
from pyticc.fine_structure.diatom_diatom import FSDiatomDiatomBasis
from pyticc.matrix.centrifugal import get_Umat_BF, get_Umat_ElectricSF, get_Umat_FS_BF, get_Umat_FS_DiatomDiatom_BF
from pyticc.system import Approx

HamiltonianArray = NDArray[np.float64] | NDArray[np.complex128]
Interaction = Callable[[float | NDArray[np.float64]], HamiltonianArray]
BlockInteraction = Callable[[NDArray[np.float64], tuple[tuple[int, ...], ...]], tuple[HamiltonianArray, ...]]
DeviceBlockInteraction = Callable[[NDArray[np.float64], tuple[tuple[int, ...], ...], JaxDevice], tuple[jax.Array, ...]]
ScatteringBasis = ChannelBasis | ChannelBasisElectricSF | FSChannelBasis | FSDiatomDiatomBasis


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ScattHamiltonian:
    r"""
    Scattering Hamiltonian projected onto one channel basis.

    Formula:
        For either a field-free BF basis or a field SF basis,

        H(R) = diag(E_int) + U/(2 mu_R R^2) + V(R),

        W(R;E_tot) = 2 mu_R [H(R)-E_tot I].

        The channel basis owns conserved block quantum numbers, while this
        object retains only the data required after Hamiltonian construction.

    Members:
        basis: ScatteringBasis - complete channel basis
        reduced_mass: float - collision reduced mass in atomic units
        approx: Approx - exact CC, CS, or NNCC approximation
        K_delta: int - neighboring-K range used by NNCC
        interaction: Interaction - callback returning V(R) for one point or a
            radial batch
        block_interaction: BlockInteraction | None - optimized interaction
            provider for field-free CS and NNCC channel blocks
        device_block_interaction: DeviceBlockInteraction | None - optional
            interaction contraction performed directly on a JAX device
        potential_grid_size: int - internal PES grid points per radial point
    """

    basis: ScatteringBasis
    reduced_mass: float
    interaction: Interaction
    approx: Approx = Approx.EXACT
    K_delta: int = 1
    block_interaction: BlockInteraction | None = None
    device_block_interaction: DeviceBlockInteraction | None = None
    potential_grid_size: int = 0

    def __post_init__(self) -> None:
        if self.K_delta < 1:
            message = f"K_delta must be positive, but got {self.K_delta}"
            logger.error(message)
            raise ValueError(message)

    @property
    def E_int(self) -> NDArray[np.float64]:
        """Return channel internal energies with shape (n_channel,)."""
        return self.basis.E_int

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
        if isinstance(self.basis, FSChannelBasis):
            return get_Umat_FS_BF(self.basis, channel_indices)
        if isinstance(self.basis, FSDiatomDiatomBasis):
            return get_Umat_FS_DiatomDiatom_BF(self.basis, channel_indices)
        return get_Umat_BF(self.basis, channel_indices)

    @property
    def U(self) -> NDArray[np.float64]:
        """Return the complete dimensionless centrifugal matrix."""
        return self.centrifugal()

    def V(self, R: float | NDArray[np.float64]) -> HamiltonianArray:
        """Evaluate the channel interaction matrix at one or more radial points."""
        return np.asarray(self.interaction(R))

    def V_blocks(
        self,
        radial_points: NDArray[np.float64],
        channel_blocks: tuple[tuple[int, ...], ...],
        device: JaxDevice | None = None,
    ) -> tuple[NDArray[np.float64] | jax.Array, ...]:
        """Evaluate interaction matrices for one or more channel blocks."""
        if device is not None and device.platform == "gpu" and self.device_block_interaction is not None:
            return self.device_block_interaction(radial_points, channel_blocks, device)
        if self.block_interaction is not None:
            return self.block_interaction(radial_points, channel_blocks)

        matrices = self.V(radial_points)
        return tuple(matrices[:, indices, :][:, :, indices] for indices in channel_blocks)

    def H(self, R: float, channel_indices: Sequence[int] | None = None) -> HamiltonianArray:
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

    def W(self, R: float, Etot: float, channel_indices: Sequence[int] | None = None) -> HamiltonianArray:
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
