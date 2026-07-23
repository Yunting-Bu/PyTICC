from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.channel import ChannelBasis
from pyticc.matrix.centrifugal import get_Umat_BF
from pyticc.system import ScattSystem

Interaction = Callable[[float | NDArray[np.float64]], NDArray[np.float64]]
BlockInteraction = Callable[[NDArray[np.float64], tuple[tuple[int, ...], ...]], tuple[NDArray[np.float64], ...]]


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ScattHamiltonian:
    """Scattering Hamiltonian projected onto one body-fixed channel basis.

    Members:
        system: ScattSystem - monomers, Jtot, system parity, and approximation
        basis: ChannelBasis - complete body-fixed channel basis
        interaction: Interaction - callback returning V(R) for one point or a batch
        block_interaction: BlockInteraction | None - optimized interaction provider
            for CS and NNCC channel blocks
        batched: bool - whether interaction accepts a radial array
        potential_grid_size: int - internal PES grid points per radial point
    """

    system: ScattSystem
    basis: ChannelBasis
    interaction: Interaction
    block_interaction: BlockInteraction | None = None
    batched: bool = True
    potential_grid_size: int = 0

    def __post_init__(self) -> None:
        if self.system.Jtot is None or self.system.system_parity is None:
            message = "ScattHamiltonian requires ScattSystem.Jtot and ScattSystem.system_parity"
            logger.error(message)
            raise ValueError(message)
        if self.system.reduced_mass is None:
            message = "ScattSystem.reduced_mass is required to build a ScattHamiltonian"
            logger.error(message)
            raise ValueError(message)
        if self.system.potential is None:
            message = "ScattSystem.potential is required to build a ScattHamiltonian"
            logger.error(message)
            raise ValueError(message)
        if self.basis.n_channel == 0:
            message = "ScattHamiltonian requires at least one channel"
            logger.error(message)
            raise ValueError(message)
        if self.potential_grid_size < 0:
            message = f"potential_grid_size must be non-negative, but got {self.potential_grid_size}"
            logger.error(message)
            raise ValueError(message)

        for channel in self.basis:
            if channel.Jtot != self.system.Jtot or channel.system_parity != self.system.system_parity:
                message = "ChannelBasis Jtot/system_parity does not match ScattSystem"
                logger.error(message)
                raise ValueError(message)

    @property
    def E_int(self) -> NDArray[np.float64]:
        """Return channel internal energies with shape ``(n_channel,)``."""
        return self.basis.E_int

    @property
    def reduced_mass(self) -> float:
        """Return the collision reduced mass in atomic units."""
        mass = self.system.reduced_mass
        if mass is None:
            message = "ScattSystem.reduced_mass is not set"
            logger.error(message)
            raise ValueError(message)
        return mass

    @property
    def U(self) -> NDArray[np.float64]:
        """Return the dimensionless body-fixed centrifugal matrix."""
        return get_Umat_BF(self.basis)

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
        expected_shape = (radial_points.size, self.basis.n_channel, self.basis.n_channel)
        if matrices.shape != expected_shape:
            message = f"interaction returned shape {matrices.shape}, but expected {expected_shape}"
            logger.error(message)
            raise ValueError(message)
        return tuple(matrices[:, indices, :][:, :, indices] for indices in channel_blocks)

    def H(self, R: float, channel_indices: Sequence[int] | None = None) -> NDArray[np.float64]:
        r"""Return ``E_int + U/(2*mu*R**2) + V(R)`` in atomic units."""
        if R <= 0.0:
            message = f"R must be positive, but got R={R}"
            logger.error(message)
            raise ValueError(message)

        indices = tuple(range(self.basis.n_channel)) if channel_indices is None else tuple(channel_indices)
        positions = np.asarray(indices, dtype=np.int64)
        interaction = self.V(R)
        expected_shape = (self.basis.n_channel, self.basis.n_channel)
        if interaction.shape != expected_shape:
            message = f"interaction returned shape {interaction.shape}, but expected {expected_shape}"
            logger.error(message)
            raise ValueError(message)

        matrix = interaction[np.ix_(positions, positions)].copy()
        matrix += get_Umat_BF(self.basis, indices) / (2.0 * self.reduced_mass * R**2)
        diagonal = np.diag_indices(len(indices))
        matrix[diagonal] += self.E_int[positions]
        return matrix

    def W(self, R: float, Etot: float, channel_indices: Sequence[int] | None = None) -> NDArray[np.float64]:
        r"""Return ``2*mu*(H(R) - Etot*I)`` for the radial equations."""
        matrix = 2.0 * self.reduced_mass * self.H(R, channel_indices)
        diagonal = np.diag_indices(matrix.shape[0])
        matrix[diagonal] -= 2.0 * self.reduced_mass * Etot
        return matrix


# ----------------------------------------------------------------------------------------
