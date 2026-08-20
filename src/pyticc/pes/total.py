from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from loguru import logger
from numpy.typing import NDArray

TotalPotential: TypeAlias = Callable[[NDArray[np.float64]], NDArray[np.float64]]


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TotalPES:
    """
    Scalar adiabatic total potential-energy surface for three atoms.

    The PES receives all three physical bond lengths at once. It returns the
    total energy on one adiabatic electronic surface; it must not subtract a
    diatomic potential, add a separate monomer potential, or change the energy
    zero. Reactive-scattering energies and channel thresholds must use the same
    zero as this surface.

    Members:
        potential: TotalPotential - callable receiving physical bonds in bohr
            with shape ``(3,n_grid)`` and leading order ``(r_AB,r_BC,r_CA)``,
            returning total energies in Hartree with shape ``(n_grid,)``
        _close: Callable[[],None] | None - optional resource-release callback
    """

    potential: TotalPotential
    _close: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def __call__(self, bonds: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the total PES on physical three-bond geometries.

        Inputs:
            bonds: NDArray[np.float64] - physical bond lengths in bohr with
                shape ``(3,n_grid)`` and order ``(r_AB,r_BC,r_CA)``

        Returns:
            values: NDArray[np.float64] - total energies in Hartree, shape
                ``(n_grid,)``
        """
        coordinates = np.asarray(bonds)
        if np.iscomplexobj(coordinates):
            message = "Total PES bond lengths must be real"
            logger.error(message)
            raise ValueError(message)
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[0] != 3:
            message = f"Total PES bonds must have shape (3,n_grid), but got {coordinates.shape}"
            logger.error(message)
            raise ValueError(message)
        if not np.all(np.isfinite(coordinates)) or np.any(coordinates <= 0.0):
            message = "Total PES bonds must contain finite positive lengths"
            logger.error(message)
            raise ValueError(message)

        values = np.asarray(self.potential(np.asfortranarray(coordinates)))
        if np.iscomplexobj(values):
            message = "Total PES must return real energies"
            logger.error(message)
            raise ValueError(message)
        values = np.asarray(values, dtype=np.float64)
        expected_shape = (coordinates.shape[1],)
        if values.shape != expected_shape:
            message = f"Total PES returned shape {values.shape}, but expected {expected_shape}"
            logger.error(message)
            raise ValueError(message)
        if not np.all(np.isfinite(values)):
            message = "Total PES returned non-finite energies"
            logger.error(message)
            raise ValueError(message)
        return values

    def close(self) -> None:
        """Release resources owned by the total PES; repeated calls must be safe."""
        if self._close is not None:
            self._close()


# ----------------------------------------------------------------------------------------
