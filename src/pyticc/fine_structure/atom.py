from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.constants import EnergyUnit, energy_to_au


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FSAtomBasis:
    """
    Experimental fine-structure monomer basis of one open-shell atom.

    Members:
        two_L: int - twice the orbital angular momentum L, necessarily even
        two_S: int - twice the electronic spin S
        atom_parity: int - intrinsic spatial parity epsilon=(-1)^(sum_a l_a)
        two_j_levels: tuple[int,...] - retained twice-j fine-structure levels in
            input order
        level_energies: NDArray[np.float64] - experimental level energies in
            Hartree, aligned with ``two_j_levels``
        energy_zero: float - threshold zero subtracted from level energies,
            Hartree
    """

    two_L: int
    two_S: int
    atom_parity: int
    two_j_levels: tuple[int, ...]
    level_energies: NDArray[np.float64]
    energy_zero: float

    @property
    def thresholds(self) -> NDArray[np.float64]:
        """Return level energies relative to ``energy_zero`` in Hartree."""
        return self.level_energies - self.energy_zero


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_fs_atom_basis(
    two_L: int,
    two_S: int,
    atom_parity: int,
    two_j_levels: Sequence[int],
    level_energies: float | Sequence[float],
    *,
    unit: EnergyUnit = "cm-1",
    energy_zero: float | None = None,
) -> FSAtomBasis:
    r"""
    Build the experimental fine-structure monomer basis of one open-shell atom.

    Formula:
        The atomic fine-structure Hamiltonian is diagonal in j with experimental
        level energies,

        h = sum_j E_j^exp |j><j|,

        and each body-fixed level expands in orbital and spin projections as

        |j kappa> = sum_{lambda,mu} <L lambda S mu|j kappa> |L lambda>|S mu>,

        with kappa = lambda + mu. The atomic spatial parity is

        epsilon = (-1)^(sum_a l_a),

        where the sum runs over all atomic electrons. The atom is assumed to
        remain in one term (fixed L and S) during the collision.

    Inputs:
        two_L: int - twice the atomic orbital angular momentum L, which must be
            an integer
        two_S: int - twice the electronic spin S
        atom_parity: int - intrinsic spatial parity, -1 or 1
        two_j_levels: Sequence[int] - retained twice-j fine-structure levels
        level_energies: float | Sequence[float] - experimental level energies; a
            scalar makes all retained levels degenerate
        unit: EnergyUnit - unit of ``level_energies`` and ``energy_zero``;
            converted to Hartree internally
        energy_zero: float | None - threshold zero in ``unit``; None uses the
            lowest retained level

    Returns:
        basis: FSAtomBasis - validated atomic fine-structure monomer basis
    """
    if isinstance(two_L, bool) or not isinstance(two_L, int) or two_L < 0 or two_L % 2:
        message = f"two_L must be a nonnegative even integer (L is integral), but got {two_L}"
        logger.error(message)
        raise ValueError(message)
    if isinstance(two_S, bool) or not isinstance(two_S, int) or two_S < 0:
        message = f"two_S must be a nonnegative integer, but got {two_S}"
        logger.error(message)
        raise ValueError(message)
    if atom_parity not in (-1, 1):
        message = f"atom_parity must be -1 or 1, but got {atom_parity}"
        logger.error(message)
        raise ValueError(message)
    levels = tuple(two_j_levels)
    if not levels or any(isinstance(level, bool) or not isinstance(level, int) or level < 0 for level in levels):
        message = f"two_j_levels must contain nonnegative integers, but got {two_j_levels}"
        logger.error(message)
        raise ValueError(message)
    if len(set(levels)) != len(levels):
        message = f"two_j_levels must be unique for one term, but got {two_j_levels}"
        logger.error(message)
        raise ValueError(message)
    for level in levels:
        if (level - two_L) % 2 != two_S % 2 or not abs(two_L - two_S) <= level <= two_L + two_S:
            message = f"j={level / 2} is not a fine-structure level of L={two_L / 2}, S={two_S / 2}; require |L-S| <= j <= L+S with j-L integer"
            logger.error(message)
            raise ValueError(message)
    values = np.asarray(level_energies, dtype=np.float64)
    if values.ndim == 0:
        energies_input = np.full(len(levels), float(values))
    elif values.ndim == 1 and values.size == len(levels):
        energies_input = values
    else:
        message = f"Expected {len(levels)} level energies, but got shape {np.shape(level_energies)}"
        logger.error(message)
        raise ValueError(message)
    if not np.all(np.isfinite(energies_input)):
        message = f"Level energies must be finite, but got {level_energies}"
        logger.error(message)
        raise ValueError(message)
    energies = np.asarray([energy_to_au(float(value), unit) for value in energies_input], dtype=np.float64)
    zero = float(np.min(energies)) if energy_zero is None else energy_to_au(float(energy_zero), unit)
    return FSAtomBasis(two_L, two_S, atom_parity, levels, energies, zero)


# ----------------------------------------------------------------------------------------
