from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.system import MolInnerState, MonomerType


# ----------------------------------------------------------------------------------------
def arrange_diatom_levels(levels: Iterable[tuple[int, int, float]], vmax: int, jmax: int) -> NDArray[np.float64]:
    """
    Arrange labeled diatomic energies into an Eint[v, j] array.

    Inputs:
        levels: Iterable[tuple[int, int, float]] - labeled levels given as (v, j, energy)
        vmax: int - maximum vibrational quantum number
        jmax: int - maximum rotational quantum number

    Returns:
        Eint: NDArray[np.float64] - rovibrational energies with shape (vmax + 1, jmax + 1)
    """
    if vmax < 0 or jmax < 0:
        message = f"vmax and jmax must be non-negative, but got vmax={vmax}, jmax={jmax}"
        logger.error(message)
        raise ValueError(message)

    Eint = np.full((vmax + 1, jmax + 1), np.inf, dtype=np.float64)
    assigned = np.zeros(Eint.shape, dtype=bool)

    for v, j, energy in levels:
        if not 0 <= v <= vmax or not 0 <= j <= jmax:
            message = f"Diatomic level (v={v}, j={j}) is outside the requested range"
            logger.error(message)
            raise ValueError(message)
        if assigned[v, j]:
            message = f"Duplicate diatomic level (v={v}, j={j})"
            logger.error(message)
            raise ValueError(message)
        if not np.isfinite(energy):
            message = f"Diatomic level (v={v}, j={j}) must have finite energy"
            logger.error(message)
            raise ValueError(message)

        Eint[v, j] = energy
        assigned[v, j] = True

    return Eint


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def set_j_parity(jpar: int) -> tuple[int, int]:
    """Return the first allowed j and increment for a rotational parity."""
    try:
        return {-1: (1, 2), 0: (0, 1), 1: (0, 2)}[jpar]
    except KeyError as error:
        message = f"jpar must be -1, 0, or 1, but got jpar={jpar}"
        logger.error(message)
        raise ValueError(message) from error


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
class AtomSpec:
    type = MonomerType.ATOM
    jpar: int = 0

    def mis_iter(self, E_cut: float):
        yield MolInnerState(j=0, Eint=0.0)

    def energy(self, mis: MolInnerState, K: int) -> float:
        return 0.0


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DiatomSpec:
    type = MonomerType.DIATOM
    Eint: NDArray
    vmax: int
    jmax: int
    vmin: int = 0
    jpar: int = 0

    def __post_init__(self) -> None:
        if self.vmax < 0 or self.jmax < 0:
            message = f"vmax and jmax must be non-negative, but got vmax={self.vmax}, jmax={self.jmax}"
            logger.error(message)
            raise ValueError(message)
        if not 0 <= self.vmin <= self.vmax:
            message = f"vmin must satisfy 0 <= vmin <= vmax, but got vmin={self.vmin}, vmax={self.vmax}"
            logger.error(message)
            raise ValueError(message)
        if self.Eint.ndim != 2:
            message = f"Eint must be a two-dimensional array indexed as Eint[v, j], but got ndim={self.Eint.ndim}"
            logger.error(message)
            raise ValueError(message)
        if self.Eint.shape[0] <= self.vmax or self.Eint.shape[1] <= self.jmax:
            message = f"Eint shape {self.Eint.shape} does not cover vmax={self.vmax} and jmax={self.jmax}"
            logger.error(message)
            raise ValueError(message)
        if not np.issubdtype(self.Eint.dtype, np.number):
            message = f"Eint must contain numeric energies, but got dtype={self.Eint.dtype}"
            logger.error(message)
            raise ValueError(message)
        set_j_parity(self.jpar)

    def mis_iter(self, E_cut: float):
        jmin, jinc = set_j_parity(self.jpar)
        for v in range(self.vmin, self.vmax + 1):
            for j in range(jmin, self.jmax + 1, jinc):
                energy = float(self.Eint[v, j])
                if np.isfinite(energy) and energy <= E_cut:
                    yield MolInnerState(j=j, v=v, Eint=energy)

    def energy(self, mis: MolInnerState, K: int) -> float:
        if mis.v is None:
            message = "Diatomic inner state requires v"
            logger.error(message)
            raise ValueError(message)
        return float(self.Eint[mis.v, mis.j])


# ----------------------------------------------------------------------------------------
