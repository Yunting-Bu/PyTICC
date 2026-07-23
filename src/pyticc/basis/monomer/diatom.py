from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.dvr import SineDVR
from pyticc.basis.podvr import RovibPODVR, build_RovibPODVR
from pyticc.system import MolInnerState, MonomerType, set_j_parity


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DiatomSpec:
    """Lightweight diatomic state table used by channel internals."""

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
        """Yield retained rovibrational states below an energy cutoff."""
        jmin, jinc = set_j_parity(self.jpar)
        for v in range(self.vmin, self.vmax + 1):
            for j in range(jmin, self.jmax + 1, jinc):
                energy = float(self.Eint[v, j])
                if np.isfinite(energy) and energy <= E_cut:
                    yield MolInnerState(j=j, v=v, Eint=energy)

    def energy(self, mis: MolInnerState, K: int) -> float:
        """Look up one labeled rovibrational energy."""
        if mis.v is None:
            message = "Diatomic inner state requires v"
            logger.error(message)
            raise ValueError(message)
        return float(self.Eint[mis.v, mis.j])

    def allows_K(self, mis: MolInnerState, K: int) -> bool:
        """Allow every system helicity admitted by angular momentum coupling."""
        return True


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DiatomBasis:
    """Complete adiabatic diatomic basis for channels and PES projection."""

    rovib: RovibPODVR
    energy_zero: float
    vmax: int
    jmax: int
    vmin: int = 0
    jpar: int = 0
    _states: DiatomSpec = field(init=False, repr=False, compare=False)

    type = MonomerType.DIATOM

    def __post_init__(self) -> None:
        grids = np.asarray(self.rovib.grids)
        energies = np.asarray(self.rovib.E_vj)
        wavefunctions = np.asarray(self.rovib.WF_vj)
        if grids.ndim != 1 or energies.ndim != 2 or wavefunctions.shape != (grids.size, *energies.shape):
            message = (
                "RovibPODVR shapes must satisfy grids=(n_grid,), E_vj=(n_v,n_j), "
                f"and WF_vj=(n_grid,n_v,n_j), but got {grids.shape}, {energies.shape}, and {wavefunctions.shape}"
            )
            logger.error(message)
            raise ValueError(message)
        if not np.isfinite(self.energy_zero):
            message = f"energy_zero must be finite, but got {self.energy_zero}"
            logger.error(message)
            raise ValueError(message)

        states = DiatomSpec(
            Eint=np.asarray(energies - self.energy_zero, dtype=np.float64),
            vmax=self.vmax,
            jmax=self.jmax,
            vmin=self.vmin,
            jpar=self.jpar,
        )
        object.__setattr__(self, "_states", states)

    @property
    def Eint(self) -> NDArray[np.float64]:
        """Return rovibrational energies relative to ``energy_zero``."""
        return self._states.Eint

    def mis_iter(self, E_cut: float):
        """Yield retained rovibrational states below an energy cutoff."""
        yield from self._states.mis_iter(E_cut)

    def energy(self, mis: MolInnerState, K: int) -> float:
        """Return one relative rovibrational energy."""
        return self._states.energy(mis, K)

    def allows_K(self, mis: MolInnerState, K: int) -> bool:
        """Allow every system helicity admitted by angular momentum coupling."""
        return self._states.allows_K(mis, K)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_DiatomBasis(
    dvr: SineDVR,
    *,
    n_podvr: int,
    vmax: int,
    jmax: int,
    mass: float,
    vmin: int = 0,
    jpar: int = 0,
    energy_zero: float | None = None,
) -> DiatomBasis:
    """Build an adiabatic diatomic monomer basis from a primitive DVR."""
    rovib = build_RovibPODVR(dvr, n_podvr, vmax, jmax, mass)
    zero = float(rovib.E_vj[0, 0]) if energy_zero is None else float(energy_zero)
    return DiatomBasis(rovib=rovib, energy_zero=zero, vmax=vmax, jmax=jmax, vmin=vmin, jpar=jpar)


# ----------------------------------------------------------------------------------------
