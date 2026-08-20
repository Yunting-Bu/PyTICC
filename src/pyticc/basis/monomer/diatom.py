from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.dvr import SineDVR, build_SineDVR
from pyticc.basis.podvr import build_RovibPODVR
from pyticc.basis.rovib import RovibBasis
from pyticc.system import MolInnerState, MonomerType


# Adiabatic diatom
# ----------------------------------------------------------------------------------------
def _validate_state_table(Eint: NDArray) -> None:
    """Validate a complete rovibrational energy table."""
    if Eint.ndim != 2:
        message = f"Eint must be a two-dimensional array indexed as Eint[v, j], but got ndim={Eint.ndim}"
        logger.error(message)
        raise ValueError(message)
    if 0 in Eint.shape:
        message = f"Eint must contain at least one vibrational and rotational level, but got shape {Eint.shape}"
        logger.error(message)
        raise ValueError(message)
    if not np.issubdtype(Eint.dtype, np.number):
        message = f"Eint must contain numeric energies, but got dtype={Eint.dtype}"
        logger.error(message)
        raise ValueError(message)


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DiatomSpec:
    """
    Lightweight diatomic state table used by channel internals.

    Members:
        Eint: NDArray - relative rovibrational energies indexed as Eint[v,j],
            shape (vmax + 1,jmax + 1)
    """

    type = MonomerType.DIATOM
    Eint: NDArray

    def __post_init__(self) -> None:
        _validate_state_table(self.Eint)

    @property
    def vmax(self) -> int:
        """Return the largest available vibrational quantum number."""
        return self.Eint.shape[0] - 1

    @property
    def jmax(self) -> int:
        """Return the largest available rotational quantum number."""
        return self.Eint.shape[1] - 1

    def mis_iter(self, E_cut: float):
        """Yield retained rovibrational states below an energy cutoff."""
        for v in range(self.vmax + 1):
            for j in range(self.jmax + 1):
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
@dataclass(frozen=True)
class DiatomBasis:
    """
    Complete adiabatic diatomic basis for channels and PES projection.

    Members:
        rovib: RovibBasis - contracted radial grids, rovibrational energies,
            and wavefunctions
        energy_zero: float - absolute energy subtracted from channel thresholds,
            in atomic units
    """

    rovib: RovibBasis
    energy_zero: float

    type = MonomerType.DIATOM

    def __post_init__(self) -> None:
        energies = np.asarray(self.rovib.E_vj)
        if not np.isfinite(self.energy_zero):
            message = f"energy_zero must be finite, but got {self.energy_zero}"
            logger.error(message)
            raise ValueError(message)
        _validate_state_table(energies)

    @property
    def vmax(self) -> int:
        """Return the largest available vibrational quantum number."""
        return self.rovib.E_vj.shape[0] - 1

    @property
    def jmax(self) -> int:
        """Return the largest available rotational quantum number."""
        return self.rovib.E_vj.shape[1] - 1

    @property
    def Eint(self) -> NDArray[np.float64]:
        """Return rovibrational energies relative to ``energy_zero``."""
        return np.asarray(self.rovib.E_vj - self.energy_zero, dtype=np.float64)

    def mis_iter(self, E_cut: float):
        """Yield retained rovibrational states below an energy cutoff."""
        for v in range(self.vmax + 1):
            for j in range(self.jmax + 1):
                energy = float(self.rovib.E_vj[v, j] - self.energy_zero)
                if np.isfinite(energy) and energy <= E_cut:
                    yield MolInnerState(j=j, v=v, Eint=energy)

    def energy(self, mis: MolInnerState, K: int) -> float:
        """Return one relative rovibrational energy."""
        if mis.v is None:
            message = "Diatomic inner state requires v"
            logger.error(message)
            raise ValueError(message)
        return float(self.rovib.E_vj[mis.v, mis.j] - self.energy_zero)

    def allows_K(self, mis: MolInnerState, K: int) -> bool:
        """Allow every system helicity admitted by angular momentum coupling."""
        return True


# ----------------------------------------------------------------------------------------
def build_DiatomBasis(
    dvr: SineDVR,
    *,
    n_podvr: int,
    vmax: int,
    jmax: int,
    mass: float,
    energy_zero: float | None = None,
) -> DiatomBasis:
    """Build an adiabatic diatomic monomer basis from a primitive DVR."""
    rovib = build_RovibPODVR(dvr, n_podvr, vmax, jmax, mass)
    zero = float(rovib.E_vj[0, 0]) if energy_zero is None else float(energy_zero)
    return DiatomBasis(rovib=rovib, energy_zero=zero)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def prepare_Diatom(
    potential: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None,
    *,
    r: tuple[float, float],
    n_dvr: int,
    n_podvr: int,
    vmax: int,
    jmax: int,
    mass: float,
    energy_zero: float | None = None,
) -> DiatomBasis:
    """
    Prepare an adiabatic diatomic monomer through the DVR and PODVR steps.

    Inputs:
        potential: Callable | None - monomer PES mapping bond-length grids with
            shape (n_dvr,) to energies with shape (n_dvr,); None raises an error
        r: tuple[float,float] - left and right sine-DVR boundaries
        n_dvr: int - number of primitive sine-DVR points
        n_podvr: int - number of contracted PODVR points
        vmax: int - highest retained vibrational quantum number
        jmax: int - highest retained rotational quantum number
        mass: float - diatomic reduced mass in atomic units
        energy_zero: float | None - absolute channel-energy zero; None uses
            E(v=0,j=0)

    Returns:
        monomer: DiatomBasis - prepared PODVR rovibrational monomer basis
    """
    if potential is None:
        message = "Diatomic monomer preparation requires a monomer potential"
        logger.error(message)
        raise ValueError(message)

    dvr = build_SineDVR(r[0], r[1], n_dvr, mass, potential)
    return build_DiatomBasis(
        dvr,
        n_podvr=n_podvr,
        vmax=vmax,
        jmax=jmax,
        mass=mass,
        energy_zero=energy_zero,
    )


# ----------------------------------------------------------------------------------------
