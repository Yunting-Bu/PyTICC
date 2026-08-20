from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from loguru import logger
from scipy.optimize import minimize_scalar

from pyticc.pes.total import TotalPES


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DelvesMonomer:
    """
    Physical monomer information required by a Delves reactive system.

    The ABC asymptotic eigenstates, automatic sine basis, and reaction channels
    are deliberately absent. They are constructed later by ``build_ScattSystem``
    from ``ChannelSpec.E_Y_cut`` and the angular truncations.

    Members:
        mass: tuple[float,float,float] - atomic masses ``(A,B,C)`` in atomic units
        energy_zero: float - native total-PES energy subtracted from every
            threshold, total energy, and Hamiltonian value, in Hartree
    """

    mass: tuple[float, float, float]
    energy_zero: float = 0.0

    def __post_init__(self) -> None:
        masses = tuple(float(value) for value in self.mass)
        if len(masses) != 3 or any(not np.isfinite(value) or value <= 0.0 for value in masses):
            message = f"Delves mass must contain three finite positive values, but got {self.mass!r}"
            logger.error(message)
            raise ValueError(message)
        if not np.isfinite(self.energy_zero):
            message = f"Delves energy_zero must be finite, but got {self.energy_zero}"
            logger.error(message)
            raise ValueError(message)
        object.__setattr__(self, "mass", masses)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _minimum_asymptotic_energy(
    total_potential: TotalPES,
    mass: tuple[float, float, float],
    scaled_r_step: float,
    scaled_r_scan_max: float,
) -> float:
    r"""
    Find the lowest arrangement-asymptotic diatomic energy located by the ABC scan.

    Formula:
        For ``a=1,2,3`` and ``s_l=l Delta s``, ``l=1,...,M``, where
        ``M=floor(s_max/Delta s)``, the scan brackets one minimum in every
        arrangement. A bounded scalar minimization then gives

        E_0 = min_a min_{s in [s_(l_a-1),s_(l_a+1)]} V_a^asym(s),

        V_a^asym(s) = V_total[R_a=100 bohr,s,gamma_a=pi/2].

    Inputs:
        total_potential: TotalPES - native total three-body PES
        mass: tuple[float,float,float] - atomic masses in atomic units
        scaled_r_step: float - asymptotic scan spacing in bohr
        scaled_r_scan_max: float - largest scanned mass-scaled bond coordinate
            in bohr

    Returns:
        energy_zero: float - native-PES energy to subtract, in Hartree
    """
    if not np.isfinite(scaled_r_step) or scaled_r_step <= 0.0:
        message = f"scaled_r_step must be finite and positive, but got {scaled_r_step}"
        logger.error(message)
        raise ValueError(message)
    if not np.isfinite(scaled_r_scan_max) or scaled_r_scan_max <= scaled_r_step:
        message = f"scaled_r_scan_max must exceed scaled_r_step, but got {scaled_r_scan_max} and {scaled_r_step}"
        logger.error(message)
        raise ValueError(message)

    from pyticc.matrix.delves import asymptotic_potential

    n_scan = int(np.floor(scaled_r_scan_max / scaled_r_step + 1.0e-12))
    scaled_r = scaled_r_step * np.arange(1, n_scan + 1, dtype=np.float64)
    potential = asymptotic_potential(total_potential, mass)
    minima: list[float] = []
    for arrangement in range(1, 4):
        values = potential(arrangement, scaled_r)
        minimum_index = int(np.argmin(values))
        minimum = float(values[minimum_index])
        if 0 < minimum_index < scaled_r.size - 1:
            lower = float(scaled_r[minimum_index - 1])
            upper = float(scaled_r[minimum_index + 1])
            refined = cast(
                Any,
                minimize_scalar(
                    lambda coordinate, arrangement=arrangement: float(potential(arrangement, np.asarray([coordinate]))[0]),
                    bounds=(lower, upper),
                    method="bounded",
                    options={"xatol": 1.0e-12},
                ),
            )
            if refined.success:
                minimum = min(minimum, float(refined.fun))
        minima.append(minimum)
    return min(minima)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def prepare_Delves(
    total_potential: TotalPES,
    mass: tuple[float, float, float],
    *,
    energy_zero: str = "native",
    scaled_r_step: float = 0.01,
    scaled_r_scan_max: float = 10.0,
) -> DelvesMonomer:
    r"""
    Prepare physical monomer information for Delves reactive scattering.

    This step selects the common energy zero but does not construct an ABC sine
    basis, solve asymptotic rovibrational states, or truncate channels. Those
    operations require the channel-level ``E_Y_cut`` and belong to
    ``build_ScattSystem``.

    Formula:
        V_a^asym(s)=V_total[R_a=100,s,gamma_a=pi/2],

        V_used=V_native-E_0,

        where ``E_0=0`` for ``energy_zero='native'`` and is the lowest refined
        arrangement-asymptotic minimum for ``energy_zero='minimum'``.

    Inputs:
        total_potential: TotalPES - native scalar total three-body PES
        mass: tuple[float,float,float] - atomic masses in atomic units
        energy_zero: str - ``'native'`` or ``'minimum'``
        scaled_r_step: float - spacing used only to bracket an optional energy
            minimum, in bohr
        scaled_r_scan_max: float - upper boundary used only for the optional
            minimum search, in bohr

    Returns:
        monomer: DelvesMonomer - masses and selected total-PES energy zero
    """
    if not isinstance(total_potential, TotalPES):
        message = "Delves monomer preparation requires a TotalPES"
        logger.error(message)
        raise TypeError(message)
    masses = tuple(float(value) for value in mass)
    if len(masses) != 3:
        message = f"mass must contain three values, but got {mass!r}"
        logger.error(message)
        raise ValueError(message)
    if energy_zero not in ("native", "minimum"):
        message = f"energy_zero must be 'native' or 'minimum', but got {energy_zero!r}"
        logger.error(message)
        raise ValueError(message)

    zero = 0.0
    if energy_zero == "minimum":
        zero = _minimum_asymptotic_energy(total_potential, masses, scaled_r_step, scaled_r_scan_max)
        logger.info(f"Delves energy zero set to asymptotic minimum {zero:.12e} Hartree in the native PES convention")
    return DelvesMonomer(mass=masses, energy_zero=zero)


# ----------------------------------------------------------------------------------------
