from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import pi
from typing import Any, cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from pyticc.basis.delves import clean_sine_phases, midpoint_quad, sine_basis, sine_kinetic, sine_reference_hamiltonian
from pyticc.pes.total import TotalPES

AsymptoticPotential = Callable[[int, NDArray[np.float64]], NDArray[np.float64]]


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DelvesDiatomBasis:
    """
    Multiple-arrangement asymptotic diatomic states in ABC's sine FBR.

    This internal intermediate contains the rovibrational states of the three
    possible diatoms, but no helicity or total-angular-momentum channels and no
    PODVR contraction.

    Members:
        mass: tuple[float,float,float] - atomic masses ``(A,B,C)`` in atomic units
        jmax: int - largest solved diatomic rotational angular momentum
        E_max: float - maximum internal energy in any ABC channel, in Hartree
        rho_min: float - ABC inner hyperradial hard wall in bohr
        scaled_r_max: float - common mass-scaled diatomic coordinate boundary in bohr
        n_sine: int - number of primitive particle-in-a-box sine functions
        n_vib_quad: int - number of midpoint quadrature points in ``s`` or ``theta``
        n_gamma_quad: int - number of Gauss--Legendre Jacobi-angle points
        qns: tuple[tuple[int,int,int],...] - retained labels ``(arrangement,v,j)``
        energies: NDArray[np.float64] - rovibrational energies in Hartree, shape
            ``(n_state,)``
        coefficients: NDArray[np.float64] - sine-FBR coefficients indexed as
            ``[sine,state]``, shape ``(n_sine,n_state)``
        energy_zero: float - native total-PES energy subtracted before construction,
            in Hartree
    """

    mass: tuple[float, float, float]
    jmax: int
    E_max: float
    rho_min: float
    scaled_r_max: float
    n_sine: int
    n_vib_quad: int
    n_gamma_quad: int
    qns: tuple[tuple[int, int, int], ...]
    energies: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    energy_zero: float = 0.0

    def __post_init__(self) -> None:
        energies = np.asarray(self.energies, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if energies.shape != (len(self.qns),):
            message = f"Delves diatom energies must have shape {(len(self.qns),)}, but got {energies.shape}"
            logger.error(message)
            raise ValueError(message)
        if coefficients.shape != (self.n_sine, len(self.qns)):
            message = f"Delves diatom coefficients must have shape {(self.n_sine, len(self.qns))}, but got {coefficients.shape}"
            logger.error(message)
            raise ValueError(message)
        if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(coefficients)) or not np.isfinite(self.energy_zero):
            message = "Delves diatom energies, coefficients, and energy_zero must be finite"
            logger.error(message)
            raise ValueError(message)
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def n_state(self) -> int:
        """Return the number of retained arrangement rovibrational states."""
        return len(self.qns)


# ----------------------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------------------
def build_delves_diatom_basis(
    asymptotic_potential: AsymptoticPotential,
    mass: Sequence[float],
    jmax: int,
    E_max: float,
    *,
    energy_zero: float = 0.0,
    scaled_r_step: float = 0.01,
    scaled_r_scan_max: float = 10.0,
    tail_cut: float = 5.0,
) -> DelvesDiatomBasis:
    r"""
    Prepare ABC's three arrangement-asymptotic diatomic bases without PODVR.

    The automatic WKB scan determines one common sine FBR. For every
    arrangement ``a=1,2,3`` and ``j=0,...,jmax``, the fixed-Jacobi problem is
    diagonalized and states below ``E_max`` are retained.

    Formula:
        u_n(s)=sqrt(2/s_max) sin(n pi s/s_max),

        H_nm^(a,j) = delta_nm (n pi/s_max)^2/(2 mu)
          + integral_0^s_max u_n(s)
            [j(j+1)/(2 mu s^2)+V_a^asym(s)]u_m(s) ds,

        H^(a,j) c^(a,v,j) = E_(a,v,j) c^(a,v,j).

    Inputs:
        asymptotic_potential: AsymptoticPotential - arrangement-indexed,
            energy-zero-adjusted asymptotic potential in Hartree
        mass: Sequence[float] - atomic masses ``(A,B,C)`` in atomic units
        jmax: int - largest diatomic rotational angular momentum
        E_max: float - maximum internal energy in any ABC channel, in Hartree
        energy_zero: float - native total-PES energy already subtracted, in Hartree
        scaled_r_step: float - WKB scan spacing in bohr
        scaled_r_scan_max: float - WKB scan upper boundary in bohr
        tail_cut: float - forbidden-tail action cutoff

    Returns:
        basis: DelvesDiatomBasis - prepared arrangement rovibrational states and
            common primitive specification
    """
    if jmax < 0:
        message = f"jmax must be non-negative, but got {jmax}"
        logger.error(message)
        raise ValueError(message)
    masses, rho_min, scaled_r_max, n_sine, n_vib_quad, n_gamma_quad = _resolve_delves_sizes(
        asymptotic_potential,
        mass,
        jmax,
        E_max,
        scaled_r_step=scaled_r_step,
        scaled_r_scan_max=scaled_r_scan_max,
        tail_cut=tail_cut,
    )
    reduced_mass = np.sqrt(np.prod(masses) / sum(masses))
    s_grid, s_weights = midpoint_quad(0.0, scaled_r_max, n_vib_quad)
    s_values = sine_basis(0.0, scaled_r_max, n_sine, s_grid)
    kinetic = sine_kinetic(n_sine, scaled_r_max, reduced_mass, theta=False)
    records: list[tuple[tuple[int, int, int], float, NDArray[np.float64]]] = []

    for arrangement in range(1, 4):
        potential = np.asarray(asymptotic_potential(arrangement, s_grid), dtype=np.float64)
        for j in range(jmax + 1):
            effective = potential + j * (j + 1) / (2.0 * reduced_mass * s_grid**2)
            hamiltonian = sine_reference_hamiltonian(s_values, s_weights, kinetic, effective)
            energies, vectors = np.linalg.eigh(hamiltonian)
            vectors = clean_sine_phases(vectors, s_values)
            for v in np.flatnonzero(energies <= E_max):
                records.append(((arrangement, int(v), j), float(energies[v]), vectors[:, v].copy()))

    if not records:
        message = f"No asymptotic diatomic states lie below E_max={E_max} Hartree"
        logger.error(message)
        raise ValueError(message)
    records.sort(key=lambda item: item[0])
    return DelvesDiatomBasis(
        mass=masses,
        jmax=jmax,
        E_max=float(E_max),
        rho_min=rho_min,
        scaled_r_max=scaled_r_max,
        n_sine=n_sine,
        n_vib_quad=n_vib_quad,
        n_gamma_quad=n_gamma_quad,
        qns=tuple(record[0] for record in records),
        energies=np.asarray([record[1] for record in records], dtype=np.float64),
        coefficients=np.column_stack([record[2] for record in records]),
        energy_zero=float(energy_zero),
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _resolve_delves_sizes(
    asymptotic_potential: AsymptoticPotential,
    mass: Sequence[float],
    jmax: int,
    E_max: float,
    *,
    scaled_r_step: float,
    scaled_r_scan_max: float,
    tail_cut: float,
) -> tuple[tuple[float, float, float], float, float, int, int, int]:
    r"""
    Run ABC's WKB scan and return the common primitive sizes.

    Formula:
        With ``mu=sqrt(m_A m_B m_C/(m_A+m_B+m_C))``, sampled points
        ``s_l=l Delta s``, and arrangement potentials ``V_a(s_l)``, forbidden
        tails are accumulated from the two ``E_max`` crossings until

        sum_l Delta s sqrt(2 mu [V_a(s_l)-E_max]) > tail_cut.

        The three limits determine

        rho_min^2 = sum_a (1-m_a/M_tot) [s_min^(a)]^2,

        s_max = max_a s_max^(a),

        n_sine = max_a floor[(2 s_max/pi)
                              sqrt(2 mu [E_max-V_min^(a)])],

        n_gamma = floor{3[2(jmax+1)+(n_sine+1)]/4},

        n_vib = floor{3[2(n_sine+1)+(jmax+1)]/4}.

    Inputs:
        asymptotic_potential: AsymptoticPotential - arrangement-indexed
            asymptotic potential in Hartree
        mass: Sequence[float] - three atomic masses in atomic units
        jmax: int - largest diatomic rotational angular momentum
        E_max: float - largest retained channel internal energy in Hartree
        scaled_r_step: float - scan spacing in bohr
        scaled_r_scan_max: float - largest scanned coordinate in bohr
        tail_cut: float - forbidden-tail action target

    Returns:
        sizes: tuple - masses, rho_min, scaled_r_max, n_sine, n_vib_quad,
            and n_gamma_quad
    """
    masses = tuple(float(value) for value in mass)
    if len(masses) != 3 or not np.all(np.isfinite(masses)) or not np.all(np.asarray(masses) > 0.0):
        message = f"mass must contain three finite positive atomic masses, but got {mass!r}"
        logger.error(message)
        raise ValueError(message)
    if not np.isfinite(E_max):
        message = f"E_max must be finite, but got {E_max}"
        logger.error(message)
        raise ValueError(message)
    if not np.isfinite(scaled_r_step) or scaled_r_step <= 0.0:
        message = f"scaled_r_step must be finite and positive, but got {scaled_r_step}"
        logger.error(message)
        raise ValueError(message)
    if not np.isfinite(scaled_r_scan_max) or scaled_r_scan_max <= scaled_r_step:
        message = f"scaled_r_scan_max must exceed scaled_r_step, but got {scaled_r_scan_max} and {scaled_r_step}"
        logger.error(message)
        raise ValueError(message)
    if not np.isfinite(tail_cut) or tail_cut <= 0.0:
        message = f"tail_cut must be finite and positive, but got {tail_cut}"
        logger.error(message)
        raise ValueError(message)

    n_scan = int(np.floor(scaled_r_scan_max / scaled_r_step + 1.0e-12))
    scaled_r = scaled_r_step * np.arange(1, n_scan + 1, dtype=np.float64)
    total_mass = sum(masses)
    reduced_mass = np.sqrt(np.prod(masses) / total_mass)
    radial_scale = 2.0 * reduced_mass
    scaled_E_max = radial_scale * E_max
    inner_limits = np.empty(3, dtype=np.float64)
    outer_limits = np.empty(3, dtype=np.float64)
    potential_minima = np.empty(3, dtype=np.float64)

    for arrangement in range(1, 4):
        potential = np.asarray(asymptotic_potential(arrangement, scaled_r), dtype=np.float64)
        if potential.shape != scaled_r.shape or not np.all(np.isfinite(potential)):
            message = (
                f"asymptotic_potential returned shape {potential.shape} for arrangement {arrangement}; "
                f"expected finite values with shape {scaled_r.shape}"
            )
            logger.error(message)
            raise ValueError(message)
        scaled_potential = radial_scale * potential
        minimum_index = int(np.argmin(scaled_potential))
        potential_minima[arrangement - 1] = scaled_potential[minimum_index]
        inner_limits[arrangement - 1], outer_limits[arrangement - 1] = _wkb_limits(
            scaled_potential,
            scaled_E_max,
            minimum_index,
            scaled_r_step,
            tail_cut,
            arrangement,
        )

    scaled_r_max = float(np.max(outer_limits))
    rho_min = float(np.sqrt(np.sum((1.0 - np.asarray(masses) / total_mass) * inner_limits**2)))
    basis_estimates = np.where(
        scaled_E_max > potential_minima,
        (2.0 * scaled_r_max / pi) * np.sqrt(np.maximum(0.0, scaled_E_max - potential_minima)),
        0.0,
    )
    n_sine = int(np.max(basis_estimates))
    if n_sine < 1:
        message = "The ABC estimate produced no sine functions; increase E_max or inspect the asymptotic potentials"
        logger.error(message)
        raise ValueError(message)
    n_gamma_quad = 3 * (2 * (jmax + 1) + (n_sine + 1)) // 4
    n_vib_quad = 3 * (2 * (n_sine + 1) + (jmax + 1)) // 4
    return masses, rho_min, scaled_r_max, n_sine, n_vib_quad, n_gamma_quad


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _wkb_limits(
    scaled_potential: NDArray[np.float64],
    scaled_E_max: float,
    minimum_index: int,
    step: float,
    tail_cut: float,
    arrangement: int,
) -> tuple[float, float]:
    """Return inner and outer WKB limits for one sampled asymptotic potential."""
    left_allowed = np.flatnonzero(scaled_potential[: minimum_index + 1] < scaled_E_max)
    left_start = int(left_allowed[0]) - 1 if left_allowed.size else minimum_index
    left_action = 0.0
    inner_index = -1
    for index in range(left_start, -1, -1):
        inner_index = index
        left_action += step * np.sqrt(scaled_potential[index] - scaled_E_max)
        if left_action > tail_cut:
            break
    if inner_index >= 0 and left_action <= tail_cut:
        message = f"The inner WKB tail for arrangement {arrangement} did not reach tail_cut={tail_cut} within the scaled-r scan"
        logger.error(message)
        raise ValueError(message)

    right_allowed = np.flatnonzero(scaled_potential[minimum_index:] < scaled_E_max)
    if not right_allowed.size:
        message = f"No outer forbidden region below scaled_r_scan_max was found for arrangement {arrangement}"
        logger.error(message)
        raise ValueError(message)
    right_start = minimum_index + int(right_allowed[-1]) + 1
    if right_start >= scaled_potential.size:
        message = f"No outer forbidden region below scaled_r_scan_max was found for arrangement {arrangement}"
        logger.error(message)
        raise ValueError(message)
    right_action = 0.0
    outer_index = scaled_potential.size
    for index in range(right_start, scaled_potential.size):
        outer_index = index
        right_action += step * np.sqrt(scaled_potential[index] - scaled_E_max)
        if right_action > tail_cut:
            break
    if outer_index >= scaled_potential.size or right_action <= tail_cut:
        message = f"The outer WKB tail for arrangement {arrangement} did not reach tail_cut={tail_cut} within the scaled-r scan"
        logger.error(message)
        raise ValueError(message)

    return (inner_index + 1) * step, (outer_index + 1) * step


# ----------------------------------------------------------------------------------------
