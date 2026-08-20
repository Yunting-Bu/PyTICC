from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.monomer.delves import DelvesMonomer
from pyticc.constants import AMU2AU
from pyticc.pes.adiabatic import PESWrapper
from pyticc.pes.diabatic import DiabaticPESWrapper
from pyticc.pes.total import TotalPES

# ----------------------------------------------------------------------------------------
# Mass
ELEMENT_MASS_AMU: dict[str, float] = {
    "H": 1.00782503223,
    "D": 2.01410177812,
    "He": 4.002602,
    "Li": 6.938,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "S": 32.06,
    "Cl": 34.968852682,
    "Ar": 39.9623831237,
    "K": 39.963998166,
    "Rb": 86.909180531,
}
# ----------------------------------------------------------------------------------------

ELEMENT_MASS_AU: dict[str, float] = {symbol: mass * AMU2AU for symbol, mass in ELEMENT_MASS_AMU.items()}


SUPPORTED_ELEMENT_SYMBOLS = tuple(ELEMENT_MASS_AU)


# ----------------------------------------------------------------------------------------
def element_mass_au(symbol: str) -> float:
    """
    Get the atomic mass of one supported element in atomic units.

    Inputs:
        symbol: str - element symbol (e.g., "H", "D", "He", "Li", "N", "O", "F", "S", "Cl", "Ar", "K", "Rb")

    Returns:
        mass: float - atomic mass in atomic units
    """
    if symbol in ELEMENT_MASS_AU:
        return ELEMENT_MASS_AU[symbol]

    supported = ", ".join(SUPPORTED_ELEMENT_SYMBOLS)
    message = f"Unsupported element symbol {symbol!r}. Supported: {supported}"
    logger.error(message)
    raise ValueError(message)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def element_masses_au(*symbols: str) -> tuple[float, ...]:
    """
    Get the atomic masses of several supported elements in atomic units.

    Inputs:
        symbols: str - element symbols in the requested output order

    Returns:
        masses: tuple[float, ...] - atomic masses in the same order as symbols
    """
    return tuple(element_mass_au(symbol) for symbol in symbols)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def reduced_mass(mass1: float, mass2: float) -> float:
    r"""
    Get the reduced mass for two positive masses.

    Formula:
        \mu = m1 * m2 / (m1 + m2)

    Inputs:
        m1: float - mass of the first particle (must be positive)
        m2: float - mass of the second particle (must be positive)

    Returns:
        \mu: float - reduced mass of the two particles
    """
    m1 = float(mass1)
    m2 = float(mass2)
    return m1 * m2 / (m1 + m2)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
class MonomerType(Enum):
    ATOM = "1"
    DIATOM = "2"
    TRIATOM = "3"


class Approx(Enum):
    EXACT = "exact"
    CS = "cs"
    NNCC = "nncc"


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MolInnerState:
    j: int
    v: int | None = None
    t: int | None = None
    Eint: float = 0.0
    electronic_state: int | None = None


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
class MonomerSpec(Protocol):
    @property
    def type(self) -> MonomerType:
        """Return the monomer structural type."""
        ...

    def mis_iter(self, E_cut: float) -> Iterator[MolInnerState]:
        """Iterate over internal states below the requested energy cutoff."""
        ...

    def energy(self, mis: MolInnerState, K: int) -> float:
        """Return the internal energy of a state in one helicity block."""
        ...

    def allows_K(self, mis: MolInnerState, K: int) -> bool:
        """Return whether a state exists in the requested helicity block."""
        ...


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
class ElectricMonomerSpec(Protocol):
    @property
    def m_values(self) -> tuple[int, ...]:
        """Return available space-fixed monomer projections."""
        ...

    def relative_energies(self, m: int) -> NDArray[np.float64]:
        """Return dressed energies for one space-fixed projection."""
        ...


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
class ChannelBasisSpec(Protocol):
    @property
    def n_channel(self) -> int:
        """Return the number of prepared scattering channels."""
        ...

    @property
    def E_int(self) -> NDArray[np.float64]:
        """Return channel thresholds with shape (n_channel,)."""
        ...


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """
    Selection rules and truncations used to construct scattering channels.

    A scalar ``vmin`` or ``exchange_parity`` applies to every electronic state.
    A tuple supplies one value per diabatic electronic state.

    Members:
        vmin_X: int | tuple[int,...] - smallest retained vibrational quantum
            number of monomer X
        vmin_Y: int | tuple[int,...] - smallest retained vibrational quantum
            number of monomer Y
        exchange_parity_X: int | tuple[int,...] - rotational exchange parity of
            monomer X: -1 odd j, 0 every j, or 1 even j
        exchange_parity_Y: int | tuple[int,...] - rotational exchange parity of
            monomer Y: -1 odd j, 0 every j, or 1 even j; this is ABC ``jpar``
            for a Delves 1+2 calculation
        E_X_cut: float - X-monomer internal-energy cutoff in atomic units
        E_Y_cut: float - Y-monomer internal-energy cutoff in atomic units; this
            is ABC ``emax`` for a Delves 1+2 calculation
        K_cut: int | None - maximum retained helicity, or None to retain every
            angular-momentum-allowed K
    """

    vmin_X: int | tuple[int, ...] = 0
    vmin_Y: int | tuple[int, ...] = 0
    exchange_parity_X: int | tuple[int, ...] = 0
    exchange_parity_Y: int | tuple[int, ...] = 0
    E_X_cut: float = math.inf
    E_Y_cut: float = math.inf
    K_cut: int | None = None

    def __post_init__(self) -> None:
        for name, values in (("vmin_X", self.vmin_X), ("vmin_Y", self.vmin_Y)):
            selected = (values,) if isinstance(values, int) else values
            if not selected or any(value < 0 for value in selected):
                message = f"{name} must contain non-negative integers, but got {values!r}"
                logger.error(message)
                raise ValueError(message)
        for name, values in (
            ("exchange_parity_X", self.exchange_parity_X),
            ("exchange_parity_Y", self.exchange_parity_Y),
        ):
            selected = (values,) if isinstance(values, int) else values
            if not selected or any(value not in (-1, 0, 1) for value in selected):
                message = f"{name} must contain only -1, 0, or 1, but got {values!r}"
                logger.error(message)
                raise ValueError(message)
        if any(np.isnan(value) or np.isneginf(value) for value in (self.E_X_cut, self.E_Y_cut)):
            message = f"Energy cutoffs must be finite or positive infinity, but got E_X_cut={self.E_X_cut}, E_Y_cut={self.E_Y_cut}"
            logger.error(message)
            raise ValueError(message)
        if self.K_cut is not None and self.K_cut < 0:
            message = f"K_cut must be non-negative, but got K_cut={self.K_cut}"
            logger.error(message)
            raise ValueError(message)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ScattSystem:
    """Physical definition of one scattering block.

    Members:
        monomer_X: MonomerSpec | DelvesMonomer - first monomer internal-state
            model, or physical Delves monomer information
        monomer_Y: MonomerSpec | ElectricMonomerSpec | None - second monomer
            internal-state model; None for a Delves calculation
        Jtot: int | None - conserved total angular momentum for a field-free
            calculation
        system_parity: int | None - conserved total parity for a field-free
            calculation, -1 or 1
        M: int | None - conserved projection on the SF electric-field axis for
            an electric-field calculation
        approx: Approx - exact CC, CS, or NNCC approximation
        K_delta: int - neighboring-K range used by NNCC
        potential: PESWrapper | DiabaticPESWrapper | None - interaction PES for
            a nonreactive calculation
        total_potential: TotalPES | None - scalar total three-body PES for a
            Delves reactive calculation
        reduced_mass: float | None - collision reduced mass in atomic units
        channel_spec: ChannelSpec | None - channel selections used to prepare
            ``basis``; None is allowed only for low-level direct construction
        basis: ChannelBasisSpec | None - channel basis prepared together with
            the system; direct construction may leave it unset for low-level use
    """

    monomer_X: MonomerSpec | DelvesMonomer
    monomer_Y: MonomerSpec | ElectricMonomerSpec | None = None
    Jtot: int | None = None
    system_parity: int | None = None
    M: int | None = None
    approx: Approx = Approx.EXACT
    K_delta: int = 1
    potential: PESWrapper | DiabaticPESWrapper | None = None
    total_potential: TotalPES | None = None
    reduced_mass: float | None = None
    channel_spec: ChannelSpec | None = None
    basis: ChannelBasisSpec | None = None

    def __post_init__(self) -> None:
        if self.Jtot is not None and self.Jtot < 0:
            message = f"Invalid Jtot {self.Jtot}. Must be non-negative."
            logger.error(message)
            raise ValueError(message)
        if self.system_parity is not None and self.system_parity not in (-1, 1):
            message = f"Invalid system_parity {self.system_parity}. Must be -1 or 1."
            logger.error(message)
            raise ValueError(message)
        if self.K_delta < 1:
            message = f"Invalid K_delta {self.K_delta}. Must be positive."
            logger.error(message)
            raise ValueError(message)
        if self.reduced_mass is not None and self.reduced_mass <= 0.0:
            message = f"Invalid reduced_mass {self.reduced_mass}. Must be positive."
            logger.error(message)
            raise ValueError(message)

    @property
    def n_channel(self) -> int:
        """Return the number of channels prepared for this system."""
        if self.basis is None:
            message = "Scattering channels have not been prepared; use build_ScattSystem"
            logger.error(message)
            raise ValueError(message)
        return self.basis.n_channel


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_ScattSystem(
    monomer_X: MonomerSpec | DelvesMonomer,
    monomer_Y: MonomerSpec | ElectricMonomerSpec | None = None,
    *,
    Jtot: int | None = None,
    system_parity: int | None = None,
    M: int | None = None,
    channel: ChannelSpec | None = None,
    jmax: int | None = None,
    lmax: int | None = None,
    approx: Approx = Approx.EXACT,
    K_delta: int = 1,
    potential: PESWrapper | DiabaticPESWrapper | None = None,
    total_potential: TotalPES | None = None,
    reduced_mass: float | None = None,
) -> ScattSystem:
    """
    Build a scattering system and its channel basis in one step.

    Inputs:
        monomer_X: MonomerSpec | DelvesMonomer - prepared first monomer, or
            physical Delves monomer information
        monomer_Y: MonomerSpec | ElectricMonomerSpec | None - prepared second
            monomer; None for a Delves calculation
        Jtot: int | None - conserved total angular momentum for a field-free block
        system_parity: int | None - conserved field-free total parity, -1 or 1
        M: int | None - conserved SF projection for an electric-field block
        channel: ChannelSpec | None - vibrational, exchange-parity, energy, and
            helicity selections used to construct channels
        jmax: int | None - largest diatomic rotational quantum number for a
            Delves channel basis
        lmax: int | None - largest end-over-end angular momentum for an
            electric-field block
        approx: Approx - exact CC, CS, or NNCC approximation
        K_delta: int - neighboring-K range used by NNCC
        potential: PESWrapper | DiabaticPESWrapper | None - interaction PES
        total_potential: TotalPES | None - Delves scalar total three-body PES
        reduced_mass: float | None - collision reduced mass in atomic units

    Returns:
        system: ScattSystem - physical system containing its prepared channels
    """
    from pyticc.basis.channel import ChannelBasis, ChannelBasisElectricSF, build_ChannelBasis, build_ChannelBasisElectricSF
    from pyticc.basis.monomer.diatom_electric import DiatomElectricBasis

    channel_spec = ChannelSpec() if channel is None else channel
    system = ScattSystem(
        monomer_X=monomer_X,
        monomer_Y=monomer_Y,
        Jtot=Jtot,
        system_parity=system_parity,
        M=M,
        approx=approx,
        K_delta=K_delta,
        potential=potential,
        total_potential=total_potential,
        reduced_mass=reduced_mass,
        channel_spec=channel_spec,
    )

    if isinstance(monomer_X, DelvesMonomer):
        from pyticc.basis.delves import build_delves_channels, build_delves_diatom_basis
        from pyticc.matrix.delves import asymptotic_potential

        if monomer_Y is not None:
            message = "Delves system construction accepts one prepared three-arrangement monomer object"
            logger.error(message)
            raise TypeError(message)
        if Jtot is None or system_parity is None:
            message = "Delves system construction requires Jtot and system_parity"
            logger.error(message)
            raise ValueError(message)
        if not isinstance(total_potential, TotalPES):
            message = "Delves system construction requires total_potential to be a TotalPES"
            logger.error(message)
            raise TypeError(message)
        if jmax is None or jmax < 0:
            message = f"Delves system construction requires non-negative jmax, but got {jmax}"
            logger.error(message)
            raise ValueError(message)
        if not np.isfinite(channel_spec.E_Y_cut):
            message = f"Delves system construction requires finite channel.E_Y_cut, but got {channel_spec.E_Y_cut}"
            logger.error(message)
            raise ValueError(message)
        if not isinstance(channel_spec.exchange_parity_Y, int):
            message = "Delves system construction requires scalar channel.exchange_parity_Y"
            logger.error(message)
            raise TypeError(message)
        if potential is not None:
            message = "Delves system construction uses total_potential, not the nonreactive interaction potential"
            logger.error(message)
            raise ValueError(message)
        if reduced_mass is not None:
            message = "Delves hyperradial mass is derived from the three atomic masses; do not provide reduced_mass"
            logger.error(message)
            raise ValueError(message)
        if approx is not Approx.EXACT:
            message = "Delves reactive scattering currently supports only exact coupled channels"
            logger.error(message)
            raise ValueError(message)
        native_total_potential = total_potential
        shifted_total_potential = native_total_potential
        if monomer_X.energy_zero != 0.0:
            shifted_total_potential = TotalPES(lambda bonds: native_total_potential(bonds) - monomer_X.energy_zero)
        diatom_basis = build_delves_diatom_basis(
            asymptotic_potential(shifted_total_potential, monomer_X.mass),
            monomer_X.mass,
            jmax=jmax,
            E_max=channel_spec.E_Y_cut,
            energy_zero=monomer_X.energy_zero,
        )
        K_cut = jmax if channel_spec.K_cut is None else channel_spec.K_cut
        delves_basis = build_delves_channels(
            diatom_basis,
            Jtot=Jtot,
            system_parity=system_parity,
            exchange_parity=channel_spec.exchange_parity_Y,
            K_cut=K_cut,
        )
        return replace(system, basis=delves_basis)

    if monomer_Y is None:
        message = "Nonreactive system construction requires two prepared monomers"
        logger.error(message)
        raise ValueError(message)
    if total_potential is not None:
        message = "Nonreactive system construction uses potential, not total_potential"
        logger.error(message)
        raise ValueError(message)

    if isinstance(monomer_Y, DiatomElectricBasis):
        if M is None or lmax is None:
            message = "Electric-field system construction requires M and lmax"
            logger.error(message)
            raise ValueError(message)
        basis: ChannelBasis | ChannelBasisElectricSF = build_ChannelBasisElectricSF(
            monomer_Y,
            M=M,
            lmax=lmax,
            channel=channel_spec,
        )
    else:
        if Jtot is None or system_parity is None:
            message = "Field-free system construction requires Jtot and system_parity"
            logger.error(message)
            raise ValueError(message)
        basis = build_ChannelBasis(system, channel_spec)

    return replace(system, basis=basis)


# ----------------------------------------------------------------------------------------
