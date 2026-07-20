from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from loguru import logger

from pyticc.constants import AMU2AU

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
}
# ----------------------------------------------------------------------------------------

ELEMENT_MASS_AU: dict[str, float] = {symbol: mass * AMU2AU for symbol, mass in ELEMENT_MASS_AMU.items()}


SUPPORTED_ELEMENT_SYMBOLS = tuple(ELEMENT_MASS_AU)


# ----------------------------------------------------------------------------------------
def element_mass_au(symbol: str) -> float:
    """
    Get the atomic mass of one supported element in atomic units.

    Inputs:
        symbol: str - element symbol (e.g., "H", "D", "He", "Li", "N", "O", "F", "S", "Cl", "Ar")

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
@dataclass(frozen=True)
class ScattSystem:
    monomer_X: MonomerSpec
    monomer_Y: MonomerSpec
    Jtot: int | None = None
    system_parity: int | None = None
    approx: Approx = Approx.EXACT

    def __post_init__(self) -> None:
        if self.Jtot is not None and self.Jtot < 0:
            message = f"Invalid Jtot {self.Jtot}. Must be non-negative."
            logger.error(message)
            raise ValueError(message)
        if self.system_parity is not None and self.system_parity not in (-1, 1):
            message = f"Invalid system_parity {self.system_parity}. Must be -1 or 1."
            logger.error(message)
            raise ValueError(message)


# ----------------------------------------------------------------------------------------
