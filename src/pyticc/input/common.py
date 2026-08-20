from pathlib import Path
from typing import Any, cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.monomer import DiatomBasis, prepare_Diatom
from pyticc.constants import CM2AU
from pyticc.energy import EnergyInput, get_Etot
from pyticc.pes.adiabatic import MonomerPES
from pyticc.propagation.config import Propagation, PropagationMode
from pyticc.system import Approx, element_masses_au, reduced_mass

TomlTable = dict[str, Any]


# ----------------------------------------------------------------------------------------
def required(table: TomlTable, key: str) -> Any:
    """Read a required TOML value."""
    try:
        return table[key]
    except KeyError as error:
        message = f"Missing required input {key!r}"
        logger.error(message)
        raise ValueError(message) from error


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def section(config: TomlTable, name: str) -> TomlTable:
    """Read and type-check one required TOML table."""
    value = required(config, name)
    if not isinstance(value, dict):
        message = f"Input section {name!r} must be a TOML table"
        logger.error(message)
        raise ValueError(message)
    return value


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def resolve_path(base: Path, value: str | Path) -> Path:
    """Resolve a user path relative to the input file."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def energies(value: Any, base: Path) -> NDArray[np.float64]:
    """Read total energies in cm-1 and convert them to atomic units."""
    source = resolve_path(base, value) if isinstance(value, str | Path) else value
    return get_Etot(cast(EnergyInput, source)) * CM2AU


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def k_cut(values: TomlTable) -> int | None:
    """Parse a non-negative helicity cutoff or the literal ``none``."""
    value = required(values, "K_cut")
    if value == "none":
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value

    message = f"K_cut must be a non-negative integer or 'none', but got {value!r}"
    logger.error(message)
    raise ValueError(message)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def approximation(config: TomlTable) -> tuple[Approx, int]:
    """Parse the exact, CS, or NNCC method and its neighboring-K range."""
    values = config.get("approximation", {"method": "exact"})
    if not isinstance(values, dict):
        message = "Input section 'approximation' must be a TOML table"
        logger.error(message)
        raise ValueError(message)

    method = required(values, "method")
    try:
        approx = Approx(str(method).lower())
    except ValueError as error:
        message = f"Approximation method must be 'exact', 'cs', or 'nncc', but got {method!r}"
        logger.error(message)
        raise ValueError(message) from error

    K_delta = int(values.get("K_delta", 1))
    if approx is Approx.NNCC and K_delta < 1:
        message = f"NNCC requires K_delta >= 1, but got K_delta={K_delta}"
        logger.error(message)
        raise ValueError(message)
    return approx, K_delta


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def propagation(config: TomlTable) -> Propagation:
    """Parse the radial propagation settings."""
    values = section(config, "propagation")
    return Propagation(
        boundaries=tuple(float(value) for value in required(values, "radial_boundaries")),
        half_steps=tuple(float(value) for value in required(values, "radial_half_steps")),
        mode=cast(PropagationMode, required(values, "mode")),
        memory_mb=float(values.get("memory_limit_mb", 512.0)),
        device=values.get("device", "auto"),
        print_verbose=values.get("print_verbose", False),
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def diatom_symbols(config: TomlTable, key: str) -> tuple[str, str]:
    """Read exactly two element symbols for one diatomic monomer."""
    symbols = tuple(str(symbol) for symbol in required(config, key))
    if len(symbols) != 2:
        message = f"{key} must contain two element symbols, but got {symbols}"
        logger.error(message)
        raise ValueError(message)
    return cast(tuple[str, str], symbols)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def state_int(values: TomlTable, key: str, n_state: int, default: int | None = None) -> int | tuple[int, ...]:
    """Read one integer shared by all electronic states or one per state."""
    value = values.get(key, default) if default is not None else required(values, key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if not isinstance(value, list) or len(value) != n_state or any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        message = f"basis {key} must be an integer or {n_state} integers, but got {value!r}"
        logger.error(message)
        raise ValueError(message)
    return tuple(value)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_diatom(symbols: tuple[str, str], values: TomlTable, potential: MonomerPES) -> tuple[DiatomBasis, float]:
    """Build an adiabatic diatomic basis and return its total mass."""
    interval = tuple(float(value) for value in required(values, "r"))
    if len(interval) != 2:
        message = f"basis r must contain the two DVR boundaries, but got {interval}"
        logger.error(message)
        raise ValueError(message)

    mass_1, mass_2 = element_masses_au(*symbols)
    total_mass = mass_1 + mass_2
    mass = reduced_mass(mass_1, mass_2)
    vmax = int(required(values, "vmax"))
    jmax = int(required(values, "jmax"))
    basis = prepare_Diatom(
        potential,
        r=(interval[0], interval[1]),
        n_dvr=int(required(values, "n_dvr")),
        n_podvr=int(required(values, "n_podvr")),
        vmax=vmax,
        jmax=jmax,
        mass=mass,
    )
    return basis, total_mass


# ----------------------------------------------------------------------------------------
