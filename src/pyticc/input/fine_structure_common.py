from dataclasses import replace
from pathlib import Path
from typing import cast

from loguru import logger

from pyticc.constants import EnergyUnit
from pyticc.fine_structure import FSConstants, FSMonomerBasis, prepare_fs_monomer
from pyticc.input.common import TomlTable, required, resolve_path, section
from pyticc.pes.adiabatic import MonomerPES
from pyticc.pes.spin_resolved_atom_diatom import AtomDiatomOrbitalState, SpinResolvedAtomDiatomPES
from pyticc.pes.spin_resolved_diatom_diatom import OrbitalState, SpinResolvedDiatomDiatomPES
from pyticc.system import element_masses_au, reduced_mass

_CONSTANT_NAMES = ("A", "B", "D", "H", "gamma", "lambda_ss", "O", "P", "Q", "M", "N")


# ----------------------------------------------------------------------------------------
def integer_sequence(values: TomlTable, key: str) -> tuple[int, ...]:
    """Read a nonempty list of integer twice-quantum numbers."""
    raw = required(values, key)
    if not isinstance(raw, list) or not raw or any(not isinstance(value, int) or isinstance(value, bool) for value in raw):
        message = f"fine_structure {key} must be a nonempty list of integers, but got {raw!r}"
        logger.error(message)
        raise ValueError(message)
    return tuple(raw)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def constants(values: TomlTable, base: Path) -> FSConstants | Path:
    """Read a shared constants table or a vibrationally resolved CSV path."""
    csv_value = values.get("constants_csv")
    inline = values.get("constants")
    if csv_value is not None and inline is not None:
        message = "Specify constants_csv or constants, not both"
        logger.error(message)
        raise ValueError(message)
    if csv_value is not None:
        return resolve_path(base, str(csv_value))
    if not isinstance(inline, dict):
        message = "A fine-structure diatom requires constants_csv or a constants table"
        logger.error(message)
        raise ValueError(message)
    unknown = set(inline) - {*_CONSTANT_NAMES, "unit"}
    if unknown:
        message = f"Unknown fine-structure constants: {', '.join(sorted(unknown))}"
        logger.error(message)
        raise ValueError(message)
    unit = cast(EnergyUnit, str(inline.get("unit", "cm-1")))
    numbers = {name: float(inline.get(name, 0.0)) for name in _CONSTANT_NAMES}
    return FSConstants.from_unit(unit, **numbers)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_fs_diatom(
    symbols: tuple[str, str],
    basis_values: TomlTable,
    fs_values: TomlTable,
    potential: MonomerPES,
    base: Path,
) -> tuple[FSMonomerBasis, float]:
    """Build one fine-structure diatom and return its total mass."""
    interval = tuple(float(value) for value in required(basis_values, "r"))
    if len(interval) != 2:
        message = f"basis r must contain the two DVR boundaries, but got {interval}"
        logger.error(message)
        raise ValueError(message)
    mass_1, mass_2 = element_masses_au(*symbols)
    basis = prepare_fs_monomer(
        potential,
        r=cast(tuple[float, float], interval),
        n_dvr=int(required(basis_values, "n_dvr")),
        n_podvr=int(required(basis_values, "n_podvr")),
        vmax=int(required(basis_values, "vmax")),
        mass=reduced_mass(mass_1, mass_2),
        two_j_values=integer_sequence(fs_values, "two_j_values"),
        two_lambda_abs=int(required(fs_values, "two_lambda_abs")),
        two_S=int(required(fs_values, "two_S")),
        constants=constants(fs_values, base),
        reflection_parity=int(fs_values.get("reflection_parity", 1)),
        energy_zero=None if "energy_zero_au" not in fs_values else float(fs_values["energy_zero_au"]),
    )
    return basis, mass_1 + mass_2


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _electronic_order(config: TomlTable) -> tuple[tuple[int, ...], list[TomlTable]]:
    """Read the total-spin and orbital matrix-axis ordering from TOML."""
    values = section(section(config, "fine_structure"), "pes")
    spins = integer_sequence(values, "two_total_spins")
    raw_states = required(values, "orbital_states")
    if not isinstance(raw_states, list) or not raw_states or any(not isinstance(state, dict) for state in raw_states):
        message = "fine_structure.pes orbital_states must be a nonempty array of inline tables"
        logger.error(message)
        raise ValueError(message)
    return spins, cast(list[TomlTable], raw_states)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def bind_atom_diatom_order(config: TomlTable, pes: SpinResolvedAtomDiatomPES) -> SpinResolvedAtomDiatomPES:
    """Attach the TOML spin and orbital ordering to an A+BC PES callback."""
    spins, raw_states = _electronic_order(config)
    states = tuple(
        AtomDiatomOrbitalState(
            int(required(state, "two_lambda_atom")),
            int(required(state, "two_lambda_Y")),
        )
        for state in raw_states
    )
    return replace(pes, two_total_spins=spins, orbital_states=states)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def bind_diatom_diatom_order(config: TomlTable, pes: SpinResolvedDiatomDiatomPES) -> SpinResolvedDiatomDiatomPES:
    """Attach the TOML spin and orbital ordering to an AB+CD PES callback."""
    spins, raw_states = _electronic_order(config)
    states = tuple(
        OrbitalState(
            int(required(state, "two_lambda_X")),
            int(required(state, "two_lambda_Y")),
        )
        for state in raw_states
    )
    return replace(pes, two_total_spins=spins, orbital_states=states)


# ----------------------------------------------------------------------------------------
