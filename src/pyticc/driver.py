import tomllib
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.atom_diatom import run_atom_diatom
from pyticc.basis.channel import TruncSpec
from pyticc.basis.dvr import build_SineDVR
from pyticc.basis.monomer import DiatomSpec
from pyticc.basis.podvr import RovibPODVR, build_RovibPODVR
from pyticc.constants import CM2AU
from pyticc.diatom_diatom import run_diatom_diatom
from pyticc.energy import EnergyInput, get_Etot
from pyticc.pes.fortran import load_fortran_pes
from pyticc.pes.wrapper import MonomerPES, PESWrapper
from pyticc.result import CoupledStatesResult, ScatteringResult
from pyticc.system import Approx, element_masses_au, reduced_mass

TomlTable = dict[str, Any]


def _required(table: TomlTable, key: str) -> Any:
    try:
        return table[key]
    except KeyError as error:
        message = f"Missing required input {key!r}"
        logger.error(message)
        raise ValueError(message) from error


def _section(config: TomlTable, name: str) -> TomlTable:
    value = _required(config, name)
    if not isinstance(value, dict):
        message = f"Input section {name!r} must be a TOML table"
        logger.error(message)
        raise ValueError(message)
    return value


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_pes(config: TomlTable, base: Path) -> PESWrapper:
    pes_input = _section(config, "pes")
    pes_dir = _resolve_path(base, pes_input.get("path", "."))
    sources = pes_input.get("sources", ["interaction-PES.f"])
    if isinstance(sources, str):
        sources = [sources]
    source_paths = [_resolve_path(pes_dir, source) for source in sources]
    wrapper = _resolve_path(pes_dir, pes_input.get("wrapper", "pyticc_wrapper.f90"))
    workdir = _resolve_path(pes_dir, pes_input.get("workdir", "."))
    return load_fortran_pes(source_paths, wrapper, workdir=workdir, processes=int(pes_input.get("processes", 1)))


def _get_energies_cm(value: Any, base: Path) -> NDArray[np.float64]:
    source = _resolve_path(base, value) if isinstance(value, str | Path) else value
    return get_Etot(cast(EnergyInput, source)) * CM2AU


def _get_K_cut(truncation_input: TomlTable) -> int | None:
    value = _required(truncation_input, "K_cut")
    if value == "none":
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value

    message = f"K_cut must be a non-negative integer or 'none', but got {value!r}"
    logger.error(message)
    raise ValueError(message)


def _get_approximation(config: TomlTable) -> tuple[Approx, int]:
    approximation_input = config.get("approximation", {"method": "exact"})
    if not isinstance(approximation_input, dict):
        message = "Input section 'approximation' must be a TOML table"
        logger.error(message)
        raise ValueError(message)

    method_value = _required(approximation_input, "method")
    try:
        approx = Approx(str(method_value).lower())
    except ValueError as error:
        message = f"Approximation method must be 'exact', 'cs', or 'nncc', but got {method_value!r}"
        logger.error(message)
        raise ValueError(message) from error

    K_delta = int(approximation_input.get("K_delta", 1))
    if approx is Approx.NNCC and K_delta < 1:
        message = f"NNCC requires K_delta >= 1, but got K_delta={K_delta}"
        logger.error(message)
        raise ValueError(message)
    return approx, K_delta


def _get_diatom_symbols(config: TomlTable, key: str) -> tuple[str, str]:
    symbols = tuple(str(symbol) for symbol in _required(config, key))
    if len(symbols) != 2:
        message = f"{key} must contain two element symbols, but got {symbols}"
        logger.error(message)
        raise ValueError(message)
    return cast(tuple[str, str], symbols)


def _build_diatom(symbols: tuple[str, str], basis_input: TomlTable, potential: MonomerPES) -> tuple[DiatomSpec, RovibPODVR, float]:
    radial_interval = tuple(float(value) for value in _required(basis_input, "r"))
    if len(radial_interval) != 2:
        message = f"basis r must contain the two DVR boundaries, but got {radial_interval}"
        logger.error(message)
        raise ValueError(message)

    mass_1, mass_2 = element_masses_au(*symbols)
    monomer_mass = mass_1 + mass_2
    monomer_reduced_mass = reduced_mass(mass_1, mass_2)
    vmax = int(_required(basis_input, "vmax"))
    jmax = int(_required(basis_input, "jmax"))
    dvr = build_SineDVR(
        a=radial_interval[0],
        b=radial_interval[1],
        n_dvr=int(_required(basis_input, "n_dvr")),
        mass=monomer_reduced_mass,
        pot_func=potential,
    )
    rovib = build_RovibPODVR(
        dvr=dvr,
        n_podvr=int(_required(basis_input, "n_podvr")),
        vmax=vmax,
        jmax=jmax,
        mass=monomer_reduced_mass,
    )
    diatom = DiatomSpec(
        Eint=rovib.E_vj - rovib.E_vj[0, 0],
        vmax=vmax,
        jmax=jmax,
        vmin=int(basis_input.get("vmin", 0)),
        jpar=int(basis_input.get("jpar", 0)),
    )
    return diatom, rovib, monomer_mass


def _run_atom_diatom(config: TomlTable, base: Path, pes: PESWrapper) -> ScatteringResult | CoupledStatesResult:
    atom_symbol = str(_required(config, "atom"))
    diatom_symbols = _get_diatom_symbols(config, "diatom")
    basis_input = _section(config, "basis")
    quadrature_input = _section(config, "quadrature")
    truncation_input = _section(config, "truncation")
    propagation_input = _section(config, "propagation")

    monomer_potential = pes.monomer_Y
    if monomer_potential is None:
        message = "Atom-diatom calculation requires the diatomic monomer potential as pes.monomer_Y"
        logger.error(message)
        raise ValueError(message)

    atom_mass = element_masses_au(atom_symbol)[0]
    diatom, rovib, diatom_mass = _build_diatom(diatom_symbols, basis_input, monomer_potential)
    collision_reduced_mass = reduced_mass(atom_mass, diatom_mass)
    approx, K_delta = _get_approximation(config)

    mode = cast(Literal["inelastic", "capture"], _required(propagation_input, "mode"))
    return run_atom_diatom(
        diatom,
        rovib,
        pes,
        Jtot=int(_required(config, "J")),
        system_parity=int(_required(config, "parity")),
        Etot=_get_energies_cm(_required(config, "energies_cm"), base),
        reduced_mass=collision_reduced_mass,
        radial_boundaries=_required(propagation_input, "radial_boundaries"),
        radial_half_steps=_required(propagation_input, "radial_half_steps"),
        trunc=TruncSpec(
            E_Y_cut=float(_required(truncation_input, "E_Y_cut_cm")) * CM2AU,
            K_cut=_get_K_cut(truncation_input),
        ),
        n_theta=int(_required(quadrature_input, "n_theta")),
        mode=mode,
        approx=approx,
        K_delta=K_delta,
    )


def _run_diatom_diatom(config: TomlTable, base: Path, pes: PESWrapper) -> ScatteringResult | CoupledStatesResult:
    symbols_X = _get_diatom_symbols(config, "diatom_X")
    symbols_Y = _get_diatom_symbols(config, "diatom_Y")
    basis_X_input = _section(config, "basis_X")
    basis_Y_input = _section(config, "basis_Y")
    quadrature_input = _section(config, "quadrature")
    truncation_input = _section(config, "truncation")
    propagation_input = _section(config, "propagation")

    if pes.monomer_X is None or pes.monomer_Y is None:
        message = "Diatom-diatom calculation requires both pes.monomer_X and pes.monomer_Y"
        logger.error(message)
        raise ValueError(message)

    diatom_X, rovib_X, mass_X = _build_diatom(symbols_X, basis_X_input, pes.monomer_X)
    diatom_Y, rovib_Y, mass_Y = _build_diatom(symbols_Y, basis_Y_input, pes.monomer_Y)
    mode = cast(Literal["inelastic", "capture"], _required(propagation_input, "mode"))
    approx, K_delta = _get_approximation(config)
    return run_diatom_diatom(
        diatom_X,
        rovib_X,
        diatom_Y,
        rovib_Y,
        pes,
        Jtot=int(_required(config, "J")),
        system_parity=int(_required(config, "parity")),
        Etot=_get_energies_cm(_required(config, "energies_cm"), base),
        reduced_mass=reduced_mass(mass_X, mass_Y),
        radial_boundaries=_required(propagation_input, "radial_boundaries"),
        radial_half_steps=_required(propagation_input, "radial_half_steps"),
        trunc=TruncSpec(
            E_X_cut=float(_required(truncation_input, "E_X_cut_cm")) * CM2AU,
            E_Y_cut=float(_required(truncation_input, "E_Y_cut_cm")) * CM2AU,
            K_cut=_get_K_cut(truncation_input),
        ),
        n_theta_X=int(_required(quadrature_input, "n_theta_X")),
        n_theta_Y=int(_required(quadrature_input, "n_theta_Y")),
        n_phi=int(_required(quadrature_input, "n_phi")),
        mode=mode,
        approx=approx,
        K_delta=K_delta,
    )


# ----------------------------------------------------------------------------------------
def run(source: str | Path, *, pes: PESWrapper | None = None) -> ScatteringResult | CoupledStatesResult:
    """
    Run a scattering calculation from a compact TOML input file.

    Relative PES and energy-file paths are resolved from the directory containing
    the input file. A Python ``PESWrapper`` may be supplied directly instead of a
    ``[pes]`` table.

    Inputs:
        source: str | Path - TOML input file
        pes: PESWrapper | None - optional Python potential-energy surfaces

    Returns:
        result: ScatteringResult | CoupledStatesResult - exact result or separated CS/NNCC blocks
    """
    input_path = Path(source).expanduser().resolve()
    try:
        with input_path.open("rb") as file:
            config = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        message = f"Failed to read PyTICC input {input_path}: {error}"
        logger.error(message)
        raise ValueError(message) from error

    potential = _load_pes(config, input_path.parent) if pes is None else pes
    calculation_type = _required(config, "type")
    if calculation_type == "atom-diatom":
        return _run_atom_diatom(config, input_path.parent, potential)
    if calculation_type == "diatom-diatom":
        return _run_diatom_diatom(config, input_path.parent, potential)

    message = f"Unsupported calculation type {calculation_type!r}; supported: 'atom-diatom', 'diatom-diatom'"
    logger.error(message)
    raise ValueError(message)
