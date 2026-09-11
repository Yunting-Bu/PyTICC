import tomllib
from pathlib import Path
from time import perf_counter, process_time

from loguru import logger

import pyticc.input.atom_diatom as atom_diatom
import pyticc.input.diabatic as diabatic
import pyticc.input.diatom_diatom as diatom_diatom
import pyticc.input.fine_structure_atom_atom as fine_structure_atom_atom
import pyticc.input.fine_structure_atom_diatom as fine_structure_atom_diatom
import pyticc.input.fine_structure_diatom_diatom as fine_structure_diatom_diatom
from pyticc.input.common import TomlTable, resolve_path, section
from pyticc.pes.adiabatic import PESWrapper
from pyticc.pes.diabatic import DiabaticPESWrapper
from pyticc.pes.fortran import load_fortran_diabatic_pes, load_fortran_pes
from pyticc.pes.spin_resolved_atom_atom import SpinResolvedAtomAtomPES
from pyticc.pes.spin_resolved_atom_diatom import SpinResolvedAtomDiatomPES
from pyticc.pes.spin_resolved_diatom_diatom import SpinResolvedDiatomDiatomPES
from pyticc.result import CoupledStatesResult, ScatteringResult, Timing
from pyticc.system import ScatteringType


# ----------------------------------------------------------------------------------------
def _load_pes(config: TomlTable, base: Path, scattering_type: ScatteringType) -> PESWrapper | DiabaticPESWrapper:
    """Build a Fortran PES wrapper from the input file's PES table."""
    values = section(config, "pes")
    pes_dir = resolve_path(base, values.get("path", "."))
    sources = values.get("sources", ["interaction-PES.f"])
    if isinstance(sources, str):
        sources = [sources]
    source_paths = [resolve_path(pes_dir, source) for source in sources]
    wrapper = resolve_path(pes_dir, values.get("wrapper", "pyticc_wrapper.f90"))
    workdir = resolve_path(pes_dir, values.get("workdir", "."))
    lapack = values.get("lapack", False)
    if scattering_type is ScatteringType.ATOM_DIATOM_DIABATIC:
        return load_fortran_diabatic_pes(
            source_paths,
            wrapper,
            n_state=int(values.get("n_state", 2)),
            workdir=workdir,
            lapack=lapack,
        )
    return load_fortran_pes(source_paths, wrapper, workdir=workdir, lapack=lapack)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _resolve_scattering_type(config: TomlTable) -> ScatteringType:
    """Select the TOML calculation path from geometry and enabled physics."""
    type_value = str(config.get("type", "auto"))
    fine_structure = config.get("fine_structure")
    if fine_structure is not None and not isinstance(fine_structure, dict):
        message = "Input section 'fine_structure' must be a TOML table"
        logger.error(message)
        raise ValueError(message)
    fs_values = fine_structure if isinstance(fine_structure, dict) else {}
    if "atom_X" in fs_values or "atom_Y" in fs_values:
        if not all(k in fs_values for k in ("atom_X", "atom_Y")):
            raise ValueError("Atomic input requires both fine_structure.atom_X and atom_Y")
        if "electric" in config or any(k in fs_values for k in ("atom", "diatom", "diatom_X", "diatom_Y")):
            raise ValueError("Do not mix atomic fine structure with external fields or molecular sections")
        if type_value not in ("auto", "A+B", ScatteringType.ATOM_ATOM_FINE_STRUCTURE.value):
            raise ValueError("TOML type conflicts with atomic fine-structure sections")
        return ScatteringType.ATOM_ATOM_FINE_STRUCTURE
    atom_fs = "atom" in fs_values
    diatom_fs = "diatom" in fs_values
    diatom_X_fs = "diatom_X" in fs_values
    diatom_Y_fs = "diatom_Y" in fs_values
    electric = "electric" in config

    if electric and (atom_fs or diatom_fs or diatom_X_fs or diatom_Y_fs):
        message = "Combined external-field and fine-structure input is not implemented"
        logger.error(message)
        raise ValueError(message)
    if atom_fs != diatom_fs:
        message = "TOML fine-structure input currently requires both [fine_structure.atom] and [fine_structure.diatom]"
        logger.error(message)
        raise ValueError(message)
    if diatom_X_fs != diatom_Y_fs:
        message = "TOML 2+2 fine-structure input requires both [fine_structure.diatom_X] and [fine_structure.diatom_Y]"
        logger.error(message)
        raise ValueError(message)
    if (atom_fs or diatom_fs) and (diatom_X_fs or diatom_Y_fs):
        message = "Do not mix A+BC and AB+CD fine-structure monomer sections"
        logger.error(message)
        raise ValueError(message)

    if atom_fs and diatom_fs:
        if type_value not in ("auto", ScatteringType.ATOM_DIATOM.value, ScatteringType.ATOM_DIATOM_BOTH_FS.value):
            message = f"TOML type {type_value!r} conflicts with A+BC fine-structure sections"
            logger.error(message)
            raise ValueError(message)
        return ScatteringType.ATOM_DIATOM_BOTH_FS
    if diatom_X_fs and diatom_Y_fs:
        if type_value not in ("auto", ScatteringType.DIATOM_DIATOM.value, ScatteringType.DIATOM_DIATOM_FINE_STRUCTURE.value):
            message = f"TOML type {type_value!r} conflicts with AB+CD fine-structure sections"
            logger.error(message)
            raise ValueError(message)
        return ScatteringType.DIATOM_DIATOM_FINE_STRUCTURE

    if type_value in ("auto", ScatteringType.ATOM_DIATOM.value):
        if electric:
            return ScatteringType.ATOM_DIATOM_ELECTRIC
        if type_value == ScatteringType.ATOM_DIATOM.value or ("atom" in config and "diatom" in config):
            return ScatteringType.ATOM_DIATOM
        if "diatom_X" in config and "diatom_Y" in config:
            return ScatteringType.DIATOM_DIATOM
        message = "Cannot infer TOML scattering geometry; provide atom/diatom or diatom_X/diatom_Y"
        logger.error(message)
        raise ValueError(message)

    try:
        selected = ScatteringType(type_value)
    except ValueError as error:
        supported = ", ".join(value.value for value in ScatteringType)
        message = f"Unsupported TOML type {type_value!r}; supported: auto, {supported}"
        logger.error(message)
        raise ValueError(message) from error
    supported_types = (
        ScatteringType.ATOM_ATOM_FINE_STRUCTURE,
        ScatteringType.ATOM_DIATOM,
        ScatteringType.ATOM_DIATOM_ELECTRIC,
        ScatteringType.ATOM_DIATOM_BOTH_FS,
        ScatteringType.ATOM_DIATOM_DIABATIC,
        ScatteringType.DIATOM_DIATOM,
        ScatteringType.DIATOM_DIATOM_FINE_STRUCTURE,
    )
    if selected not in supported_types:
        supported = ", ".join(value.value for value in supported_types)
        message = f"Unsupported TOML type {type_value!r}; supported: auto, {supported}"
        logger.error(message)
        raise ValueError(message)
    if selected in (ScatteringType.ATOM_ATOM_FINE_STRUCTURE, ScatteringType.ATOM_DIATOM_BOTH_FS, ScatteringType.DIATOM_DIATOM_FINE_STRUCTURE):
        message = f"{selected.value} requires its fine-structure monomer sections"
        logger.error(message)
        raise ValueError(message)
    return selected


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def run(
    source: str | Path,
    *,
    pes: PESWrapper | DiabaticPESWrapper | SpinResolvedAtomAtomPES | SpinResolvedAtomDiatomPES | SpinResolvedDiatomDiatomPES | None = None,
) -> ScatteringResult | CoupledStatesResult:
    """Run a scattering calculation from a TOML input file."""
    wall_start = perf_counter()
    cpu_start = process_time()
    input_path = Path(source).expanduser().resolve()
    try:
        with input_path.open("rb") as file:
            config = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        message = f"Failed to read PyTICC input {input_path}: {error}"
        logger.error(message)
        raise ValueError(message) from error

    scattering_type = _resolve_scattering_type(config)

    if pes is None and scattering_type in (
        ScatteringType.ATOM_ATOM_FINE_STRUCTURE,
        ScatteringType.ATOM_DIATOM_BOTH_FS,
        ScatteringType.DIATOM_DIATOM_FINE_STRUCTURE,
    ):
        message = "Fine-structure TOML input requires a spin-resolved PES object through pes=..."
        logger.error(message)
        raise ValueError(message)
    potential = _load_pes(config, input_path.parent, scattering_type) if pes is None else pes
    if scattering_type is ScatteringType.ATOM_DIATOM and isinstance(potential, PESWrapper):
        result = atom_diatom.run(config, input_path.parent, potential)
    elif scattering_type is ScatteringType.ATOM_DIATOM_ELECTRIC and isinstance(potential, PESWrapper):
        result = atom_diatom.run_electric(config, input_path.parent, potential)
    elif scattering_type is ScatteringType.DIATOM_DIATOM and isinstance(potential, PESWrapper):
        result = diatom_diatom.run(config, input_path.parent, potential)
    elif scattering_type is ScatteringType.ATOM_DIATOM_DIABATIC and isinstance(potential, DiabaticPESWrapper):
        result = diabatic.run(config, input_path.parent, potential)
    elif scattering_type is ScatteringType.ATOM_ATOM_FINE_STRUCTURE and isinstance(potential, SpinResolvedAtomAtomPES):
        result = fine_structure_atom_atom.run(config, input_path.parent, potential)
    elif scattering_type is ScatteringType.ATOM_DIATOM_BOTH_FS and isinstance(potential, SpinResolvedAtomDiatomPES):
        result = fine_structure_atom_diatom.run(config, input_path.parent, potential)
    elif scattering_type is ScatteringType.DIATOM_DIATOM_FINE_STRUCTURE and isinstance(potential, SpinResolvedDiatomDiatomPES):
        result = fine_structure_diatom_diatom.run(config, input_path.parent, potential)
    else:
        if scattering_type is ScatteringType.ATOM_DIATOM_DIABATIC:
            expected = DiabaticPESWrapper
        elif scattering_type is ScatteringType.ATOM_DIATOM_BOTH_FS:
            expected = SpinResolvedAtomDiatomPES
        elif scattering_type is ScatteringType.DIATOM_DIATOM_FINE_STRUCTURE:
            expected = SpinResolvedDiatomDiatomPES
        else:
            expected = PESWrapper
        message = f"Calculation type {scattering_type.value!r} requires {expected.__name__}"
        logger.error(message)
        raise TypeError(message)

    total_timing = Timing(wall_seconds=perf_counter() - wall_start, cpu_seconds=process_time() - cpu_start)
    logger.info(f"Calculation complete: total {total_timing}")
    return result


# ----------------------------------------------------------------------------------------
