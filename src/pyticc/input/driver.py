import tomllib
from pathlib import Path
from time import perf_counter, process_time

from loguru import logger

import pyticc.input.atom_diatom as atom_diatom
import pyticc.input.diabatic as diabatic
import pyticc.input.diatom_diatom as diatom_diatom
from pyticc.input.common import TomlTable, required, resolve_path, section
from pyticc.pes.adiabatic import PESWrapper
from pyticc.pes.diabatic import DiabaticPESWrapper
from pyticc.pes.fortran import load_fortran_diabatic_pes, load_fortran_pes
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
def run(source: str | Path, *, pes: PESWrapper | DiabaticPESWrapper | None = None) -> ScatteringResult | CoupledStatesResult:
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

    type_value = str(required(config, "type"))
    try:
        scattering_type = ScatteringType(type_value)
    except ValueError as error:
        supported = ", ".join(
            value.value
            for value in (
                ScatteringType.ATOM_DIATOM,
                ScatteringType.ATOM_DIATOM_ELECTRIC,
                ScatteringType.ATOM_DIATOM_DIABATIC,
                ScatteringType.DIATOM_DIATOM,
            )
        )
        message = f"Unsupported TOML type {type_value!r}; supported: {supported}"
        logger.error(message)
        raise ValueError(message) from error

    potential = _load_pes(config, input_path.parent, scattering_type) if pes is None else pes
    if scattering_type is ScatteringType.ATOM_DIATOM and isinstance(potential, PESWrapper):
        result = atom_diatom.run(config, input_path.parent, potential)
    elif scattering_type is ScatteringType.ATOM_DIATOM_ELECTRIC and isinstance(potential, PESWrapper):
        result = atom_diatom.run_electric(config, input_path.parent, potential)
    elif scattering_type is ScatteringType.DIATOM_DIATOM and isinstance(potential, PESWrapper):
        result = diatom_diatom.run(config, input_path.parent, potential)
    elif scattering_type is ScatteringType.ATOM_DIATOM_DIABATIC and isinstance(potential, DiabaticPESWrapper):
        result = diabatic.run(config, input_path.parent, potential)
    else:
        expected = DiabaticPESWrapper if scattering_type is ScatteringType.ATOM_DIATOM_DIABATIC else PESWrapper
        message = f"Calculation type {scattering_type.value!r} requires {expected.__name__}"
        logger.error(message)
        raise TypeError(message)

    total_timing = Timing(wall_seconds=perf_counter() - wall_start, cpu_seconds=process_time() - cpu_start)
    logger.info(f"Calculation complete: total {total_timing}")
    return result


# ----------------------------------------------------------------------------------------
