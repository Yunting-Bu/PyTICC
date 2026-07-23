import tomllib
from dataclasses import replace
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


# ----------------------------------------------------------------------------------------
def _load_pes(config: TomlTable, base: Path, calculation_type: str) -> PESWrapper | DiabaticPESWrapper:
    """Build a Fortran PES wrapper from the input file's PES table."""
    values = section(config, "pes")
    pes_dir = resolve_path(base, values.get("path", "."))
    sources = values.get("sources", ["interaction-PES.f"])
    if isinstance(sources, str):
        sources = [sources]
    source_paths = [resolve_path(pes_dir, source) for source in sources]
    wrapper = resolve_path(pes_dir, values.get("wrapper", "pyticc_wrapper.f90"))
    workdir = resolve_path(pes_dir, values.get("workdir", "."))
    processes = int(values.get("processes", 1))
    lapack = values.get("lapack", False)
    if calculation_type == "diabatic-atom-diatom":
        return load_fortran_diabatic_pes(
            source_paths,
            wrapper,
            n_state=int(values.get("n_state", 2)),
            workdir=workdir,
            processes=processes,
            lapack=lapack,
        )
    return load_fortran_pes(source_paths, wrapper, workdir=workdir, processes=processes, lapack=lapack)


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

    calculation_type = str(required(config, "type"))
    potential = _load_pes(config, input_path.parent, calculation_type) if pes is None else pes
    if calculation_type == "atom-diatom" and isinstance(potential, PESWrapper):
        result = atom_diatom.run(config, input_path.parent, potential)
    elif calculation_type == "diatom-diatom" and isinstance(potential, PESWrapper):
        result = diatom_diatom.run(config, input_path.parent, potential)
    elif calculation_type == "diabatic-atom-diatom" and isinstance(potential, DiabaticPESWrapper):
        result = diabatic.run(config, input_path.parent, potential)
    else:
        expected = DiabaticPESWrapper if calculation_type == "diabatic-atom-diatom" else PESWrapper
        if calculation_type in {"atom-diatom", "diatom-diatom", "diabatic-atom-diatom"}:
            message = f"Calculation type {calculation_type!r} requires {expected.__name__}"
            logger.error(message)
            raise TypeError(message)

        message = f"Unsupported calculation type {calculation_type!r}; supported: 'atom-diatom', 'diabatic-atom-diatom', 'diatom-diatom'"
        logger.error(message)
        raise ValueError(message)

    timing = Timing(wall_seconds=perf_counter() - wall_start, cpu_seconds=process_time() - cpu_start)
    logger.info(f"Calculation complete: {timing}")
    return replace(result, timing=timing)


# ----------------------------------------------------------------------------------------
