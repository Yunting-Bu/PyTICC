import tomllib
from collections.abc import Sequence
from pathlib import Path

from loguru import logger

from pyticc.pes.adiabatic import PESWrapper
from pyticc.pes.diabatic import DiabaticPESWrapper
from pyticc.pes.fortran.compiler import prepare_diabatic_extension, prepare_extension, prepare_lambda_extension, prepare_total_extension
from pyticc.pes.fortran.executor import create_diabatic_pes_wrapper, create_lambda_pes, create_pes_wrapper, create_total_pes
from pyticc.pes.lambda_pes import LambdaPES
from pyticc.pes.total import TotalPES


# ----------------------------------------------------------------------------------------
def load_fortran_pes(
    sources: Sequence[str | Path] | str | Path,
    wrapper: str | Path | None = None,
    *,
    workdir: str | Path | None = None,
    lapack: bool = False,
) -> PESWrapper:
    """
    Compile or load fixed-interface Fortran potential-energy surfaces.

    ``sources`` can also be a short TOML file. Relative paths in TOML are
    resolved from the directory containing that TOML file.

    Inputs:
        sources: Sequence[str | Path] | str | Path - Fortran sources or a TOML file
        wrapper: str | Path | None - source implementing the PyTICC grid routines
        workdir: str | Path | None - directory containing PES runtime data files
        lapack: bool - whether the PES requires LAPACK

    Returns:
        pes: PESWrapper - compiled monomer and interaction potential interfaces
    """

    requested_lapack = _require_lapack(lapack)
    source_paths, wrapper_path, runtime_dir, configured_lapack = _resolve_inputs(sources, wrapper, workdir)
    module_name, extension = prepare_extension(source_paths, wrapper_path, lapack=requested_lapack or configured_lapack)
    return create_pes_wrapper(module_name, extension, runtime_dir)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def load_fortran_diabatic_pes(
    sources: Sequence[str | Path] | str | Path,
    wrapper: str | Path | None = None,
    *,
    n_state: int = 2,
    workdir: str | Path | None = None,
    lapack: bool = False,
) -> DiabaticPESWrapper:
    """
    Compile or load fixed-interface Fortran diabatic potential-energy matrices.

    The Fortran wrapper receives PyTICC coordinates rather than native PES
    coordinates. For atom-diatom calculations these are RR plus coordinate rows
    (r, theta) in bohr and radians; the wrapper owns any conversion to bond lengths
    or other PES-specific coordinates.

    Inputs:
        sources: Sequence[str | Path] | str | Path - Fortran sources or a TOML file
        wrapper: str | Path | None - source implementing the diabatic PyTICC grid routines
        n_state: int - number of diabatic electronic states
        workdir: str | Path | None - directory containing PES runtime data files
        lapack: bool - whether the PES requires LAPACK

    Returns:
        pes: DiabaticPESWrapper - compiled monomer potentials and diabatic interaction matrix
    """
    if n_state < 1:
        message = f"n_state must be positive, but got {n_state}"
        logger.error(message)
        raise ValueError(message)
    requested_lapack = _require_lapack(lapack)
    source_paths, wrapper_path, runtime_dir, configured_lapack = _resolve_inputs(sources, wrapper, workdir)
    module_name, extension = prepare_diabatic_extension(source_paths, wrapper_path, lapack=requested_lapack or configured_lapack)
    return create_diabatic_pes_wrapper(module_name, extension, runtime_dir, n_state)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def load_fortran_lambda_pes(
    sources: Sequence[str | Path] | str | Path,
    wrapper: str | Path | None = None,
    *,
    workdir: str | Path | None = None,
    lapack: bool = False,
) -> LambdaPES:
    """
    Compile or load a Fortran signed-Lambda interaction PES.

    The wrapper must define ``pyticc_lambda_grid`` returning ``(V_sum,V_dif)``
    in Hartree. PyTICC supplies R and internal coordinates in bohr and radians.

    Inputs:
        sources: Sequence[str | Path] | str | Path - Fortran sources or TOML file
        wrapper: str | Path | None - source implementing the PyTICC grid routine
        workdir: str | Path | None - PES runtime-data directory
        lapack: bool - whether the native PES requires LAPACK

    Returns:
        pes: LambdaPES - validated two-component interaction interface
    """
    requested_lapack = _require_lapack(lapack)
    source_paths, wrapper_path, runtime_dir, configured_lapack = _resolve_inputs(sources, wrapper, workdir)
    module_name, extension = prepare_lambda_extension(source_paths, wrapper_path, lapack=requested_lapack or configured_lapack)
    return create_lambda_pes(module_name, extension, runtime_dir)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def load_fortran_total_pes(
    sources: Sequence[str | Path] | str | Path,
    wrapper: str | Path | None = None,
    *,
    workdir: str | Path | None = None,
    lapack: bool = False,
) -> TotalPES:
    """
    Compile or load a Fortran scalar adiabatic total PES.

    The wrapper must define ``pyticc_total_grid(bonds,V,n_grid)``. PyTICC passes
    physical bond lengths in bohr with leading order ``(r_AB,r_BC,r_CA)``; the
    wrapper must return total energies in Hartree without changing their zero.

    Inputs:
        sources: Sequence[str | Path] | str | Path - Fortran sources or a TOML file
        wrapper: str | Path | None - source implementing ``pyticc_total_grid``
        workdir: str | Path | None - directory containing PES runtime data files
        lapack: bool - whether the PES requires LAPACK

    Returns:
        pes: TotalPES - validated total potential-energy surface
    """
    requested_lapack = _require_lapack(lapack)
    source_paths, wrapper_path, runtime_dir, configured_lapack = _resolve_inputs(sources, wrapper, workdir)
    module_name, extension = prepare_total_extension(source_paths, wrapper_path, lapack=requested_lapack or configured_lapack)
    return create_total_pes(module_name, extension, runtime_dir)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _resolve_inputs(
    sources: Sequence[str | Path] | str | Path,
    wrapper: str | Path | None,
    workdir: str | Path | None,
) -> tuple[tuple[Path, ...], Path, Path | None, bool]:
    """Resolve direct or TOML-configured PES sources, wrapper, and runtime directory."""
    base = Path.cwd()
    source_values: Sequence[str | Path]
    configured_lapack = False

    if wrapper is None:
        if not isinstance(sources, str | Path):
            message = "Fortran PES requires a wrapper when sources are passed directly"
            logger.error(message)
            raise ValueError(message)
        config = Path(sources).expanduser().resolve()
        base = config.parent
        with config.open("rb") as file:
            values = tomllib.load(file)
        source_values = values.get("sources", ())
        wrapper = values.get("wrapper")
        configured_lapack = _require_lapack(values.get("lapack", False))
        if workdir is None:
            workdir = values.get("workdir")
    else:
        source_values = (sources,) if isinstance(sources, str | Path) else sources

    if not source_values or wrapper is None:
        message = "Fortran PES requires sources and wrapper"
        logger.error(message)
        raise ValueError(message)

    source_paths = tuple(_resolve_path(path, base) for path in source_values)
    wrapper_path = _resolve_path(wrapper, base)
    for path in (*source_paths, wrapper_path):
        if not path.is_file():
            message = f"Fortran PES source does not exist: {path}"
            logger.error(message)
            raise FileNotFoundError(message)

    runtime_dir = None if workdir is None else _resolve_path(workdir, base)
    if runtime_dir is not None and not runtime_dir.is_dir():
        message = f"Fortran PES workdir does not exist: {runtime_dir}"
        logger.error(message)
        raise FileNotFoundError(message)
    return source_paths, wrapper_path, runtime_dir, configured_lapack


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _resolve_path(path: str | Path, base: Path) -> Path:
    """Resolve an absolute or base-relative filesystem path."""
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (base / value).resolve()


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _require_lapack(value: object) -> bool:
    """Return a validated LAPACK switch."""
    if not isinstance(value, bool):
        message = f"lapack must be a boolean, but got {value!r}"
        logger.error(message)
        raise ValueError(message)
    return value


# ----------------------------------------------------------------------------------------
