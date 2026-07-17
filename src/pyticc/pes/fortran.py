import hashlib
import importlib.machinery
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from types import ModuleType

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.pes.wrapper import MonomerPES, PESWrapper

_ROUTINES = ("pyticc_interaction_grid", "pyticc_monomer_x_grid", "pyticc_monomer_y_grid")
_FORTRAN_LOCK = RLock()


# ----------------------------------------------------------------------------------------
def load_fortran_pes(
    sources: Sequence[str | Path] | str | Path,
    wrapper: str | Path | None = None,
    *,
    workdir: str | Path | None = None,
) -> PESWrapper:
    """
    Compile or load fixed-interface Fortran potential-energy surfaces.

    ``sources`` can also be a short TOML file. Relative paths in TOML are
    resolved from the directory containing that TOML file.

    Inputs:
        sources: Sequence[str | Path] | str | Path - Fortran sources or a TOML file
        wrapper: str | Path | None - source implementing the PyTICC grid routines
        workdir: str | Path | None - directory containing PES runtime data files

    Returns:
        pes: PESWrapper - compiled monomer and interaction potential interfaces
    """
    source_paths, wrapper_path, runtime_dir = _resolve_inputs(sources, wrapper, workdir)
    routines = _find_routines(wrapper_path)
    digest = _digest(source_paths, wrapper_path)
    module_name = f"pyticc_pes_{digest[:12]}"
    build_dir = _cache_dir() / module_name
    extension = _find_extension(build_dir, module_name)

    if extension is None:
        compiler = _require_build_tools()
        extension = _build(build_dir, module_name, source_paths, wrapper_path, routines, compiler)

    module = _load_module(module_name, extension)
    return _make_wrapper(module, runtime_dir)


def _resolve_inputs(
    sources: Sequence[str | Path] | str | Path,
    wrapper: str | Path | None,
    workdir: str | Path | None,
) -> tuple[tuple[Path, ...], Path, Path | None]:
    base = Path.cwd()
    source_values: Sequence[str | Path]

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
    return source_paths, wrapper_path, runtime_dir


def _resolve_path(path: str | Path, base: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _cache_dir() -> Path:
    configured = os.environ.get("PYTICC_CACHE_DIR")
    return Path(configured).expanduser() if configured is not None else Path.home() / ".cache" / "pyticc" / "pes"


def _digest(sources: tuple[Path, ...], wrapper: Path) -> str:
    digest = hashlib.sha256()
    for path in (*sources, wrapper):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(sys.version.encode())
    digest.update(np.__version__.encode())
    return digest.hexdigest()


def _find_executable(name: str) -> str | None:
    return shutil.which(name) or shutil.which(name, path=str(Path(sys.executable).parent))


def _find_compiler() -> str | None:
    FC = os.environ.get("FC")
    if FC is not None and shutil.which(FC) is not None:
        return shutil.which(FC)
    for name in ("gfortran", "flang-new", "ifx", "ifort", "nvfortran"):
        compiler = shutil.which(name)
        if compiler is not None:
            return compiler
    return None


def _build_tools() -> tuple[bool, str | None, str | None, str | None]:
    f2py = (
        subprocess.run(
            [sys.executable, "-m", "numpy.f2py", "-v"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    return f2py, _find_executable("meson"), _find_executable("ninja"), _find_compiler()


def _require_build_tools() -> str:
    f2py, meson, ninja, compiler = _build_tools()
    if f2py and meson and ninja and compiler:
        return compiler
    message = (
        "Cannot build Fortran PES because build tools are missing.\n"
        f"NumPy/f2py: {'available' if f2py else 'missing'}\n"
        f"Meson: {meson or 'missing'}\n"
        f"Ninja: {ninja or 'missing'}\n"
        f"Fortran compiler: {compiler or 'missing'}\n"
        "Install Meson and Ninja with `uv sync --extra fortran`; a system Fortran compiler is also required."
    )
    logger.error(message)
    raise RuntimeError(message)


def _find_routines(wrapper: Path) -> set[str]:
    source = wrapper.read_text(errors="ignore")
    routines = {name for name in _ROUTINES if re.search(rf"^\s*subroutine\s+{name}\b", source, re.IGNORECASE | re.MULTILINE)}
    if "pyticc_interaction_grid" not in routines:
        message = f"Fortran wrapper must define pyticc_interaction_grid: {wrapper}"
        logger.error(message)
        raise ValueError(message)
    return routines


def _signature(module_name: str, routines: set[str]) -> str:
    blocks = [
        """        subroutine pyticc_interaction_grid(RR, coordinates, V, n_coordinate, n_grid)
            real*8 intent(in) :: RR
            real*8 intent(in), dimension(n_coordinate, n_grid) :: coordinates
            real*8 intent(out), dimension(n_grid) :: V
            integer intent(hide), depend(coordinates) :: n_coordinate = shape(coordinates, 0)
            integer intent(hide), depend(coordinates) :: n_grid = shape(coordinates, 1)
        end subroutine pyticc_interaction_grid"""
    ]
    for name in ("pyticc_monomer_x_grid", "pyticc_monomer_y_grid"):
        if name in routines:
            blocks.append(
                f"""        subroutine {name}(r, V, n_grid)
            real*8 intent(in), dimension(n_grid) :: r
            real*8 intent(out), dimension(n_grid) :: V
            integer intent(hide), depend(r) :: n_grid = shape(r, 0)
        end subroutine {name}"""
            )
    return f"python module {module_name}\n    interface\n" + "\n".join(blocks) + f"\n    end interface\nend python module {module_name}\n"


def _build(build_dir: Path, module_name: str, sources: tuple[Path, ...], wrapper: Path, routines: set[str], compiler: str) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    signature = build_dir / f"{module_name}.pyf"
    signature.write_text(_signature(module_name, routines))
    command = [
        sys.executable,
        "-m",
        "numpy.f2py",
        "-c",
        str(signature),
        *(str(path) for path in sources),
        str(wrapper),
        "--backend",
        "meson",
        "--f77flags=-O3",
        "--f90flags=-O3",
    ]
    environment = os.environ.copy()
    environment["FC"] = compiler
    result = subprocess.run(command, cwd=build_dir, env=environment, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    build_log = build_dir / "build.log"
    build_log.write_text(result.stdout)
    extension = _find_extension(build_dir, module_name)
    if result.returncode != 0 or extension is None:
        message = f"Failed to build Fortran PES. Build log: {build_log}"
        logger.error(message)
        raise RuntimeError(message)
    return extension


def _find_extension(build_dir: Path, module_name: str) -> Path | None:
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        extension = build_dir / f"{module_name}{suffix}"
        if extension.is_file():
            return extension
    candidates = sorted(build_dir.glob(f"{module_name}*.so")) + sorted(build_dir.glob(f"{module_name}*.pyd"))
    return candidates[0] if candidates else None


def _load_module(module_name: str, extension: Path) -> ModuleType:
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, extension)
    if spec is None or spec.loader is None:
        message = f"Cannot load compiled Fortran PES extension: {extension}"
        logger.error(message)
        raise ImportError(message)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _in_workdir(workdir: Path | None):
    with _FORTRAN_LOCK:
        previous = Path.cwd()
        try:
            if workdir is not None:
                os.chdir(workdir)
            yield
        finally:
            if workdir is not None:
                os.chdir(previous)


def _make_wrapper(module: ModuleType, workdir: Path | None) -> PESWrapper:
    def interaction(R: float, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        with _in_workdir(workdir):
            return np.asarray(module.pyticc_interaction_grid(R, np.asfortranarray(coordinates)), dtype=np.float64)

    def monomer(name: str) -> MonomerPES | None:
        routine: Callable[..., object] | None = getattr(module, name, None)
        if routine is None:
            return None

        def potential(r: NDArray[np.float64]) -> NDArray[np.float64]:
            with _in_workdir(workdir):
                return np.asarray(routine(np.ascontiguousarray(r)), dtype=np.float64)

        return potential

    return PESWrapper(
        interaction=interaction,
        monomer_X=monomer("pyticc_monomer_x_grid"),
        monomer_Y=monomer("pyticc_monomer_y_grid"),
    )


# ----------------------------------------------------------------------------------------
