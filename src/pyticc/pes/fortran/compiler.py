import hashlib
import importlib.machinery
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from loguru import logger

_ROUTINES = ("pyticc_interaction_grid", "pyticc_monomer_x_grid", "pyticc_monomer_y_grid")


# ----------------------------------------------------------------------------------------
def prepare_extension(sources: tuple[Path, ...], wrapper: Path) -> tuple[str, Path]:
    """
    Return a cached or newly compiled Fortran PES extension.

    Inputs:
        sources: tuple[Path, ...] - Fortran PES source files
        wrapper: Path - source implementing the PyTICC grid routines

    Returns:
        module_name: str - hash-qualified Python extension name
        extension: Path - compiled platform-specific extension path
    """
    routines = _find_routines(wrapper)
    module_name = f"pyticc_pes_{_source_digest(sources, wrapper)[:12]}"
    build_dir = _cache_dir() / module_name
    extension = _find_extension(build_dir, module_name)
    if extension is None:
        extension = _compile_extension(build_dir, module_name, sources, wrapper, routines, _require_build_tools())
    return module_name, extension


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _cache_dir() -> Path:
    """Return the configured or default compiled-PES cache directory."""
    configured = os.environ.get("PYTICC_CACHE_DIR")
    return Path(configured).expanduser() if configured is not None else Path.home() / ".cache" / "pyticc" / "pes"


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _source_digest(sources: tuple[Path, ...], wrapper: Path) -> str:
    """Hash PES sources and the active Python/NumPy runtime for cache invalidation."""
    digest = hashlib.sha256()
    for path in (*sources, wrapper):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(sys.version.encode())
    digest.update(np.__version__.encode())
    return digest.hexdigest()


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _build_tools() -> tuple[bool, str | None, str | None, str | None]:
    """Report availability of f2py, Meson, Ninja, and a Fortran compiler."""
    f2py = (
        subprocess.run(
            [sys.executable, "-m", "numpy.f2py", "-v"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    executable_dir = str(Path(sys.executable).parent)
    meson = shutil.which("meson") or shutil.which("meson", path=executable_dir)
    ninja = shutil.which("ninja") or shutil.which("ninja", path=executable_dir)

    compiler = None
    selected_compiler = os.environ.get("FC")
    if selected_compiler is not None:
        compiler = shutil.which(selected_compiler)
    if compiler is None:
        for name in ("gfortran", "flang-new", "ifx", "ifort", "nvfortran"):
            compiler = shutil.which(name)
            if compiler is not None:
                break
    return f2py, meson, ninja, compiler


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _require_build_tools() -> str:
    """Return the compiler path or raise a diagnostic listing missing build tools."""
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


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _find_routines(wrapper: Path) -> set[str]:
    """Detect supported PyTICC wrapper subroutines and require the interaction entry point."""
    source = wrapper.read_text(errors="ignore")
    routines = {name for name in _ROUTINES if re.search(rf"^\s*subroutine\s+{name}\b", source, re.IGNORECASE | re.MULTILINE)}
    if "pyticc_interaction_grid" not in routines:
        message = f"Fortran wrapper must define pyticc_interaction_grid: {wrapper}"
        logger.error(message)
        raise ValueError(message)
    return routines


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _f2py_signature(module_name: str, routines: set[str]) -> str:
    """Generate the explicit f2py signature for the routines exposed by a wrapper."""
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


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _compile_extension(
    build_dir: Path,
    module_name: str,
    sources: tuple[Path, ...],
    wrapper: Path,
    routines: set[str],
    compiler: str,
) -> Path:
    """Compile a Fortran PES extension with f2py/Meson and return its shared library."""
    build_dir.mkdir(parents=True, exist_ok=True)
    signature = build_dir / f"{module_name}.pyf"
    signature.write_text(_f2py_signature(module_name, routines))
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


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _find_extension(build_dir: Path, module_name: str) -> Path | None:
    """Find a platform-specific compiled extension in one build directory."""
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        extension = build_dir / f"{module_name}{suffix}"
        if extension.is_file():
            return extension
    candidates = sorted(build_dir.glob(f"{module_name}*.so")) + sorted(build_dir.glob(f"{module_name}*.pyd"))
    return candidates[0] if candidates else None
