import atexit
import importlib.util
import multiprocessing as mp
import os
import sys
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.pool import Pool
from pathlib import Path
from threading import RLock
from types import ModuleType

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.pes.wrapper import MonomerPES, PESWrapper


@dataclass(frozen=True)
class _FortranPESSpec:
    """Compiled module identity and optional runtime-data directory."""

    module_name: str
    extension: Path
    workdir: Path | None


_MODULE: ModuleType | None = None
_COORDINATES: NDArray[np.float64] | None = None
_FORTRAN_LOCK = RLock()


# ----------------------------------------------------------------------------------------
def create_pes_wrapper(module_name: str, extension: Path, workdir: Path | None, processes: int) -> PESWrapper:
    """
    Load a compiled extension and expose its scalar, batched, and monomer PES calls.

    Inputs:
        module_name: str - Python name embedded in the compiled extension
        extension: Path - platform-specific compiled extension path
        workdir: Path | None - directory containing PES runtime data files
        processes: int - persistent processes used for batched radial evaluation

    Returns:
        pes: PESWrapper - compiled monomer and interaction potential interfaces
    """
    spec = _FortranPESSpec(module_name, extension, workdir)
    module = _load_module(module_name, extension)
    interaction = _make_interaction(module, workdir)
    executor = _FortranPESExecutor(spec, processes, interaction)

    def interaction_many(R: NDArray[np.float64], coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate R with shape (n_R,), returning potential values of shape (n_R, n_grid)."""
        return executor.evaluate(R, coordinates)

    def monomer(name: str) -> MonomerPES | None:
        """Wrap an optional compiled monomer routine by its exported name."""
        routine: Callable[..., object] | None = getattr(module, name, None)
        if routine is None:
            return None

        def potential(r: NDArray[np.float64]) -> NDArray[np.float64]:
            """Evaluate monomer coordinates with shape (n_grid,), returning shape (n_grid,)."""
            with _in_workdir(workdir):
                return np.asarray(routine(np.ascontiguousarray(r)), dtype=np.float64)

        return potential

    return PESWrapper(
        interaction=interaction,
        interaction_many=interaction_many,
        monomer_X=monomer("pyticc_monomer_x_grid"),
        monomer_Y=monomer("pyticc_monomer_y_grid"),
        _close=executor.close,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _load_module(module_name: str, extension: Path) -> ModuleType:
    """Import a compiled PES extension, reusing an already loaded module when possible."""
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


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@contextmanager
def _in_workdir(workdir: Path | None):
    """Temporarily enter the PES runtime directory under a process-local lock."""
    with _FORTRAN_LOCK:
        previous = Path.cwd()
        try:
            if workdir is not None:
                os.chdir(workdir)
            yield
        finally:
            if workdir is not None:
                os.chdir(previous)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _make_interaction(module: ModuleType, workdir: Path | None) -> Callable[[float, NDArray[np.float64]], NDArray[np.float64]]:
    """Wrap a routine mapping coordinates of shape (n_coordinate, n_grid) to shape (n_grid,)."""

    def interaction(R: float, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate one R for coordinates of shape (n_coordinate, n_grid), returning (n_grid,)."""
        with _in_workdir(workdir):
            return np.asarray(module.pyticc_interaction_grid(R, np.asfortranarray(coordinates)), dtype=np.float64)

    return interaction


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _initialize_worker(spec: _FortranPESSpec, coordinates: NDArray[np.float64]) -> None:
    """Load an isolated PES module and coordinates with shape (n_coordinate, n_grid)."""
    global _MODULE, _COORDINATES

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if spec.workdir is not None:
        os.chdir(spec.workdir)
    _MODULE = _load_module(spec.module_name, spec.extension)
    _COORDINATES = np.asfortranarray(coordinates)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _evaluate_worker(RR: float) -> NDArray[np.float64]:
    """Evaluate one radial point in a prepared worker, returning shape (n_grid,)."""
    if _MODULE is None or _COORDINATES is None:
        raise RuntimeError("Fortran PES worker is not initialized")
    return np.asarray(_MODULE.pyticc_interaction_grid(RR, _COORDINATES), dtype=np.float64)


# ----------------------------------------------------------------------------------------
class _FortranPESExecutor:
    """Keep spawned Fortran PES workers alive across radial windows."""

    def __init__(
        self,
        spec: _FortranPESSpec,
        processes: int,
        interaction: Callable[[float, NDArray[np.float64]], NDArray[np.float64]],
    ) -> None:
        self.spec = spec
        self.processes = processes
        self.interaction = interaction
        self._pool: Pool | None = None
        self._coordinates: NDArray[np.float64] | None = None
        self._lock = RLock()
        atexit.register(self.close)

    def _close_unlocked(self, terminate: bool = False) -> None:
        """Shut down the active pool while the caller holds the executor lock."""
        if self._pool is None:
            return
        if terminate:
            self._pool.terminate()
        else:
            self._pool.close()
        self._pool.join()
        self._pool = None
        self._coordinates = None

    def _ensure_pool(self, coordinates: NDArray[np.float64]) -> Pool:
        """Return a pool initialized for coordinates with shape (n_coordinate, n_grid)."""
        coordinates_F = np.asfortranarray(coordinates)
        same_coordinates = (
            self._coordinates is not None and self._coordinates.shape == coordinates_F.shape and np.array_equal(self._coordinates, coordinates_F)
        )
        if self._pool is None or not same_coordinates:
            self._close_unlocked()
            self._coordinates = coordinates_F.copy(order="F")
            context = mp.get_context("spawn")
            self._pool = context.Pool(
                processes=self.processes,
                initializer=_initialize_worker,
                initargs=(self.spec, self._coordinates),
            )
        return self._pool

    def evaluate(self, R: NDArray[np.float64], coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate R with shape (n_R,), returning potential values with shape (n_R, n_grid)."""
        radial_points = np.asarray(R, dtype=np.float64)
        if radial_points.size == 0:
            return np.empty((0, coordinates.shape[1]), dtype=np.float64)
        if self.processes == 1:
            return np.stack([self.interaction(float(RR), coordinates) for RR in radial_points])

        with self._lock:
            pool = self._ensure_pool(coordinates)
            chunksize = max(1, radial_points.size // (4 * self.processes))
            try:
                values = pool.map(_evaluate_worker, radial_points, chunksize=chunksize)
            except BaseException:
                self._close_unlocked(terminate=True)
                raise
        return np.stack(values)

    def close(self) -> None:
        """Close and join persistent worker processes; repeated calls are safe."""
        with self._lock:
            self._close_unlocked()
