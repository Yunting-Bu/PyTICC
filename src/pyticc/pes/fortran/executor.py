import importlib.util
import multiprocessing as mp
import os
import sys
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.pool import Pool
from pathlib import Path
from threading import RLock
from time import perf_counter
from types import ModuleType

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.pes.adiabatic import MonomerPES, PESWrapper
from pyticc.pes.diabatic import DiabaticPESWrapper
from pyticc.pes.lambda_pes import LambdaPES
from pyticc.pes.total import TotalPES


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _FortranPESSpec:
    """Compiled module identity and optional runtime-data directory."""

    module_name: str
    extension: Path
    workdir: Path | None
    interaction_routine: str = "pyticc_interaction_grid"
    interaction_args: tuple[int, ...] = ()


# ----------------------------------------------------------------------------------------


_MODULE: ModuleType | None = None
_COORDINATES: NDArray[np.float64] | None = None
_INTERACTION_ROUTINE = "pyticc_interaction_grid"
_INTERACTION_ARGS: tuple[int, ...] = ()
_FORTRAN_LOCK = RLock()


# ----------------------------------------------------------------------------------------
def create_pes_wrapper(module_name: str, extension: Path, workdir: Path | None) -> PESWrapper:
    """
    Load a compiled extension and expose its scalar, batched, and monomer PES calls.

    Inputs:
        module_name: str - Python name embedded in the compiled extension
        extension: Path - platform-specific compiled extension path
        workdir: Path | None - directory containing PES runtime data files
    Returns:
        pes: PESWrapper - compiled monomer and interaction potential interfaces
    """
    spec = _FortranPESSpec(module_name, extension, workdir)
    module = _load_module(module_name, extension)
    interaction = _make_interaction(module, workdir, spec.interaction_routine, spec.interaction_args)

    def interaction_many(R: NDArray[np.float64], coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate R with shape (n_R,), returning potential values of shape (n_R, n_grid)."""
        return interaction_many_processes(R, coordinates, 1)

    def interaction_many_processes(
        R: NDArray[np.float64],
        coordinates: NDArray[np.float64],
        processes: int,
    ) -> NDArray[np.float64]:
        """Evaluate one radial batch with temporary Fortran worker processes."""
        return _evaluate_many_processes(spec, processes, interaction, R, coordinates)

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
        _interaction_many_processes=interaction_many_processes,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def create_diabatic_pes_wrapper(
    module_name: str,
    extension: Path,
    workdir: Path | None,
    n_state: int,
) -> DiabaticPESWrapper:
    """Load a compiled extension and expose its diabatic monomer and interaction matrices."""
    routine_name = "pyticc_diabatic_interaction_grid"
    spec = _FortranPESSpec(module_name, extension, workdir, routine_name, (n_state,))
    module = _load_module(module_name, extension)
    interaction = _make_interaction(module, workdir, routine_name, (n_state,))
    monomer_routine: Callable[..., object] = module.pyticc_diabatic_monomer_grid

    def interaction_many(R: NDArray[np.float64], coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate a radial batch, returning shape ``(n_R, n_grid, n_state, n_state)``."""
        return interaction_many_processes(R, coordinates, 1)

    def interaction_many_processes(
        R: NDArray[np.float64],
        coordinates: NDArray[np.float64],
        processes: int,
    ) -> NDArray[np.float64]:
        """Evaluate one radial batch with temporary Fortran worker processes."""
        return _evaluate_many_processes(spec, processes, interaction, R, coordinates)

    def monomer(r: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate all monomer states, returning shape ``(n_grid, n_state)``."""
        with _in_workdir(workdir):
            return np.asarray(monomer_routine(np.ascontiguousarray(r), n_state), dtype=np.float64)

    return DiabaticPESWrapper(
        n_state=n_state,
        monomer=monomer,
        interaction=interaction,
        interaction_many=interaction_many,
        _interaction_many_processes=interaction_many_processes,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def create_lambda_pes(module_name: str, extension: Path, workdir: Path | None) -> LambdaPES:
    """Load a compiled extension exposing V_sum and V_dif interaction grids."""
    routine_name = "pyticc_lambda_grid"
    spec = _FortranPESSpec(module_name, extension, workdir, routine_name)
    module = _load_module(module_name, extension)
    interaction = _make_interaction(module, workdir, routine_name)
    monomer_routine: Callable[..., object] | None = getattr(module, "pyticc_monomer_y_grid", None)

    def interaction_many(R: NDArray[np.float64], coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate a radial batch, returning shape (n_R,n_grid,2)."""
        return interaction_many_processes(R, coordinates, 1)

    def interaction_many_processes(
        R: NDArray[np.float64],
        coordinates: NDArray[np.float64],
        processes: int,
    ) -> NDArray[np.float64]:
        """Evaluate one radial batch with temporary Fortran worker processes."""
        return _evaluate_many_processes(spec, processes, interaction, R, coordinates)

    monomer_Y: MonomerPES | None = None
    if monomer_routine is not None:

        def evaluate_monomer_Y(r: NDArray[np.float64]) -> NDArray[np.float64]:
            with _in_workdir(workdir):
                return np.asarray(monomer_routine(np.ascontiguousarray(r)), dtype=np.float64)

        monomer_Y = evaluate_monomer_Y

    return LambdaPES(
        interaction=interaction,
        interaction_many=interaction_many,
        monomer_Y=monomer_Y,
        _interaction_many_processes=interaction_many_processes,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def create_total_pes(module_name: str, extension: Path, workdir: Path | None) -> TotalPES:
    """
    Load a compiled extension and expose its scalar total potential.

    Inputs:
        module_name: str - Python name embedded in the compiled extension
        extension: Path - platform-specific compiled extension path
        workdir: Path | None - directory containing PES runtime data files

    Returns:
        pes: TotalPES - total PES accepting ``(r_AB,r_BC,r_CA)`` bond arrays
            in bohr and returning energies in Hartree
    """
    module = _load_module(module_name, extension)
    routine: Callable[..., object] = module.pyticc_total_grid

    def potential(bonds: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate bonds with shape ``(3,n_grid)``, returning shape ``(n_grid,)``."""
        with _in_workdir(workdir):
            return np.asarray(routine(np.asfortranarray(bonds)), dtype=np.float64)

    return TotalPES(potential=potential)


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
def _make_interaction(
    module: ModuleType,
    workdir: Path | None,
    routine_name: str = "pyticc_interaction_grid",
    routine_args: tuple[int, ...] = (),
) -> Callable[[float, NDArray[np.float64]], NDArray[np.float64]]:
    """Wrap a scalar-R Fortran routine while preserving all PES output axes."""
    routine: Callable[..., object] = getattr(module, routine_name)

    def interaction(R: float, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate one R for coordinates of shape ``(n_coordinate, n_grid)``."""
        with _in_workdir(workdir):
            return np.asarray(routine(R, np.asfortranarray(coordinates), *routine_args), dtype=np.float64)

    return interaction


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _initialize_worker(spec: _FortranPESSpec, coordinates: NDArray[np.float64]) -> None:
    """Load an isolated PES module and coordinates with shape (n_coordinate, n_grid)."""
    global _MODULE, _COORDINATES, _INTERACTION_ROUTINE, _INTERACTION_ARGS

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if spec.workdir is not None:
        os.chdir(spec.workdir)
    _MODULE = _load_module(spec.module_name, spec.extension)
    _COORDINATES = np.asfortranarray(coordinates)
    _INTERACTION_ROUTINE = spec.interaction_routine
    _INTERACTION_ARGS = spec.interaction_args


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _evaluate_worker(RR: float) -> NDArray[np.float64]:
    """Evaluate one radial point in a prepared worker."""
    if _MODULE is None or _COORDINATES is None:
        raise RuntimeError("Fortran PES worker is not initialized")
    routine: Callable[..., object] = getattr(_MODULE, _INTERACTION_ROUTINE)
    return np.asarray(routine(RR, _COORDINATES, *_INTERACTION_ARGS), dtype=np.float64)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
class _FortranPESExecutor:
    """Manage spawned Fortran PES workers for one radial-batch evaluation."""

    def __init__(
        self,
        spec: _FortranPESSpec,
        processes: int,
        interaction: Callable[[float, NDArray[np.float64]], NDArray[np.float64]],
    ) -> None:
        if not isinstance(processes, int) or isinstance(processes, bool) or processes < 1:
            message = f"processes must be a positive integer, but got {processes!r}"
            logger.error(message)
            raise ValueError(message)
        self.spec = spec
        self.processes = processes
        self.interaction = interaction
        self._pool: Pool | None = None
        self._coordinates: NDArray[np.float64] | None = None
        self._lock = RLock()

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
        """Evaluate R with shape ``(n_R,)``, preserving every trailing PES output axis."""
        radial_points = np.asarray(R, dtype=np.float64)
        if radial_points.size == 0:
            return np.empty((0, coordinates.shape[1]), dtype=np.float64)
        started = perf_counter()
        if self.processes == 1:
            values = (self.interaction(float(RR), coordinates) for RR in radial_points)
            return self._collect(values, radial_points, started)

        with self._lock:
            pool = self._ensure_pool(coordinates)
            chunksize = max(1, radial_points.size // (4 * self.processes))
            try:
                values = self._collect(pool.imap(_evaluate_worker, radial_points, chunksize=chunksize), radial_points, started)
            except BaseException:
                self._close_unlocked(terminate=True)
                raise
        return values

    @staticmethod
    def _collect(
        values: Iterable[NDArray[np.float64]],
        radial_points: NDArray[np.float64],
        started: float,
    ) -> NDArray[np.float64]:
        """Collect ordered radial results and log about ten progress updates."""
        iterator = iter(values)
        total = radial_points.size
        progress_interval = max(1, (total + 9) // 10)
        next_progress = progress_interval
        collected: list[NDArray[np.float64]] = []
        for completed, value in enumerate(iterator, start=1):
            collected.append(value)
            if completed == total or completed >= next_progress:
                logger.info(
                    f"Potential: {completed}/{total} radial points, R={radial_points[completed - 1]:.6f} bohr, wall={perf_counter() - started:.3f} s"
                )
                while next_progress <= completed:
                    next_progress += progress_interval
        return np.stack(collected)

    def close(self) -> None:
        """Close and join active worker processes; repeated calls are safe."""
        with self._lock:
            self._close_unlocked()


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _evaluate_many_processes(
    spec: _FortranPESSpec,
    processes: int,
    interaction: Callable[[float, NDArray[np.float64]], NDArray[np.float64]],
    R: NDArray[np.float64],
    coordinates: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate one radial batch and always release its temporary worker pool."""
    executor = _FortranPESExecutor(spec, processes, interaction)
    try:
        return executor.evaluate(R, coordinates)
    finally:
        executor.close()


# ----------------------------------------------------------------------------------------
