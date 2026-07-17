import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class _FortranPESSpec:
    module_name: str
    extension: Path
    workdir: Path | None


_MODULE: ModuleType | None = None
_COORDINATES: NDArray[np.float64] | None = None


def _initialize_worker(spec: _FortranPESSpec, coordinates: NDArray[np.float64]) -> None:
    global _MODULE, _COORDINATES

    from pyticc.pes.fortran import _load_module

    if spec.workdir is not None:
        os.chdir(spec.workdir)
    _MODULE = _load_module(spec.module_name, spec.extension)
    _COORDINATES = np.asfortranarray(coordinates)


def _evaluate_worker(RR: float) -> NDArray[np.float64]:
    if _MODULE is None or _COORDINATES is None:
        raise RuntimeError("Fortran PES worker is not initialized")
    return np.asarray(_MODULE.pyticc_interaction_grid(RR, _COORDINATES), dtype=np.float64)


def _evaluate_fortran_many(
    spec: _FortranPESSpec,
    R: NDArray[np.float64],
    coordinates: NDArray[np.float64],
    processes: int,
) -> NDArray[np.float64]:
    """
    Evaluate independent radial PES grids in isolated spawned processes.

    Inputs:
        spec: _FortranPESSpec - compiled extension and runtime-data locations
        R: NDArray[np.float64] - radial separations in atomic units
        coordinates: NDArray[np.float64] - internal coordinates with shape (n_coordinate, n_grid)
        processes: int - maximum number of worker processes

    Returns:
        values: NDArray[np.float64] - potential values with shape (n_R, n_grid)
    """
    n_process = min(processes, R.size)
    if n_process == 1:
        from pyticc.pes.fortran import _load_module, _make_interaction

        interaction = _make_interaction(_load_module(spec.module_name, spec.extension), spec.workdir)
        return np.stack([interaction(float(RR), coordinates) for RR in R])

    context = mp.get_context("spawn")
    with context.Pool(
        processes=n_process,
        initializer=_initialize_worker,
        initargs=(spec, coordinates),
    ) as pool:
        return np.stack(pool.map(_evaluate_worker, R, chunksize=1))
