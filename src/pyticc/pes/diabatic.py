from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from loguru import logger
from numpy.typing import NDArray

DiabaticMonomerPES = Callable[[NDArray[np.float64]], NDArray[np.float64]]
DiabaticInteractionPES = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]
DiabaticInteractionManyPES = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
DiabaticInteractionManyProcessesPES = Callable[[NDArray[np.float64], NDArray[np.float64], int], NDArray[np.float64]]
MonomerStatePES = Callable[[NDArray[np.float64]], NDArray[np.float64]]
RadialInput: TypeAlias = float | Sequence[float] | NDArray[np.float64]


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DiabaticPESWrapper:
    """
    Diabatic monomer potentials and interaction potential-energy matrix.

    ``monomer`` returns shape ``(n_grid, n_state)``. ``interaction`` returns the
    diabatic interaction matrix after subtracting each diagonal monomer potential, with shape
    (n_grid, n_state, n_state). Atom-diatom grid helpers pass R separately and
    arrange coordinate rows as (r, theta). A user PES adapter is responsible for
    conversion to its native coordinates, atom order, and units. The PyTICC-side
    contract uses bohr, radians, and Hartree.
    """

    n_state: int
    monomer: DiabaticMonomerPES
    interaction: DiabaticInteractionPES
    interaction_many: DiabaticInteractionManyPES | None = None
    _interaction_many_processes: DiabaticInteractionManyProcessesPES | None = field(default=None, repr=False, compare=False)
    _close: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.n_state < 1:
            message = f"n_state must be positive, but got {self.n_state}"
            logger.error(message)
            raise ValueError(message)

    def monomer_values(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate every diabatic monomer potential, returning shape ``(n_r, n_state)``."""
        coordinates = np.asarray(r, dtype=np.float64)
        if coordinates.ndim != 1:
            message = f"Monomer coordinates must be one-dimensional, but got shape {coordinates.shape}"
            logger.error(message)
            raise ValueError(message)

        values = _as_real_array(self.monomer(coordinates), "Diabatic monomer PES")
        expected_shape = (coordinates.size, self.n_state)
        _validate_values(values, expected_shape, "Diabatic monomer PES")
        return values

    def monomer_state(self, electronic_state: int) -> MonomerStatePES:
        """Return one zero-based electronic state's monomer potential callable."""
        if not 0 <= electronic_state < self.n_state:
            message = f"Electronic state {electronic_state} is outside 0..{self.n_state - 1}"
            logger.error(message)
            raise ValueError(message)

        def potential(r: NDArray[np.float64]) -> NDArray[np.float64]:
            """Evaluate one monomer potential, returning shape ``(n_r,)``."""
            return self.monomer_values(r)[:, electronic_state]

        return potential

    def close(self) -> None:
        """Release persistent resources owned by a compiled diabatic PES wrapper."""
        if self._close is not None:
            self._close()


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _as_real_array(values: object, name: str) -> NDArray[np.float64]:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        message = f"{name} must return real values"
        logger.error(message)
        raise ValueError(message)
    return np.asarray(array, dtype=np.float64)


# ----------------------------------------------------------------------------------------
def _validate_values(values: NDArray[np.float64], expected_shape: tuple[int, ...], name: str) -> None:
    if values.shape != expected_shape:
        message = f"{name} returned shape {values.shape}, but expected {expected_shape}"
        logger.error(message)
        raise ValueError(message)
    if not np.all(np.isfinite(values)):
        message = f"{name} returned non-finite values"
        logger.error(message)
        raise ValueError(message)


# ----------------------------------------------------------------------------------------
def _validate_diabatic_matrix(values: NDArray[np.float64], expected_shape: tuple[int, ...]) -> None:
    _validate_values(values, expected_shape, "Diabatic interaction PES")
    if not np.allclose(values, np.swapaxes(values, -2, -1), rtol=1.0e-12, atol=1.0e-12):
        message = "Diabatic interaction PES must return a symmetric matrix"
        logger.error(message)
        raise ValueError(message)


# ----------------------------------------------------------------------------------------
def _evaluate_diabatic_matrix(
    pes: DiabaticPESWrapper,
    R: RadialInput,
    coordinates: NDArray[np.float64],
    grid_shape: tuple[int, ...],
    *,
    processes: int,
) -> NDArray[np.float64]:
    """Dispatch scalar or batched radial diabatic-matrix evaluation and restore tensor-grid axes."""
    if not isinstance(processes, int) or isinstance(processes, bool) or processes < 1:
        message = f"processes must be a positive integer, but got {processes!r}"
        logger.error(message)
        raise ValueError(message)
    radial_points = np.asarray(R, dtype=np.float64)
    matrix_shape = (pes.n_state, pes.n_state)
    if radial_points.ndim == 0:
        values = _as_real_array(pes.interaction(float(radial_points), coordinates), "Diabatic interaction PES")
        expected_shape = (coordinates.shape[1], *matrix_shape)
        output_shape = (*grid_shape, *matrix_shape)
    elif radial_points.ndim == 1:
        if radial_points.size == 0:
            values = np.empty((0, coordinates.shape[1], *matrix_shape), dtype=np.float64)
        elif pes._interaction_many_processes is not None:
            values = _as_real_array(
                pes._interaction_many_processes(radial_points, coordinates, processes),
                "Diabatic interaction PES",
            )
        elif pes.interaction_many is None:
            values = np.stack([pes.interaction(float(RR), coordinates) for RR in radial_points])
            values = _as_real_array(values, "Diabatic interaction PES")
        else:
            values = _as_real_array(pes.interaction_many(radial_points, coordinates), "Diabatic interaction PES")
        expected_shape = (radial_points.size, coordinates.shape[1], *matrix_shape)
        output_shape = (radial_points.size, *grid_shape, *matrix_shape)
    else:
        message = f"R must be a scalar or one-dimensional array, but got shape {radial_points.shape}"
        logger.error(message)
        raise ValueError(message)

    _validate_diabatic_matrix(values, expected_shape)
    return values.reshape(output_shape)


# ----------------------------------------------------------------------------------------
def get_diabatic_potential_grid_atom_diatom(
    pes: DiabaticPESWrapper,
    R: RadialInput,
    r: NDArray[np.float64],
    theta: NDArray[np.float64],
    *,
    processes: int = 1,
) -> NDArray[np.float64]:
    """
    Evaluate an atom-diatom diabatic interaction matrix on a tensor-product Jacobi grid.

    Inputs:
        pes: DiabaticPESWrapper - diabatic monomer and interaction potentials
        R: RadialInput - scalar separation or one-dimensional radial batch in bohr
        r: NDArray[np.float64] - diatomic bond-length grid in bohr, shape ``(n_r,)``
        theta: NDArray[np.float64] - Jacobi-angle grid in radians, shape ``(n_theta,)``
        processes: int - temporary Fortran worker processes for a radial batch

    Returns:
        potential: NDArray[np.float64] - shape ``(n_r, n_theta, n_state, n_state)``
            for scalar R, or the same shape preceded by ``n_R`` for batched R
    """
    r_values = np.asarray(r, dtype=np.float64)
    theta_values = np.asarray(theta, dtype=np.float64)
    if r_values.ndim != 1 or theta_values.ndim != 1:
        message = f"r and theta must be one-dimensional, but got shapes {r_values.shape} and {theta_values.shape}"
        logger.error(message)
        raise ValueError(message)

    grids = np.meshgrid(r_values, theta_values, indexing="ij")
    coordinates = np.asfortranarray(np.stack(tuple(grid.reshape(-1) for grid in grids)))
    return _evaluate_diabatic_matrix(pes, R, coordinates, grids[0].shape, processes=processes)


# ----------------------------------------------------------------------------------------
