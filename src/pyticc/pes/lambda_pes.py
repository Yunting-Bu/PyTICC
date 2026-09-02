from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.pes.adiabatic import MonomerPES, PESWrapper

LambdaInteraction = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]
LambdaInteractionMany = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
LambdaInteractionManyProcesses = Callable[[NDArray[np.float64], NDArray[np.float64], int], NDArray[np.float64]]
RadialInput: TypeAlias = float | Sequence[float] | NDArray[np.float64]


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LambdaPES:
    r"""
    Interaction PES for one signed-Lambda electronic manifold.

    Formula:
        The final output axis contains

        V_sum = (V_A'' + V_A')/2,
        V_dif = (V_A'' - V_A')/2.

        In the signed basis {+Lambda,-Lambda}, this convention gives diagonal
        V_sum and signed-Lambda-flipping coupling V_dif. For Lambda=0 the
        second component is identically zero.

    Members:
        interaction: LambdaInteraction - interaction evaluated at scalar R and
            coordinates with shape (n_coordinate,n_grid), returning
            (n_grid,2) in Hartree
        monomer_Y: MonomerPES | None - optional diatomic monomer potential
        interaction_many: LambdaInteractionMany | None - radial-batch evaluator
            returning shape (n_R,n_grid,2)
    """

    interaction: LambdaInteraction
    monomer_Y: MonomerPES | None = None
    interaction_many: LambdaInteractionMany | None = None
    _interaction_many_processes: LambdaInteractionManyProcesses | None = field(default=None, repr=False, compare=False)
    _close: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def close(self) -> None:
        """Release persistent resources owned by a compiled PES wrapper."""
        if self._close is not None:
            self._close()


# ----------------------------------------------------------------------------------------
def as_lambda_pes(pes: PESWrapper) -> LambdaPES:
    r"""
    Adapt a scalar PES to the Lambda=0 convention without changing its values.

    Formula:
        V_sum(R,q) = V(R,q),    V_dif(R,q) = 0.

    Inputs:
        pes: PESWrapper - scalar interaction PES in bohr, radians, and Hartree

    Returns:
        lambda_pes: LambdaPES - two-component view of the scalar PES
    """

    def interaction(R: float, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(pes.interaction(R, coordinates), dtype=np.float64)
        return np.stack((values, np.zeros_like(values)), axis=-1)

    interaction_many: LambdaInteractionMany | None = None
    if pes.interaction_many is not None:
        source_interaction_many = pes.interaction_many

        def evaluate_many(R: NDArray[np.float64], coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
            values = np.asarray(source_interaction_many(R, coordinates), dtype=np.float64)
            return np.stack((values, np.zeros_like(values)), axis=-1)

        interaction_many = evaluate_many

    interaction_many_processes: LambdaInteractionManyProcesses | None = None
    if pes._interaction_many_processes is not None:
        source_interaction_many_processes = pes._interaction_many_processes

        def evaluate_many_processes(
            R: NDArray[np.float64],
            coordinates: NDArray[np.float64],
            processes: int,
        ) -> NDArray[np.float64]:
            values = np.asarray(source_interaction_many_processes(R, coordinates, processes), dtype=np.float64)
            return np.stack((values, np.zeros_like(values)), axis=-1)

        interaction_many_processes = evaluate_many_processes

    return LambdaPES(
        interaction=interaction,
        monomer_Y=pes.monomer_Y,
        interaction_many=interaction_many,
        _interaction_many_processes=interaction_many_processes,
        _close=pes.close,
    )


# ----------------------------------------------------------------------------------------
def _validate(values: NDArray[np.float64], expected_shape: tuple[int, ...]) -> None:
    if values.shape != expected_shape:
        message = f"Lambda PES returned shape {values.shape}, but expected {expected_shape}"
        logger.error(message)
        raise ValueError(message)
    if not np.all(np.isfinite(values)):
        message = "Lambda PES returned non-finite values"
        logger.error(message)
        raise ValueError(message)


# ----------------------------------------------------------------------------------------
def _evaluate_lambda(
    pes: LambdaPES,
    R: RadialInput,
    coordinates: NDArray[np.float64],
    grid_shape: tuple[int, ...],
    *,
    processes: int,
) -> NDArray[np.float64]:
    """Evaluate a scalar or radial batch and restore tensor-grid axes."""
    if not isinstance(processes, int) or isinstance(processes, bool) or processes < 1:
        message = f"processes must be a positive integer, but got {processes!r}"
        logger.error(message)
        raise ValueError(message)
    radial_points = np.asarray(R, dtype=np.float64)
    if radial_points.ndim == 0:
        values = np.asarray(pes.interaction(float(radial_points), coordinates), dtype=np.float64)
        expected_shape = (coordinates.shape[1], 2)
        output_shape = (*grid_shape, 2)
    elif radial_points.ndim == 1:
        if radial_points.size == 0:
            values = np.empty((0, coordinates.shape[1], 2), dtype=np.float64)
        elif pes._interaction_many_processes is not None:
            values = np.asarray(pes._interaction_many_processes(radial_points, coordinates, processes), dtype=np.float64)
        elif pes.interaction_many is None:
            values = np.stack([pes.interaction(float(RR), coordinates) for RR in radial_points])
        else:
            values = np.asarray(pes.interaction_many(radial_points, coordinates), dtype=np.float64)
        expected_shape = (radial_points.size, coordinates.shape[1], 2)
        output_shape = (radial_points.size, *grid_shape, 2)
    else:
        message = f"R must be a scalar or one-dimensional array, but got shape {radial_points.shape}"
        logger.error(message)
        raise ValueError(message)
    _validate(values, expected_shape)
    return values.reshape(output_shape)


# ----------------------------------------------------------------------------------------
def get_lambda_grid_atom_diatom(
    pes: LambdaPES,
    R: RadialInput,
    r: NDArray[np.float64],
    theta: NDArray[np.float64],
    *,
    processes: int = 1,
) -> NDArray[np.float64]:
    """
    Evaluate V_sum and V_dif on an atom-diatom Jacobi tensor grid.

    Inputs:
        pes: LambdaPES - two-component interaction PES
        R: RadialInput - scalar separation or radial batch in bohr
        r: NDArray[np.float64] - diatomic bond grid in bohr, shape (n_r,)
        theta: NDArray[np.float64] - Jacobi-angle grid in radians, shape
            (n_theta,)
        processes: int - temporary Fortran worker processes for a radial batch

    Returns:
        potential: NDArray[np.float64] - shape (n_r,n_theta,2), optionally
            preceded by a radial-batch axis
    """
    r_values = np.asarray(r, dtype=np.float64)
    theta_values = np.asarray(theta, dtype=np.float64)
    if r_values.ndim != 1 or theta_values.ndim != 1:
        message = f"r and theta must be one-dimensional, but got shapes {r_values.shape} and {theta_values.shape}"
        logger.error(message)
        raise ValueError(message)
    grids = np.meshgrid(r_values, theta_values, indexing="ij")
    coordinates = np.asfortranarray(np.stack(tuple(grid.reshape(-1) for grid in grids)))
    return _evaluate_lambda(pes, R, coordinates, grids[0].shape, processes=processes)


# ----------------------------------------------------------------------------------------
