from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from loguru import logger
from numpy.typing import NDArray

MonomerPES = Callable[[NDArray[np.float64]], NDArray[np.float64]]
InteractionPES = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]
InteractionManyPES = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
RadialInput: TypeAlias = float | Sequence[float] | NDArray[np.float64]


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PESWrapper:
    """
    Monomer and interaction potential-energy surfaces.

    Members:
        interaction: InteractionPES - interaction energy evaluated from coordinates
            with shape (n_coordinate, n_grid), returning shape (n_grid,)
        monomer_X: MonomerPES | None - first monomer potential mapping coordinates
            with shape (n_grid,) to values with shape (n_grid,)
        monomer_Y: MonomerPES | None - second monomer potential mapping coordinates
            with shape (n_grid,) to values with shape (n_grid,)
        interaction_many: InteractionManyPES | None - batched interaction accepting
            R with shape (n_R,) and coordinates with shape (n_coordinate, n_grid),
            returning shape (n_R, n_grid)
    """

    interaction: InteractionPES
    monomer_X: MonomerPES | None = None
    monomer_Y: MonomerPES | None = None
    interaction_many: InteractionManyPES | None = None
    _close: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def close(self) -> None:
        """Release persistent resources owned by a compiled PES wrapper."""
        if self._close is not None:
            self._close()


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _evaluate(
    pes: PESWrapper,
    R: RadialInput,
    coordinates: NDArray[np.float64],
    grid_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """
    Dispatch scalar or batched radial PES evaluation and restore tensor-grid axes.

    ``coordinates`` has shape (n_coordinate, n_grid). A scalar R returns
    ``grid_shape``; R with shape (n_R,) returns shape (n_R, *grid_shape).
    """
    radial_points = np.asarray(R, dtype=np.float64)
    if radial_points.ndim == 0:
        values = np.asarray(pes.interaction(float(radial_points), coordinates), dtype=np.float64)
        expected_shape = (coordinates.shape[1],)
        output_shape = grid_shape
    elif radial_points.ndim == 1:
        if radial_points.size == 0:
            values = np.empty((0, coordinates.shape[1]), dtype=np.float64)
        elif pes.interaction_many is None:
            values = np.stack([pes.interaction(float(RR), coordinates) for RR in radial_points])
        else:
            values = np.asarray(pes.interaction_many(radial_points, coordinates), dtype=np.float64)
        expected_shape = (radial_points.size, coordinates.shape[1])
        output_shape = (radial_points.size, *grid_shape)
    else:
        message = f"R must be a scalar or one-dimensional array, but got shape {radial_points.shape}"
        logger.error(message)
        raise ValueError(message)

    if values.shape != expected_shape:
        message = f"Interaction PES returned shape {values.shape}, but expected {expected_shape}"
        logger.error(message)
        raise ValueError(message)
    return values.reshape(output_shape)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def get_Vgrid_atom_diatom(
    pes: PESWrapper,
    R: RadialInput,
    r: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Evaluate an atom-diatom interaction PES on tensor-product internal grids.

    Inputs:
        pes: PESWrapper - atom-diatom potential-energy surfaces
        R: float | Sequence[float] | NDArray[np.float64] - scalar separation or
            separations with shape (n_R,) in atomic units
        r: NDArray[np.float64] - diatomic bond-length grids in atomic units, shape
            (n_r,)
        theta: NDArray[np.float64] - Jacobi-angle grids in radians, shape
            (n_theta,)

    Returns:
        V: NDArray[np.float64] - interaction grid with shape (n_r, n_theta) for
            scalar R, or (n_R, n_r, n_theta) for batched R
    """
    grids = np.meshgrid(r, theta, indexing="ij")
    coordinates = np.asfortranarray(np.stack(tuple(grid.reshape(-1) for grid in grids)))
    return _evaluate(pes, R, coordinates, grids[0].shape)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def get_Vgrid_atom_diatom_electric_sf(
    pes: PESWrapper,
    R: RadialInput,
    r: NDArray[np.float64],
    gamma: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Evaluate an atom-diatom interaction PES on an SF angular geometry grid.

    Inputs:
        pes: PESWrapper - atom-diatom potential-energy surfaces
        R: RadialInput - scalar separation or separations with shape (n_R,)
            in atomic units
        r: NDArray[np.float64] - diatomic PODVR bond-length grids in atomic
            units, shape (n_r,)
        gamma: NDArray[np.float64] - Jacobi angles in radians with arbitrary
            angular tensor shape (n_theta_r,n_theta_R,n_delta)

    Returns:
        V: NDArray[np.float64] - interaction grid with shape
            (n_r,*gamma.shape) for scalar R, or the same shape preceded by n_R
    """
    radial_grid = np.asarray(r, dtype=np.float64)
    angle_grid = np.asarray(gamma, dtype=np.float64)
    grid_shape = (radial_grid.size, *angle_grid.shape)
    radial_coordinates = np.broadcast_to(radial_grid.reshape((-1, *(1 for _ in angle_grid.shape))), grid_shape)
    angular_coordinates = np.broadcast_to(angle_grid[None, ...], grid_shape)
    coordinates = np.asfortranarray(np.stack((radial_coordinates.reshape(-1), angular_coordinates.reshape(-1))))
    return _evaluate(pes, R, coordinates, grid_shape)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def get_Vgrid_atom_triatom(
    pes: PESWrapper,
    R: RadialInput,
    r_1: NDArray[np.float64],
    r_2: NDArray[np.float64],
    theta_1: NDArray[np.float64],
    theta_2: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Evaluate an atom-triatom interaction PES on tensor-product internal grids.

    Inputs:
        pes: PESWrapper - atom-triatom potential-energy surfaces
        R: float | Sequence[float] | NDArray[np.float64] - scalar separation or
            separations with shape (n_R,) in atomic units
        r_1: NDArray[np.float64] - first triatomic radial grid, shape (n_r1,)
        r_2: NDArray[np.float64] - second triatomic radial grid, shape (n_r2,)
        theta_1: NDArray[np.float64] - triatomic bend grids in radians, shape
            (n_theta_1,)
        theta_2: NDArray[np.float64] - external polar grids in radians, shape
            (n_theta_2,)
        phi: NDArray[np.float64] - dihedral grids in radians, shape (n_phi,)

    Returns:
        V: NDArray[np.float64] - interaction grid with shape
            (n_r1, n_r2, n_theta_1, n_theta_2, n_phi) for scalar R, or the same
            shape preceded by n_R for batched R
    """
    grids = np.meshgrid(r_1, r_2, theta_1, theta_2, phi, indexing="ij")
    coordinates = np.asfortranarray(np.stack(tuple(grid.reshape(-1) for grid in grids)))
    return _evaluate(pes, R, coordinates, grids[0].shape)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def get_Vgrid_diatom_diatom(
    pes: PESWrapper,
    R: RadialInput,
    r_X: NDArray[np.float64],
    r_Y: NDArray[np.float64],
    theta_X: NDArray[np.float64],
    theta_Y: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Evaluate a diatom-diatom interaction PES on tensor-product internal grids.

    Inputs:
        pes: PESWrapper - diatom-diatom potential-energy surfaces
        R: float | Sequence[float] | NDArray[np.float64] - scalar separation or
            separations with shape (n_R,) in atomic units
        r_X: NDArray[np.float64] - first diatom bond-length grids, shape (n_r_X,)
        r_Y: NDArray[np.float64] - second diatom bond-length grids, shape (n_r_Y,)
        theta_X: NDArray[np.float64] - first polar-angle grids, shape (n_theta_X,)
        theta_Y: NDArray[np.float64] - second polar-angle grids, shape (n_theta_Y,)
        phi: NDArray[np.float64] - dihedral-angle grids, shape (n_phi,)

    Returns:
        V: NDArray[np.float64] - interaction grid with shape
            (n_r_X, n_r_Y, n_theta_X, n_theta_Y, n_phi) for scalar R, or the same
            shape preceded by n_R for batched R
    """
    grids = np.meshgrid(r_X, r_Y, theta_X, theta_Y, phi, indexing="ij")
    coordinates = np.asfortranarray(np.stack(tuple(grid.reshape(-1) for grid in grids)))
    return _evaluate(pes, R, coordinates, grids[0].shape)
