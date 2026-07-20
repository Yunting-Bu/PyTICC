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


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    from pyticc.constants import AU2CM
    from pyticc.pes.fortran import load_fortran_pes

    pes_dir = Path("/Users/byt/software/PES/ArHF")
    pes = load_fortran_pes(
        [pes_dir / "interaction-PES.f"],
        pes_dir / "pyticc_wrapper.f90",
        workdir=pes_dir,
    )

    labels = np.array(["global minimum", "linear local minimum", "off-minimum point"])
    R = np.array([6.470, 6.421, 6.647])
    r = np.full(3, 1.7325)
    theta_degree = np.array([0.0, 180.0, 109.724])
    theta = np.deg2rad(theta_degree)

    V = np.array([pes.interaction(R_i, np.array([[r_i], [theta_i]], dtype=np.float64))[0] for R_i, r_i, theta_i in zip(R, r, theta, strict=True)])

    print("ArHF interaction potential batch:")
    for label, R_i, r_i, theta_i, V_i in zip(labels, R, r, theta_degree, V * AU2CM, strict=True):
        print(f"{label:20s}  R={R_i:6.3f} bohr  r={r_i:6.4f} bohr  theta={theta_i:7.3f} deg  V={V_i:12.6f} cm-1")
