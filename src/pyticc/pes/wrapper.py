from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
        interaction: InteractionPES - interaction energy evaluated as V(R, coordinates)
        monomer_X: MonomerPES | None - first monomer potential evaluated as V(r)
        monomer_Y: MonomerPES | None - second monomer potential evaluated as V(r)
        interaction_many: InteractionManyPES | None - interaction energies evaluated for multiple R
    """

    interaction: InteractionPES
    monomer_X: MonomerPES | None = None
    monomer_Y: MonomerPES | None = None
    interaction_many: InteractionManyPES | None = None


def _evaluate(
    pes: PESWrapper,
    R: RadialInput,
    coordinates: NDArray[np.float64],
    grid_shape: tuple[int, ...],
) -> NDArray[np.float64]:
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
        R: float | Sequence[float] | NDArray[np.float64] - atom-diatom center-of-mass separations in atomic units
        r: NDArray[np.float64] - diatomic bond-length grids in atomic units
        theta: NDArray[np.float64] - Jacobi-angle grids in radians

    Returns:
        V: NDArray[np.float64] - interaction grid with axes (r, theta), preceded by R for batched input
    """
    grids = np.meshgrid(r, theta, indexing="ij")
    coordinates = np.asfortranarray(np.stack(tuple(grid.reshape(-1) for grid in grids)))
    return _evaluate(pes, R, coordinates, grids[0].shape)


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
        R: float | Sequence[float] | NDArray[np.float64] - center-of-mass separations in atomic units
        r_X: NDArray[np.float64] - first diatom bond-length grids in atomic units
        r_Y: NDArray[np.float64] - second diatom bond-length grids in atomic units
        theta_X: NDArray[np.float64] - first polar-angle grids in radians
        theta_Y: NDArray[np.float64] - second polar-angle grids in radians
        phi: NDArray[np.float64] - dihedral-angle grids in radians

    Returns:
        V: NDArray[np.float64] - interaction grid with internal axes, preceded by R for batched input
    """
    grids = np.meshgrid(r_X, r_Y, theta_X, theta_Y, phi, indexing="ij")
    coordinates = np.asfortranarray(np.stack(tuple(grid.reshape(-1) for grid in grids)))
    return _evaluate(pes, R, coordinates, grids[0].shape)


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
