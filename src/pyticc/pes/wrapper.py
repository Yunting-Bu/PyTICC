from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray

MonomerPES = Callable[[NDArray[np.float64]], NDArray[np.float64]]
InteractionPES = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PESWrapper:
    """
    Monomer and interaction potential-energy surfaces.

    Members:
        interaction: InteractionPES - interaction energy evaluated as V(R, coordinates)
        monomer_X: MonomerPES | None - first monomer potential evaluated as V(r)
        monomer_Y: MonomerPES | None - second monomer potential evaluated as V(r)
    """

    interaction: InteractionPES
    monomer_X: MonomerPES | None = None
    monomer_Y: MonomerPES | None = None


def _evaluate(pes: PESWrapper, R: float, coordinates: NDArray[np.float64], grid_shape: tuple[int, ...]) -> NDArray[np.float64]:
    values = np.asarray(pes.interaction(float(R), coordinates), dtype=np.float64)
    if values.shape != (coordinates.shape[1],):
        message = f"Interaction PES returned shape {values.shape}, but expected {(coordinates.shape[1],)}"
        logger.error(message)
        raise ValueError(message)
    return values.reshape(grid_shape)


# ----------------------------------------------------------------------------------------
def get_Vgrid_atom_diatom(
    pes: PESWrapper,
    R: float,
    r: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Evaluate an atom-diatom interaction PES on tensor-product internal grids.

    Inputs:
        pes: PESWrapper - atom-diatom potential-energy surfaces
        R: float - atom-diatom center-of-mass separation in atomic units
        r: NDArray[np.float64] - diatomic bond-length grids in atomic units
        theta: NDArray[np.float64] - Jacobi-angle grids in radians

    Returns:
        V: NDArray[np.float64] - interaction grid with axes (r, theta) in atomic units
    """
    grids = np.meshgrid(r, theta, indexing="ij")
    coordinates = np.asfortranarray(np.stack(tuple(grid.reshape(-1) for grid in grids)))
    return _evaluate(pes, R, coordinates, grids[0].shape)


# ----------------------------------------------------------------------------------------
def get_Vgrid_diatom_diatom(
    pes: PESWrapper,
    R: float,
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
        R: float - separation between the two diatom centers of mass in atomic units
        r_X: NDArray[np.float64] - first diatom bond-length grids in atomic units
        r_Y: NDArray[np.float64] - second diatom bond-length grids in atomic units
        theta_X: NDArray[np.float64] - first polar-angle grids in radians
        theta_Y: NDArray[np.float64] - second polar-angle grids in radians
        phi: NDArray[np.float64] - dihedral-angle grids in radians

    Returns:
        V: NDArray[np.float64] - interaction grid with axes (r_X, r_Y, theta_X, theta_Y, phi)
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
