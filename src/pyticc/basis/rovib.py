from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray


@dataclass(frozen=True)
class RovibBasis:
    """
    Diatomic rovibrational states represented on a radial grid.

    The grid may be a complete primitive DVR grid or a contracted PODVR grid.
    The representation is determined by the builder, not by the data type.

    Members:
        grids: NDArray[np.float64] - radial grid in atomic units, shape (n_grid,)
        E_vj: NDArray[np.float64] - rovibrational energies indexed as E_vj[v,j],
            shape (n_v,n_j)
        WF_vj: NDArray[np.float64] - radial wavefunctions indexed as
            WF_vj[grid,v,j], shape (n_grid,n_v,n_j)
    """

    grids: NDArray[np.float64]
    E_vj: NDArray[np.float64]
    WF_vj: NDArray[np.float64]

    def __post_init__(self) -> None:
        grids = np.asarray(self.grids)
        energies = np.asarray(self.E_vj)
        wavefunctions = np.asarray(self.WF_vj)
        if grids.ndim != 1 or energies.ndim != 2 or wavefunctions.shape != (grids.size, *energies.shape):
            message = (
                "RovibBasis shapes must satisfy grids=(n_grid,), E_vj=(n_v,n_j), "
                f"and WF_vj=(n_grid,n_v,n_j), but got {grids.shape}, {energies.shape}, and {wavefunctions.shape}"
            )
            logger.error(message)
            raise ValueError(message)
