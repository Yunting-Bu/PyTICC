from pathlib import Path
from typing import TypeAlias

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike, NDArray

EnergyInput: TypeAlias = ArrayLike | str | Path


# ----------------------------------------------------------------------------------------
def get_Etot(source: EnergyInput) -> NDArray[np.float64]:
    """
    Read and validate total energies in atomic units.

    Inputs:
        source: EnergyInput - array-like input with shape (n_energy,), or a
            one-column text file

    Returns:
        energies: NDArray[np.float64] - total energies in atomic units, shape
            (n_energy,)
    """
    if isinstance(source, str | Path):
        path = Path(source)
        try:
            energies = np.loadtxt(path, dtype=np.float64, comments="#", ndmin=1)
        except (OSError, ValueError) as error:
            message = f"Failed to read total energies from {path}: {error}"
            logger.error(message)
            raise ValueError(message) from error
    else:
        try:
            energies = np.asarray(source, dtype=np.float64)
        except (TypeError, ValueError) as error:
            message = f"Failed to convert total energies to an array: {error}"
            logger.error(message)
            raise ValueError(message) from error

    if energies.ndim != 1:
        message = f"Total energies must be one-dimensional, but got shape={energies.shape}"
        logger.error(message)
        raise ValueError(message)
    if energies.size == 0:
        message = "At least one total energy is required"
        logger.error(message)
        raise ValueError(message)
    if not np.all(np.isfinite(energies)):
        message = "Total energies must be finite"
        logger.error(message)
        raise ValueError(message)

    energies = energies.copy()
    energies.setflags(write=False)
    return energies
