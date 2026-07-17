import numpy as np
from loguru import logger
from numpy.typing import NDArray


# ----------------------------------------------------------------------------------------
def get_Wmat(
    R: float,
    Etot: float,
    reduced_mass: float,
    E_int: NDArray[np.float64],
    Umat: NDArray[np.float64],
    Vmat: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""
    Get the radial coupled-equation matrix.

    Formula:
        W(R; Etot) = U / R**2
                     + 2 * reduced_mass * V(R)
                     + 2 * reduced_mass * diag(E_int - Etot)

        The radial coupled equations are

        d**2 F(R) / dR**2 = W(R; Etot) F(R).

    Inputs:
        R: float - intermolecular separation in atomic units
        Etot: float - total energy in atomic units
        reduced_mass: float - collision reduced mass in atomic units
        E_int: NDArray[np.float64] - channel internal energies in atomic units
        Umat: NDArray[np.float64] - dimensionless centrifugal matrix
        Vmat: NDArray[np.float64] - interaction potential matrix in atomic units

    Returns:
        Wmat: NDArray[np.float64] - radial equation matrix in inverse length squared
    """
    if R <= 0.0:
        message = f"R must be positive, but got R={R}"
        logger.error(message)
        raise ValueError(message)
    if reduced_mass <= 0.0:
        message = f"reduced_mass must be positive, but got reduced_mass={reduced_mass}"
        logger.error(message)
        raise ValueError(message)

    n_channel = E_int.size
    expected_shape = (n_channel, n_channel)
    if E_int.ndim != 1 or Umat.shape != expected_shape or Vmat.shape != expected_shape:
        message = f"E_int, Umat, and Vmat shapes are inconsistent: E_int.shape={E_int.shape}, Umat.shape={Umat.shape}, Vmat.shape={Vmat.shape}"
        logger.error(message)
        raise ValueError(message)

    Wmat = Umat / R**2 + 2.0 * reduced_mass * Vmat
    diagonal = np.diag_indices(n_channel)
    Wmat[diagonal] += 2.0 * reduced_mass * (E_int - Etot)
    return Wmat


# ----------------------------------------------------------------------------------------
