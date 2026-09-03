"""Validation of scalar PES grids for complete-diatom exchange."""

import numpy as np
from numpy.typing import NDArray


# ----------------------------------------------------------------------------------------
def validate_exchange_quadrature(
    r_X: NDArray[np.float64],
    r_Y: NDArray[np.float64],
    cos_X: NDArray[np.float64],
    cos_Y: NDArray[np.float64],
    weights_X: NDArray[np.float64],
    weights_Y: NDArray[np.float64],
) -> None:
    """Require a tensor grid closed under complete-molecule exchange.

    Inputs:
        r_X, r_Y: NDArray[np.float64] - equal radial nodes in bohr
        cos_X, cos_Y: NDArray[np.float64] - equal polar cosine nodes, each
            invariant under sign reversal and reversal of array order
        weights_X, weights_Y: NDArray[np.float64] - equal, reversal-symmetric
            dimensionless polar quadrature weights

    Returns:
        None; raises ValueError when the grid is not exchange closed
    """
    pairs = ((r_X, r_Y), (cos_X, cos_Y), (cos_X, -cos_X[::-1]), (weights_X, weights_Y), (weights_X, weights_X[::-1]))
    if any(a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=1.0e-14) for a, b in pairs):
        raise ValueError("Molecule exchange requires identical radial grids and equal, reflection-symmetric X/Y polar quadratures")


# ----------------------------------------------------------------------------------------
def validate_exchange_potential(values: NDArray[np.float64]) -> None:
    r"""Check scalar PES invariance without averaging or modifying the PES.

    Formula:
        In the unsigned torsional convention phi in [0,pi], exchange maps
        (r_X,r_Y,theta_X,theta_Y,phi) to
        (r_Y,r_X,pi-theta_Y,pi-theta_X,phi), at unchanged separation R.
        The corresponding grid values must agree to atol=1e-12 Hartree and
        rtol=1e-10. Polar nodes must satisfy validate_exchange_quadrature.

    Inputs:
        values: NDArray[np.float64] - scalar Hartree values, shape
            (n_r,n_r,n_theta,n_theta,n_phi), optionally preceded by n_R

    Returns:
        None; raises ValueError for nonfinite or exchange-asymmetric values
    """
    if values.ndim not in (5, 6) or values.shape[-5] != values.shape[-4] or values.shape[-3] != values.shape[-2]:
        raise ValueError("Molecule-exchange PES grid must have equal radial and polar X/Y dimensions")
    if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
        raise ValueError("Molecule-exchange scalar PES must contain finite real values")
    exchanged = np.flip(np.swapaxes(np.swapaxes(values, -5, -4), -3, -2), axis=(-3, -2))
    if not np.allclose(values, exchanged, rtol=1.0e-10, atol=1.0e-12):
        error = float(np.max(np.abs(values - exchanged)))
        raise ValueError(f"PES violates complete-molecule exchange symmetry (maximum difference {error:.6g} Hartree)")
