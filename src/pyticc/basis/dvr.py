from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.linalg import eigh


# --------------------------------------------------------------------------------
def sine_dvr_grids(a: float, b: float, n: int) -> tuple[NDArray[np.float64], float]:
    r"""
    Get sine dvr grids in the interval [a, b] with n points.
    Formula:
    x_l = a + \frac{l(b-a)}{n+1}, l = 1, 2, ..., n
    w = \frac{b-a}{n+1}

    Inputs:
        a: float - left boundary of the interval
        b: float - right boundary of the interval
        n: int - number of points

    Returns:
        x: NDArray[np.float64] - sine DVR grids, shape (n,)
        w: float - weights
    """
    if n < 2:
        message = f"n should be greater than 1, but got n = {n}"
        logger.warning(message)

    i = np.arange(1, n + 1, dtype=np.float64)
    length = b - a
    x = a + i * length / (n + 1)
    w = length / (n + 1)

    return x, w


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
def sine_dvr_to_fbr(n: int) -> NDArray[np.float64]:
    r"""
    Get dvr to fbr transformation matrix.
    Formula:
        B_{ml} = \sqrt{\frac{2}{n+1}}\sin(\frac{ml\pi}{n+1})

    Inputs:
        n: int - number of points

    Returns:
        B: NDArray[np.float64] - DVR-to-FBR transformation matrix, shape (n, n)
    """
    idx_m = np.arange(1, n + 1, dtype=np.float64)
    idx_l = np.arange(1, n + 1, dtype=np.float64)
    B = np.sqrt(2.0 / (n + 1)) * np.sin(np.pi * idx_m[:, None] * idx_l[None, :] / (n + 1))

    return B


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
def sine_dvr_kinetic(a: float, b: float, n: int, mass: float) -> NDArray[np.float64]:
    r"""
    Get sine dvr kinetic energy matrix.

    Formula:
        T_{ij} = \frac{\pi^2}{4m(b-a)^2} \left[
            \left( \frac{2(n+1)^2 + 1}{3}
            - \sin\left(\frac{i\pi}{n+1}\right)^{-2} \right) \delta_{ij}
            + \left( \sin\left(\frac{(i-j)\pi}{2(n+1)}\right)^{-2}
            - \sin\left(\frac{(i+j)\pi}{2(n+1)}\right)^{-2} \right) (-1)^{i-j} (1-\delta_{ij})
        \right]

    Inputs:
        a: float - left boundary of the interval
        b: float - right boundary of the interval
        n: int - number of points
        mass: float - mass in atomic unit

    Returns:
        T: NDArray[np.float64] - sine DVR kinetic energy matrix, shape (n, n)
    """

    pre_factor = (np.pi**2) / (4.0 * mass * (b - a) ** 2)
    m = n + 1

    idx = np.arange(1, n + 1, dtype=np.float64)  # i = 1, ..., n
    i_grid, j_grid = np.meshgrid(idx, idx, indexing="ij")

    # diagonal
    sin_i = np.sin(np.pi * idx / m)
    diag = (2.0 * m**2 + 1.0) / 3.0 - 1.0 / sin_i**2

    # off-diagonal only; mask out diagonal where sin(pi*(i-j)/2m) = 0
    sin_diff = np.sin(np.pi * (i_grid - j_grid) / (2.0 * m))
    sin_sum = np.sin(np.pi * (i_grid + j_grid) / (2.0 * m))
    with np.errstate(divide="ignore", invalid="ignore"):
        off = 1.0 / sin_diff**2 - 1.0 / sin_sum**2
    off[~np.isfinite(off)] = 0.0
    np.fill_diagonal(off, 0.0)  # the diagonal contribution of the off-term is 0 by (1-delta_ij)

    phase = np.where((i_grid - j_grid) % 2 == 0, 1.0, -1.0)
    np.fill_diagonal(phase, 1.0)

    T = diag[:, None] * np.eye(n) + off * phase
    T *= pre_factor
    return T


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
def phase_fix(A: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Flip the overall sign so the first significant element is positive.

    Inputs:
        A: NDArray[np.float64] - input vector with shape (n,), or matrix with shape
            (n_row, n_column)

    Returns:
        A: NDArray[np.float64] - phase-fixed array with the same shape as the input
    """
    threshold = 1e-4
    res = np.array(A, dtype=np.float64, copy=True)

    if res.ndim == 1:
        for value in res:
            if abs(value) <= threshold:
                continue

            if value < 0.0:
                res *= -1.0

            return res
        message = "All elements are below the threshold."
        logger.error(message)
        raise ValueError(message)

    if res.ndim == 2:
        for col in range(res.shape[1]):
            res[:, col] = phase_fix(res[:, col])
        return res

    message = "Input array must be 1D or 2D, returning the original array."
    logger.warning(message)
    return res


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
def sine_dvr_vib(T: NDArray[np.float64], V: NDArray[np.float64], n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Get the eigenvalues and eigenvectors of the vibrational Hamiltonian in sine DVR.
    Formula:
        H_{ij} = T_{ij} + V_i \delta_{ij}

    Inputs:
        T: NDArray[np.float64] - kinetic energy matrix, shape (n, n)
        V: NDArray[np.float64] - potential energy on the DVR grids, shape (n,)
        n: int - number of points

    Returns:
        eigen_val: NDArray[np.float64] - eigenvalues of the vibrational Hamiltonian,
            shape (n,)
        eigen_vec: NDArray[np.float64] - column eigenvectors of the vibrational
            Hamiltonian, shape (n, n)
    """
    H = T.copy()
    diag_indices = np.diag_indices(n)
    H[diag_indices] += V

    eigen_val, eigen_vec = eigh(H)
    eigen_vec = phase_fix(eigen_vec)

    return eigen_val, eigen_vec


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
@dataclass(frozen=True)
class SineDVR:
    """
    One-dimensional sine-DVR basis and vibrational eigenstates.

    Members:
        n_dvr: int - number of DVR points
        interval: tuple[float, float] - left and right coordinate boundaries
        grids: NDArray[np.float64] - DVR coordinates, shape (n_dvr,)
        weights: float - uniform DVR quadrature weight
        dvr_to_fbr: NDArray[np.float64] - DVR-to-FBR transformation, shape
            (n_dvr, n_dvr)
        eigen_val: NDArray[np.float64] - vibrational eigenvalues, shape (n_dvr,)
        eigen_vec: NDArray[np.float64] - column eigenvectors on the DVR grid, shape
            (n_dvr, n_dvr)
    """

    n_dvr: int
    interval: tuple[float, float]
    grids: NDArray[np.float64]
    weights: float
    dvr_to_fbr: NDArray[np.float64]
    eigen_val: NDArray[np.float64]
    eigen_vec: NDArray[np.float64]


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
@dataclass(frozen=True)
class RovibDVR:
    """
    Diatomic rovibrational basis on one complete primitive DVR grid.

    Members:
        grids: NDArray[np.float64] - primitive bond-length grid, shape (n_dvr,)
        E_vj: NDArray[np.float64] - rovibrational energies, shape (n_v, n_j)
        WF_vj: NDArray[np.float64] - primitive-grid wavefunctions, shape
            (n_dvr, n_v, n_j)
    """

    grids: NDArray[np.float64]
    E_vj: NDArray[np.float64]
    WF_vj: NDArray[np.float64]


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
def build_RovibDVR(dvr: SineDVR, vmax: int, jmax: int, mass: float) -> RovibDVR:
    """
    Build full primitive-DVR rovibrational states from a vibrational SineDVR.

    The complete eigendecomposition stored by SineDVR reconstructs the j=0
    Hamiltonian without requiring the original potential callable. Each rotational
    Hamiltonian then adds j(j+1)/(2*mass*r**2) on the same primitive coordinate
    basis, preserving one common grid gauge across electronic states.

    Inputs:
        dvr: SineDVR - complete primitive vibrational DVR basis
        vmax: int - highest retained vibrational quantum number
        jmax: int - highest retained rotational quantum number
        mass: float - diatomic reduced mass in atomic units

    Returns:
        rovib: RovibDVR - energies and wavefunctions on the primitive DVR grid
    """
    if vmax < 0 or jmax < 0:
        message = f"vmax and jmax must be non-negative, but got vmax={vmax}, jmax={jmax}"
        logger.error(message)
        raise ValueError(message)
    if mass <= 0.0 or not np.isfinite(mass):
        message = f"mass must be positive and finite, but got {mass}"
        logger.error(message)
        raise ValueError(message)

    n_dvr = dvr.grids.size
    n_v = min(vmax + 1, n_dvr)
    energies = np.zeros((vmax + 1, jmax + 1), dtype=np.float64)
    wavefunctions = np.zeros((n_dvr, vmax + 1, jmax + 1), dtype=np.float64)
    reference_hamiltonian = (dvr.eigen_vec * dvr.eigen_val[None, :]) @ dvr.eigen_vec.T
    diagonal = np.diag_indices(n_dvr)

    for j in range(jmax + 1):
        hamiltonian = reference_hamiltonian.copy()
        hamiltonian[diagonal] += j * (j + 1) / (2.0 * mass * dvr.grids**2)
        eigenvalues, eigenvectors = eigh(hamiltonian, subset_by_index=(0, n_v - 1))
        energies[:n_v, j] = eigenvalues
        wavefunctions[:, :n_v, j] = phase_fix(eigenvectors)

    return RovibDVR(grids=dvr.grids, E_vj=energies, WF_vj=wavefunctions)


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
def build_SineDVR(a: float, b: float, n_dvr: int, mass: float, pot_func: Callable[[NDArray[np.float64]], NDArray[np.float64]]) -> SineDVR:
    """
    Build a SineDVR object with the given parameters.

    Inputs:
        a: float - left boundary of the interval
        b: float - right boundary of the interval
        n_dvr: int - number of points
        mass: float - mass in atomic unit
        pot_func: Callable[[NDArray[np.float64]], NDArray[np.float64]] - vectorized
            potential mapping coordinates with shape (n_dvr,) to values with shape
            (n_dvr,)

    Returns:
        SineDVR: SineDVR - basis containing arrays whose leading DVR dimension is
            n_dvr
    """
    if n_dvr < 2:
        message = f"n_dvr should be greater than 1, but got n_dvr = {n_dvr}"
        logger.warning(message)

    if a >= b:
        message = f"Invalid interval: a ({a}) must be less than b ({b})"
        logger.error(message)
        raise ValueError(message)

    grids, weights = sine_dvr_grids(a, b, n_dvr)
    dvr_to_fbr = sine_dvr_to_fbr(n_dvr)
    pot = np.asarray(pot_func(grids), dtype=np.float64)
    T_mat = sine_dvr_kinetic(a, b, n_dvr, mass)
    eigen_val, eigen_vec = sine_dvr_vib(T_mat, pot, n_dvr)

    return SineDVR(n_dvr=n_dvr, interval=(a, b), grids=grids, weights=weights, dvr_to_fbr=dvr_to_fbr, eigen_val=eigen_val, eigen_vec=eigen_vec)


# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------


if __name__ == "__main__":
    import numpy as np
    from numpy.typing import NDArray

    from pyticc.constants import AMU2AU, ANG2AU, EV2AU

    def H2_morse(x: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Morse potential energy function.
        Formula:
            V(x) = D_e (1 - e^{-a(x-x_e)})^2

        Inputs:
            x: NDArray[np.float64] - positions in atomic units, shape (n_point,)

        Returns:
            V: NDArray[np.float64] - potential energies in atomic units, shape
                (n_point,)
        """
        D_e = 4.7446 * EV2AU  # au (from eV)
        a = 1.9426 / ANG2AU  # 1/au (from 1/Angstrom)
        x_e = 0.7416 * ANG2AU  # au (from Angstrom)

        V = D_e * (1 - np.exp(-a * (x - x_e))) ** 2
        return V

    m_H = 1.00782503223 * AMU2AU  # au
    m_H2 = m_H / 2.0  # reduced mass of H2 in au
    n_dvr = 100
    a = -1.0  # au
    b = 6.0 * ANG2AU  # au

    H2_dvr = build_SineDVR(a, b, n_dvr, m_H2, H2_morse)

    print("H2 vibrational energies (in eV):")
    print(f"n = 0, E_0 = {H2_dvr.eigen_val[0] / EV2AU:.6f} eV")
