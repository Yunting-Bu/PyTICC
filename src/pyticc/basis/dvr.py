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
        x: NDArray[np.float64] - sine dvr grids
        w: float - weights
    """
    if n < 2:
        logger.warning(f"n should be greater than 1, but got n = {n}")

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
        B: NDArray[np.float64] - dvr to fbr transformation matrix
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
        T: NDArray[np.float64] - sine dvr kinetic energy matrix
    """
    T = np.zeros((n, n), dtype=np.float64)
    pre_factor = (np.pi**2) / (4.0 * mass * (b - a) ** 2)

    for i0 in range(n):
        i = i0 + 1

        diag_term = (2.0 * (n + 1) ** 2 + 1.0) / 3.0 - np.sin(np.pi * i / (n + 1)) ** (-2)
        T[i0, i0] = pre_factor * diag_term

        for j0 in range(i0):
            j = j0 + 1
            off_term = np.sin(np.pi * (i - j) / (2.0 * (n + 1))) ** (-2) - np.sin(np.pi * (i + j) / (2.0 * (n + 1))) ** (-2)
            phase_factor = (-1.0) ** (i - j)

            T[i0, j0] = pre_factor * off_term * phase_factor
            T[j0, i0] = T[i0, j0]

    return T


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
def phase_fix(A: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Flip the overall sign so the first significant element is positive.

    Inputs:
        A: NDArray[np.float64] - 1D/2D input array

    Returns:
        A: NDArray[np.float64] - 1D/2D output array with fixed phase
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
        logger.error("All elements are below the threshold.")
        raise ValueError("All elements are below the threshold.")

    if res.ndim == 2:
        for col in range(res.shape[1]):
            res[:, col] = phase_fix(res[:, col])
        return res

    logger.warning("Input array must be 1D or 2D, returning the original array.")
    return res


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
def sine_dvr_vib(T: NDArray[np.float64], V: NDArray[np.float64], n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Get the eigenvalues and eigenvectors of the vibrational Hamiltonian in sine DVR.
    Formula:
        H_{ij} = T_{ij} + V_i \delta_{ij}

    Inputs:
        T: NDArray[np.float64] - kinetic energy matrix
        V: NDArray[np.float64] - potential energy in dvr grids
        n: int - number of points

    Returns:
        eigen_val: NDArray[np.float64] - eigenvalues of the vibrational Hamiltonian
        eigen_vec: NDArray[np.float64] - eigenvectors of the vibrational Hamiltonian
    """
    H = T.copy()
    diag_indices = np.diag_indices(n)
    H[diag_indices] += V

    eigen_val, eigen_vec = eigh(H)
    eigen_vec = phase_fix(eigen_vec)

    return eigen_val, eigen_vec


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
            x: NDArray[np.float64] - position in atomic unit

        Returns:
            V: NDArray[np.float64] - potential energy in atomic unit
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

    grids, _ = sine_dvr_grids(a, b, n_dvr)
    T_mat = sine_dvr_kinetic(a, b, n_dvr, m_H2)
    V_mat = H2_morse(grids)
    E_vib, _ = sine_dvr_vib(T_mat, V_mat, n_dvr)
    print("H2 vibrational energies (in eV):")
    print(f"n = 0, E_0 = {E_vib[0] / EV2AU:.6f} eV")
