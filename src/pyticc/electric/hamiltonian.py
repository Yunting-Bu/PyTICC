import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

from pyticc.basis.dvr import phase_fix
from pyticc.electric.angular import rotor_orientation_moment_matrices


# ----------------------------------------------------------------------------------------
def solve_diatom_electric_block(
    h_reference: NDArray[np.float64],
    grids: NDArray[np.float64],
    electric_coefficients_grid: NDArray[np.float64],
    *,
    m: int,
    jmax: int,
    mass: float,
    n_alpha: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Build and diagonalize one fixed-m monomer Hamiltonian.

    Formula:
        In the direct-product basis |p>|jm>,

        H_{p'j',pj}^{(m)}
          = delta_{j'j} H_{p'p}^{PO}
            + delta_{p'p} delta_{j'j}
              j(j+1)/(2 mu_r r_p^2)
            + delta_{p'p} sum_{n=0}^{3}
              a_n(r_p) X_{j'j}^{(n,m)},

        where

        X_{j'j}^{(n,m)} = <j'm|cos^n(theta)|jm>.

        Diagonalization gives

        sum_{pj} H_{p'j',pj}^{(m)} C_{pj}^{alpha m}
          = E_{alpha m} C_{p'j'}^{alpha m}.

    Inputs:
        h_reference: NDArray[np.float64] - zero-field radial Hamiltonian
            H^(PO) on the PODVR grid, shape (n_podvr, n_podvr)
        grids: NDArray[np.float64] - PODVR bond-length grids, shape (n_podvr,)
        electric_coefficients_grid: NDArray[np.float64] - electric coefficients
            a_n(r_p), shape (4, n_podvr)
        m: int - fixed SF projection
        jmax: int - largest primitive diatomic angular momentum
        mass: float - diatomic reduced mass mu_r in atomic units
        n_alpha: int - number of lowest eigenstates to retain

    Returns:
        j_values: NDArray[np.int64] - retained angular momenta |m|, ..., jmax,
            shape (n_j,)
        energies: NDArray[np.float64] - dressed energies E_{alpha m},
            shape (n_alpha,)
        coefficients: NDArray[np.float64] - eigenvectors C_{pj}^{alpha m},
            shape (n_podvr, n_j, n_alpha)
    """
    j_values = np.arange(abs(m), jmax + 1, dtype=np.int64)
    dimension = grids.size * j_values.size
    hamiltonian = np.kron(h_reference, np.eye(j_values.size))
    centrifugal = j_values * (j_values + 1) / (2.0 * mass * grids[:, None] ** 2)
    hamiltonian[np.diag_indices(dimension)] += centrifugal.ravel()
    for radial_coefficient, orientation_matrix in zip(electric_coefficients_grid, rotor_orientation_moment_matrices(jmax, m), strict=True):
        hamiltonian += np.kron(np.diag(radial_coefficient), orientation_matrix)

    energies, eigenvectors = eigh(hamiltonian, subset_by_index=(0, n_alpha - 1))
    eigenvectors = phase_fix(eigenvectors)
    coefficients = eigenvectors.reshape(grids.size, j_values.size, n_alpha)
    return j_values, energies, coefficients


# ----------------------------------------------------------------------------------------
