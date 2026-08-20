from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.linalg import eigh

from pyticc.basis.dvr import SineDVR, phase_fix
from pyticc.basis.rovib import RovibBasis


# PODVR contraction
# --------------------------------------------------------------------------------
def podvr_grids(
    dvr_grids: NDArray[np.float64], dvr_wf: NDArray[np.float64], n_podvr: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Get podvr grids and transformation matrix from dvr to contracted basis.

    Inputs:
        dvr_grids: NDArray[np.float64] - sine-DVR grids, shape (n_dvr,)
        dvr_wf: NDArray[np.float64] - wavefunctions (eigenvectors) of the reference vibrational
            Hamiltonian in sine-DVR, shape (n_dvr, n_dvr). Only the first ``n_podvr`` columns are used.
        n_podvr: int - number of contracted (PODVR) basis functions to retain

    Returns:
        po_grids: NDArray[np.float64] - PODVR grid points, shape (n_podvr,)
        dvr_to_cfbr: NDArray[np.float64] - sine-DVR -> contracted-FBR transform, shape (n_dvr, n_podvr)
        po_to_cfbr: NDArray[np.float64] - PODVR -> contracted-FBR transform, shape (n_podvr, n_podvr)
    """
    if n_podvr > dvr_wf.shape[1]:
        message = f"n_podvr = {n_podvr} exceeds available dvr_wf columns ({dvr_wf.shape[1]}); truncated."
        logger.warning(message)

    n_podvr = min(n_podvr, dvr_wf.shape[1])
    dvr_to_cfbr = dvr_wf[:, :n_podvr].copy()
    # coordinate matrix <v|r|v'> in the contracted FBR basis
    x_mat = np.einsum("ia,i,ib->ab", dvr_to_cfbr, dvr_grids, dvr_to_cfbr, optimize=True)
    po_grids, po_to_cfbr = eigh(x_mat)
    po_to_cfbr = phase_fix(po_to_cfbr)

    return po_grids, dvr_to_cfbr, po_to_cfbr


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
def podvr_vib(po_to_cfbr: NDArray[np.float64], dvr_eigen: NDArray[np.float64], vmax: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Get reference vibrational energies and contracted-basis wavefunctions on the PODVR grid.

    ``dvr_eigen`` is truncated inside to the first ``n_podvr`` entries, consistent with
    the contracted basis retained by :func:`podvr_grids`. The reference Hamiltonian, diagonal in
    the contracted-FBR basis, is transformed to the PODVR representation and diagonalized; this
    simply recovers the same eigenpairs and selects the lowest ``vmax + 1``.

    Inputs:
        po_to_cfbr: NDArray[np.float64] - PODVR -> contracted-FBR transform, shape (n_podvr, n_podvr)
        dvr_eigen: NDArray[np.float64] - eigenvalues of the 1D reference Hamiltonian from the
            full sine-DVR diagonalization, ordered ascendingly, shape (n_dvr,)
        vmax: int - highest vibrational quantum number to retain (returns v = 0 .. vmax)

    Returns:
        po_Evib: NDArray[np.float64] - vibrational energies, shape (vmax + 1,)
        po_WFvib: NDArray[np.float64] - vibrational wavefunctions on PODVR grids, shape (n_podvr, vmax + 1)
    """
    n_podvr = po_to_cfbr.shape[0]
    ref_eigen = dvr_eigen[:n_podvr]
    n_keep = min(vmax + 1, n_podvr)

    H_ref = np.einsum("ia,i,ib->ab", po_to_cfbr, ref_eigen, po_to_cfbr, optimize=True)
    eigen_val, eigen_vec = eigh(H_ref)
    eigen_vec = phase_fix(eigen_vec)

    po_Evib = np.zeros(vmax + 1, dtype=np.float64)
    po_WFvib = np.zeros((n_podvr, vmax + 1), dtype=np.float64)
    po_Evib[:n_keep] = eigen_val[:n_keep]
    po_WFvib[:, :n_keep] = eigen_vec[:, :n_keep]
    return po_Evib, po_WFvib


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
def podvr_vibrot(
    po_grids: NDArray[np.float64],
    po_to_cfbr: NDArray[np.float64],
    dvr_eigen: NDArray[np.float64],
    vmax: int,
    jmax: int,
    mass: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Get reference rovibrational energies and contracted-basis wavefunctions on the PODVR grid.

    ``dvr_eigen`` is truncated inside to the first ``n_podvr`` entries, consistent with the
    contracted basis retained by :func:`podvr_grids`. For each rotational quantum number
    j = 0 .. jmax, the rigid-rotor centrifugal potential ``j(j+1)/(2*mu*r^2)`` is added *in the
    PODVR representation* (where r is diagonal) to the reference Hamiltonian expressed in PODVR
    representation. The resulting matrix is diagonalized to yield a j-dependent contracted basis.
    Only the lowest ``vmax + 1`` states are returned.

    Inputs:
        po_grids: NDArray[np.float64] - PODVR grid points, shape (n_podvr,)
        po_to_cfbr: NDArray[np.float64] - PODVR -> contracted-FBR transform, shape (n_podvr, n_podvr)
        dvr_eigen: NDArray[np.float64] - eigenvalues of the 1D reference Hamiltonian from the
            full sine-DVR diagonalization, ordered ascendingly, shape (n_dvr,)
        vmax: int - highest vibrational quantum number to retain
        jmax: int - highest rotational quantum number to retain
        mass: float - reduced mass of the diatomic (atomic units)

    Returns:
        po_Evibrot: NDArray[np.float64] - rovibrational energies, shape (vmax + 1, jmax + 1)
        po_WFvibrot: NDArray[np.float64] - rovibrational wavefunctions on PODVR grids, shape (n_podvr, vmax + 1, jmax + 1)
    """
    n_podvr = po_grids.shape[0]
    ref_eigen = dvr_eigen[:n_podvr]
    n_v = min(vmax + 1, n_podvr)
    po_Evibrot = np.zeros((vmax + 1, jmax + 1), dtype=np.float64)
    po_WFvibrot = np.zeros((n_podvr, vmax + 1, jmax + 1), dtype=np.float64)

    # Reference Hamiltonian in the PODVR representation (common to all j).
    H_ref = np.einsum("ia,i,ib->ab", po_to_cfbr, ref_eigen, po_to_cfbr, optimize=True)
    diag_idx = np.diag_indices(n_podvr)

    for j in range(jmax + 1):
        H = H_ref.copy()
        H[diag_idx] += j * (j + 1) / (2.0 * mass * po_grids**2)
        eigen_val, eigen_vec = eigh(H)
        eigen_vec = phase_fix(eigen_vec)

        po_Evibrot[:n_v, j] = eigen_val[:n_v]
        po_WFvibrot[:, :n_v, j] = eigen_vec[:, :n_v]

    return po_Evibrot, po_WFvibrot


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
@dataclass(frozen=True)
class VibPODVR:
    """
    Contracted one-dimensional vibrational basis on PODVR grids.

    Members:
        grids: NDArray[np.float64] - PODVR coordinate grids in atomic units, shape
            (n_podvr,)
        energies: NDArray[np.float64] - reference vibrational energies indexed by v,
            shape (n_v,)
        wavefunctions: NDArray[np.float64] - wavefunctions indexed as
            wavefunctions[grid, v], shape (n_podvr, n_v)
    """

    grids: NDArray[np.float64]
    energies: NDArray[np.float64]
    wavefunctions: NDArray[np.float64]


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
def build_VibPODVR(dvr: SineDVR, n_podvr: int, vmax: int) -> VibPODVR:
    """
    Build a contracted one-dimensional vibrational basis from a sine-DVR calculation.

    Inputs:
        dvr: SineDVR - reference vibrational sine-DVR basis
        n_podvr: int - number of PODVR grids to retain
        vmax: int - highest vibrational quantum number

    Returns:
        vib: VibPODVR - PODVR grids, energies, and wavefunctions
    """
    po_grids, _, po_to_cfbr = podvr_grids(dvr.grids, dvr.eigen_vec, n_podvr)
    energies, wavefunctions = podvr_vib(po_to_cfbr, dvr.eigen_val, vmax)
    return VibPODVR(grids=po_grids, energies=energies, wavefunctions=wavefunctions)


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
def build_RovibPODVR(dvr: SineDVR, n_podvr: int, vmax: int, jmax: int, mass: float) -> RovibBasis:
    """
    Build a contracted diatomic rovibrational basis from a sine-DVR calculation.

    Inputs:
        dvr: SineDVR - reference vibrational sine-DVR basis
        n_podvr: int - number of PODVR grids to retain
        vmax: int - highest vibrational quantum number
        jmax: int - highest rotational quantum number
        mass: float - diatomic reduced mass in atomic units

    Returns:
        rovib: RovibBasis - PODVR grids, rovibrational energies, and wavefunctions
    """
    po_grids, _, po_to_cfbr = podvr_grids(dvr.grids, dvr.eigen_vec, n_podvr)
    E_vj, WF_vj = podvr_vibrot(po_grids, po_to_cfbr, dvr.eigen_val, vmax, jmax, mass)
    return RovibBasis(grids=po_grids, E_vj=E_vj, WF_vj=WF_vj)
