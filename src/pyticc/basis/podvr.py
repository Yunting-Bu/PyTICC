import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.linalg import eigh

from pyticc.basis.dvr import phase_fix


# --------------------------------------------------------------------------------
def podvr_grids(
    dvr_grids: NDArray[np.float64], dvr_wf: NDArray[np.float64], n_podvr: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Get podvr grids and transformation matrix from dvr to contracted basis.

    Inputs:
        dvr_grids: NDArray[np.float64] - sine-DVR grids
        dvr_wf: NDArray[np.float64] - wavefunctions (eigenvectors) of the reference vibrational
            Hamiltonian in sine-DVR, shape (n_dvr, n_dvr). Only the first ``n_podvr`` columns are used.
        n_podvr: int - number of contracted (PODVR) basis functions to retain

    Returns:
        po_grids: NDArray[np.float64] - PODVR grid points, shape (n_podvr,)
        dvr_to_cfbr: NDArray[np.float64] - sine-DVR -> contracted-FBR transform, shape (n_dvr, n_podvr)
        po_to_cfbr: NDArray[np.float64] - PODVR -> contracted-FBR transform, shape (n_podvr, n_podvr)
    """
    if n_podvr > dvr_wf.shape[1]:
        logger.warning(f"n_podvr = {n_podvr} exceeds available dvr_wf columns ({dvr_wf.shape[1]}); truncated.")

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
if __name__ == "__main__":
    import numpy as np
    from numpy.typing import NDArray

    from pyticc.basis.dvr import sine_dvr_grids, sine_dvr_kinetic, sine_dvr_vib
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
    n_podvr = 4
    v_max = 2
    j_max = 5
    a = -1.0  # au
    b = 6.0 * ANG2AU  # au

    grids, _ = sine_dvr_grids(a, b, n_dvr)
    T_mat = sine_dvr_kinetic(a, b, n_dvr, m_H2)
    V_mat = H2_morse(grids)
    E_vib, WF_vib = sine_dvr_vib(T_mat, V_mat, n_dvr)

    po_grids, dvr_to_c, po_to_c = podvr_grids(grids, WF_vib, n_podvr)
    E_v, WF_v = podvr_vib(po_to_c, E_vib, v_max)
    E_vr, WF_vr = podvr_vibrot(po_grids, po_to_c, E_vib, v_max, j_max, m_H2)

    print("H2 PODVR v=0 rotational energies (in cm^-1, relative to E(v=0,j=0)):")
    E0 = E_vr[0, 0]
    exp_E = {1: 118.495, 2: 354.376, 3: 705.524, 4: 1168.815, 5: 1741.119}  # cm^-1, v=0
    AU2CM = 219474.6313705
    print(f"{'j':>3} {'PODVR/cm^-1':>16} {'Exp/cm^-1':>16}")
    for j in range(1, j_max + 1):
        E_rel = (E_vr[0, j] - E0) * AU2CM
        print(f"{j:>3} {E_rel:>16.3f} {exp_E[j]:>16.3f}")
