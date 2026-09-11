from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyticc.basis.angle import clebsch_gordan_half as cg
from pyticc.fine_structure.atom_atom import FSAtomAtomBasis
from pyticc.pes.spin_resolved_atom_atom import SpinResolvedAtomAtomPES


@dataclass(frozen=True)
class FSAtomAtomVBasis:
    r"""Explicit atomic A and pure-spin Q for BF diabatic contraction.

    Members:
        amplitudes: array - A[c,a,u], normalized positive-K channel amplitudes;
            a=(lambda_X,lambda_Y), u=(mu_X,mu_Y), callback orbital order
        projectors: array - Q[s,u,v], dimensionless total-spin projectors
        two_K: array - twice helicities in channel order

    Formula:
        V[c,d]=delta_(Kc,Kd) sum_(s,a,b,u,v)
        A[c,a,u]* Q[s,u,v] W[s,a,b] A[d,b,v].
        Rotational orthogonality removes unequal signed K. For a scalar,
        parity-conserving interaction the normalized +/-K pair gives exactly
        this positive-K expression, including the surviving K=0 channels.
        Q contains no potential values. A and Q are dimensionless.
    """

    amplitudes: NDArray[np.float64]
    projectors: NDArray[np.float64]
    two_K: NDArray[np.int64]

    def contract(self, values: NDArray) -> NDArray:
        r"""Contract A-dagger Q W A in Hartree.

        Inputs:
            values: array - W[...,s,a,b] in Hartree, arbitrary radial batch
        Returns:
            matrix: array - V[...,c,d] in Hartree, complete channel order
        """
        result = np.einsum("cau,suv,...sab,dbv->...cd", self.amplitudes, self.projectors, values, self.amplitudes, optimize=True)
        return result * (self.two_K[:, None] == self.two_K[None, :])


def prepare_fs_atom_atom_vbasis(basis: FSAtomAtomBasis, pes: SpinResolvedAtomAtomPES) -> FSAtomAtomVBasis:
    r"""Construct atomic recoupling amplitudes and separate spin projectors.

    Formula:
        A[c,a,u]=<L_X lambda_X S_X mu_X|j_X k_X>
          <L_Y lambda_Y S_Y mu_Y|j_Y k_Y>
          <j_X k_X j_Y k_Y|j_12 K>,
        k_i=lambda_i+mu_i and k_X+k_Y=K.
        Q[S,u,v]=sum_M <S_X mu_X S_Y mu_Y|S M>
          <S_X nu_X S_Y nu_Y|S M>.
        All CG coefficients use normalized Condon--Shortley states. Each
        atomic term has fixed L,S; experimental energies do not enter A or Q.

    Inputs:
        basis: FSAtomAtomBasis - parity-adapted channel list
        pes: SpinResolvedAtomAtomPES - orbital and total-spin ordering
    Returns:
        vbasis: FSAtomAtomVBasis - dimensionless contraction arrays
    """
    X, Y = basis.monomer_X, basis.monomer_Y
    pes.validate_basis(X.two_L, X.two_S, Y.two_L, Y.two_S)
    spins = [(x, y) for x in range(-X.two_S, X.two_S + 1, 2) for y in range(-Y.two_S, Y.two_S + 1, 2)]
    amplitudes = np.zeros((basis.n_channel, len(pes.orbital_states), len(spins)))
    for c, channel in enumerate(basis):
        jx, jy = X.two_j_levels[channel.level_X], Y.two_j_levels[channel.level_Y]
        for a, orbital in enumerate(pes.orbital_states):
            lx, ly = orbital.two_lambda_X, orbital.two_lambda_Y
            for u, (mx, my) in enumerate(spins):
                kx, ky = lx + mx, ly + my
                if kx + ky != channel.two_K or abs(kx) > jx or abs(ky) > jy:
                    continue
                amplitudes[c, a, u] = cg(X.two_L, lx, X.two_S, mx, jx) * cg(Y.two_L, ly, Y.two_S, my, jy) * cg(jx, kx, jy, ky, channel.two_j12)
    projectors = np.zeros((len(pes.two_total_spins), len(spins), len(spins)))
    for s, total in enumerate(pes.two_total_spins):
        for u, (mx, my) in enumerate(spins):
            for v, (nx, ny) in enumerate(spins):
                if mx + my == nx + ny:
                    projectors[s, u, v] = cg(X.two_S, mx, Y.two_S, my, total) * cg(X.two_S, nx, Y.two_S, ny, total)
    return FSAtomAtomVBasis(amplitudes, projectors, np.asarray([c.two_K for c in basis], dtype=np.int64))


def magnetic_dipole_matrix(basis: FSAtomAtomBasis, vbasis: FSAtomAtomVBasis) -> NDArray[np.float64]:
    r"""Return the spin-only magnetic dipole angular matrix in BF channels.

    Formula:
        D=S_X dot S_Y-3 S_Xz S_Yz
         =-2 S_Xz S_Yz+(S_X+ S_Y- + S_X- S_Y+)/2.
        V_dd(R)=C_dd D/R^3, with dimensionless spin operators (hbar=1).
        The orbital operator is identity. C_dd is supplied separately in
        Hartree bohr^3; no electric or orbital magnetic moment is included.

    Inputs:
        basis: FSAtomAtomBasis - two fixed-spin atoms
        vbasis: FSAtomAtomVBasis - normalized recoupling amplitudes
    Returns:
        matrix: array - dimensionless D[c,d] in channel order
    """
    sx, sy = basis.monomer_X.two_S, basis.monomer_Y.two_S
    spins = [(x, y) for x in range(-sx, sx + 1, 2) for y in range(-sy, sy + 1, 2)]
    lookup = {state: i for i, state in enumerate(spins)}
    D = np.zeros((len(spins), len(spins)))
    for u, (x, y) in enumerate(spins):
        D[u, u] = -x * y / 2
        for sign in (-1, 1):
            v = lookup.get((x + 2 * sign, y - 2 * sign))
            if v is not None:
                D[v, u] = np.sqrt((sx / 2 * (sx / 2 + 1) - x / 2 * (x / 2 + sign)) * (sy / 2 * (sy / 2 + 1) - y / 2 * (y / 2 - sign))) / 2
    result = np.einsum("cau,uv,dav->cd", vbasis.amplitudes, D, vbasis.amplitudes, optimize=True)
    return result * (vbasis.two_K[:, None] == vbasis.two_K[None, :])
