import math
from collections.abc import Sequence
from math import prod

import numpy as np
from loguru import logger
from numpy.typing import NDArray

import pyticc.matrix.interaction.atom_diatom as vmat
from pyticc.basis.angle import gauss_legendre_dvr
from pyticc.basis.channel import ChannelBuilder, TruncSpec, build_ChannelBasisElectricSF
from pyticc.basis.monomer import AtomSpec, DiatomBasis, DiatomElectricBasis
from pyticc.matrix.interaction import contract
from pyticc.pes.adiabatic import PESWrapper, RadialInput, get_Vgrid_atom_diatom, get_Vgrid_atom_diatom_electric_sf
from pyticc.scattering.hamiltonian import ScattHamiltonian
from pyticc.system import ScattSystem


# ----------------------------------------------------------------------------------------
def build_hamiltonian(
    system: ScattSystem,
    *,
    trunc: TruncSpec | None = None,
    n_theta: int = 16,
) -> ScattHamiltonian:
    """Build an adiabatic atom-diatom scattering Hamiltonian.

    Inputs:
        system: ScattSystem - atom-diatom system with a scalar PES
        trunc: TruncSpec | None - channel-energy and helicity truncations
        n_theta: int - retained Jacobi-angle quadrature points

    Returns:
        hamiltonian: ScattHamiltonian - projected body-fixed Hamiltonian
    """
    if not isinstance(system.monomer_X, AtomSpec) or not isinstance(system.monomer_Y, DiatomBasis):
        message = "Atom-diatom Hamiltonian requires AtomSpec as monomer_X and DiatomBasis as monomer_Y"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, PESWrapper):
        message = "Adiabatic atom-diatom Hamiltonian requires a PESWrapper"
        logger.error(message)
        raise TypeError(message)

    diatom = system.monomer_Y
    rovib = diatom.rovib
    pes = system.potential
    basis = ChannelBuilder(system, TruncSpec() if trunc is None else trunc).build()
    cos_theta, theta_weights = gauss_legendre_dvr(-1.0, 1.0, n_theta, symmetry=diatom.jpar != 0)
    theta = np.arccos(cos_theta)
    V_basis = vmat.prepare(basis, rovib, cos_theta, theta_weights)

    def Vgrid(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the atom-diatom PES grid."""
        return get_Vgrid_atom_diatom(pes, radial_points, rovib.grids, theta)

    def Vmat(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """Contract the PES grid into the channel basis."""
        return contract(V_basis, Vgrid(radial_points))

    def V_blocks(
        radial_points: NDArray[np.float64],
        channel_blocks: tuple[tuple[int, ...], ...],
    ) -> tuple[NDArray[np.float64], ...]:
        """Contract one shared PES grid into several channel blocks."""
        potential_grid = Vgrid(radial_points)
        return tuple(contract(V_basis, potential_grid, indices) for indices in channel_blocks)

    return ScattHamiltonian(
        system=system,
        basis=basis,
        interaction=Vmat,
        block_interaction=V_blocks,
        potential_grid_size=prod(V_basis.grid_shape),
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_hamiltonian_electric_sf(
    system: ScattSystem,
    *,
    lmax: int,
    E_cut: float = math.inf,
    n_theta_r: int = 16,
    n_theta_R: int = 16,
    n_delta: int = 16,
    delta_symmetry: bool = True,
) -> ScattHamiltonian:
    r"""
    Build an electric-field atom-diatom Hamiltonian in the space-fixed representation.

    Formula:
        Channels eta = (alpha,m,l,m_l) are generated from

        m = M-m_l,    l = 0,...,lmax,    m_l = -l,...,l,

        and retained when epsilon_{alpha m} <= E_cut. The three angular
        integrations use Gauss-Legendre rules in

        x_r = cos(theta_r) in [-1,1],
        x_R = cos(theta_R) in [-1,1],
        delta = phi_r-phi_R in [0,2 pi].

        The interaction callback evaluates Delta V(R,r,gamma) and contracts it
        into V^M(R). The resulting Hamiltonian is

        H(R) = diag(E_int) + U_SF/(2 mu_R R^2) + V^M(R).

        When delta_symmetry is True, n_delta nodes are retained from the lower
        half of a 2*n_delta rule and paired weights are doubled. This is exact
        for a scalar PES depending on delta only through cos(gamma).

    Inputs:
        system: ScattSystem - atom-diatom system containing AtomSpec,
            DiatomElectricBasis, M, interaction PES, and collision reduced mass
        lmax: int - largest retained end-over-end angular momentum
        E_cut: float - largest retained relative monomer threshold in atomic
            units
        n_theta_r: int - Gauss-Legendre order for cos(theta_r)
        n_theta_R: int - Gauss-Legendre order for cos(theta_R)
        n_delta: int - retained relative-azimuth quadrature points
        delta_symmetry: bool - whether to use the paired half-interval rule

    Returns:
        hamiltonian: ScattHamiltonian - electric-field fixed-M SF Hamiltonian
    """
    if not isinstance(system.monomer_X, AtomSpec) or not isinstance(system.monomer_Y, DiatomElectricBasis):
        message = "Electric atom-diatom Hamiltonian requires AtomSpec as monomer_X and DiatomElectricBasis as monomer_Y"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, PESWrapper):
        message = "Electric atom-diatom Hamiltonian requires a PESWrapper"
        logger.error(message)
        raise TypeError(message)
    if system.M is None:
        message = "Electric atom-diatom Hamiltonian requires M"
        logger.error(message)
        raise ValueError(message)

    M = system.M
    monomer_basis = system.monomer_Y
    pes = system.potential
    basis = build_ChannelBasisElectricSF(monomer_basis, M=M, lmax=lmax, E_cut=E_cut)
    cos_theta_r, theta_r_weights = gauss_legendre_dvr(-1.0, 1.0, n_theta_r)
    cos_theta_R, theta_R_weights = gauss_legendre_dvr(-1.0, 1.0, n_theta_R)
    delta, delta_weights = gauss_legendre_dvr(0.0, 2.0 * np.pi, n_delta, symmetry=delta_symmetry)
    V_basis = vmat.build_AtomDiatomVBasisElectricSF(
        basis,
        monomer_basis,
        cos_theta_r,
        theta_r_weights,
        cos_theta_R,
        theta_R_weights,
        delta,
        delta_weights,
    )

    def Vgrid(radial_points: RadialInput) -> NDArray[np.float64]:
        """Evaluate the atom-diatom PES on the prepared SF geometry grid."""
        return get_Vgrid_atom_diatom_electric_sf(pes, radial_points, V_basis.r, V_basis.gamma)

    def Vmat(radial_points: RadialInput) -> NDArray[np.float64]:
        """Contract the SF PES grid into the fixed-M channel basis."""
        return vmat.contract_electric_sf(V_basis, Vgrid(radial_points))

    return ScattHamiltonian(
        system=system,
        basis=basis,
        interaction=Vmat,
        potential_grid_size=prod(V_basis.grid_shape),
    )


# ----------------------------------------------------------------------------------------
