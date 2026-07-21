from collections.abc import Sequence
from math import prod

import numpy as np
from loguru import logger
from numpy.typing import NDArray

import pyticc.matrix.interaction.atom_diatom as vmat
from pyticc.basis.angle import gauss_legendre_dvr
from pyticc.basis.channel import ChannelBuilder, TruncSpec
from pyticc.basis.monomer import AtomSpec, DiatomBasis
from pyticc.matrix.interaction import contract
from pyticc.pes.wrapper import PESWrapper, get_Vgrid_atom_diatom
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
