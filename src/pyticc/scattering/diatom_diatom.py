from collections.abc import Sequence
from math import prod

import numpy as np
from loguru import logger
from numpy.typing import NDArray

import pyticc.matrix.interaction.diatom_diatom as vmat
from pyticc.basis.angle import gauss_legendre_dvr
from pyticc.basis.channel import ChannelBuilder, TruncSpec
from pyticc.basis.monomer import DiatomBasis
from pyticc.matrix.interaction import contract
from pyticc.pes.wrapper import PESWrapper, get_Vgrid_diatom_diatom
from pyticc.scattering.hamiltonian import ScattHamiltonian
from pyticc.system import ScattSystem


# ----------------------------------------------------------------------------------------
def build_hamiltonian(
    system: ScattSystem,
    *,
    trunc: TruncSpec | None = None,
    n_theta_X: int = 15,
    n_theta_Y: int = 15,
    n_phi: int = 12,
) -> ScattHamiltonian:
    """Build a diatom-diatom scattering Hamiltonian."""
    if not isinstance(system.monomer_X, DiatomBasis) or not isinstance(system.monomer_Y, DiatomBasis):
        message = "Diatom-diatom Hamiltonian requires two DiatomBasis monomers"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, PESWrapper):
        message = "Diatom-diatom Hamiltonian requires a PESWrapper"
        logger.error(message)
        raise TypeError(message)

    pes = system.potential
    rovib_X = system.monomer_X.rovib
    rovib_Y = system.monomer_Y.rovib
    basis = ChannelBuilder(system, TruncSpec() if trunc is None else trunc).build()
    cos_theta_X, theta_weights_X = gauss_legendre_dvr(-1.0, 1.0, n_theta_X)
    cos_theta_Y, theta_weights_Y = gauss_legendre_dvr(-1.0, 1.0, n_theta_Y)
    phi, phi_weights = gauss_legendre_dvr(0.0, np.pi, n_phi)
    theta_X = np.arccos(cos_theta_X)
    theta_Y = np.arccos(cos_theta_Y)
    V_basis = vmat.prepare(
        basis,
        rovib_X,
        rovib_Y,
        cos_theta_X,
        theta_weights_X,
        cos_theta_Y,
        theta_weights_Y,
        phi,
        phi_weights,
    )

    def Vgrid(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the diatom-diatom PES grid."""
        return get_Vgrid_diatom_diatom(pes, radial_points, rovib_X.grids, rovib_Y.grids, theta_X, theta_Y, phi)

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
