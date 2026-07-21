from collections.abc import Sequence

import numpy as np
from loguru import logger
from numpy.typing import NDArray

import pyticc.matrix.interaction.diabatic_atom_diatom as vmat
from pyticc.basis.angle import gauss_legendre_dvr
from pyticc.basis.channel import ChannelBuilder, TruncSpec
from pyticc.basis.monomer.atom import AtomSpec
from pyticc.basis.monomer.diabatic import DiabaticDiatomBasis
from pyticc.pes.diabatic import DiabaticPESWrapper
from pyticc.scattering.hamiltonian import ScattHamiltonian
from pyticc.system import Approx, ScattSystem


def build_hamiltonian(
    system: ScattSystem,
    *,
    trunc: TruncSpec | None = None,
    n_theta: int = 16,
) -> ScattHamiltonian:
    """Build a diabatic atom-diatom scattering Hamiltonian."""
    if not isinstance(system.monomer_X, AtomSpec) or not isinstance(system.monomer_Y, DiabaticDiatomBasis):
        message = "Diabatic atom-diatom Hamiltonian requires AtomSpec and DiabaticDiatomBasis monomers"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, DiabaticPESWrapper):
        message = "Diabatic atom-diatom Hamiltonian requires a DiabaticPESWrapper"
        logger.error(message)
        raise TypeError(message)
    if system.approx is not Approx.EXACT:
        message = "Diabatic atom-diatom Hamiltonian currently requires approx='exact'"
        logger.error(message)
        raise ValueError(message)

    diatom = system.monomer_Y
    pes = system.potential
    if pes.n_state != diatom.n_state:
        message = f"PES has {pes.n_state} electronic states, but the diatomic basis has {diatom.n_state}"
        logger.error(message)
        raise ValueError(message)
    if n_theta < 1:
        message = f"n_theta must be positive, but got {n_theta}"
        logger.error(message)
        raise ValueError(message)

    basis = ChannelBuilder(system, TruncSpec() if trunc is None else trunc).build()
    angular_symmetry = all(jpar != 0 for jpar in diatom.rotational_parities)
    cos_theta, theta_weights = gauss_legendre_dvr(-1.0, 1.0, n_theta, symmetry=angular_symmetry)
    V_basis = vmat.prepare(basis, diatom, cos_theta, theta_weights)

    def Vmat(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """Sample and contract the diabatic potential."""
        potential_grid = vmat.sample(pes, radial_points, V_basis)
        return vmat.contract(V_basis, potential_grid)

    potential_grid_size = V_basis.theta.size * (sum(grid.size for grid in V_basis.diagonal_grids) + V_basis.coupling_grid.size) * diatom.n_state**2
    return ScattHamiltonian(
        system=system,
        basis=basis,
        interaction=Vmat,
        potential_grid_size=potential_grid_size,
    )
