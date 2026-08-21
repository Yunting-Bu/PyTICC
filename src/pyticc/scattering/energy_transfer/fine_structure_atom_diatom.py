from math import prod

import numpy as np
from loguru import logger

import pyticc.matrix.interaction.fs_atom_diatom as vmat
from pyticc.basis.angle import gauss_legendre_dvr
from pyticc.basis.monomer import AtomSpec
from pyticc.fine_structure.channel import FSChannelBasis, FSMonomerBasis
from pyticc.pes.lambda_pes import LambdaPES, RadialInput, get_lambda_grid_atom_diatom
from pyticc.scattering.hamiltonian import ScattHamiltonian
from pyticc.system import Approx, ScattSystem


def build_hamiltonian(
    system: ScattSystem,
    *,
    n_theta: int = 24,
) -> ScattHamiltonian:
    """
    Build an exact BF atom-diatom Hamiltonian in the FS basis.

    Inputs:
        system: ScattSystem - atom plus fine-structure diatom system containing
            prepared channels, LambdaPES, and collision reduced mass
        n_theta: int - full Gauss-Legendre angular order

    Returns:
        hamiltonian: ScattHamiltonian - exact coupled-channel Hamiltonian
    """
    if not isinstance(system.monomer_X, AtomSpec) or not isinstance(system.monomer_Y, FSMonomerBasis):
        message = "Fine-structure atom-diatom Hamiltonian requires AtomSpec and FSMonomerBasis monomers"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, LambdaPES):
        message = "Fine-structure atom-diatom Hamiltonian requires a LambdaPES"
        logger.error(message)
        raise TypeError(message)
    if system.reduced_mass is None:
        message = "Fine-structure atom-diatom Hamiltonian requires a collision reduced mass"
        logger.error(message)
        raise ValueError(message)
    if system.approx is not Approx.EXACT:
        message = "Fine-structure atom-diatom Hamiltonian currently requires approx='exact'"
        logger.error(message)
        raise ValueError(message)
    if not isinstance(system.basis, FSChannelBasis):
        message = "Fine-structure atom-diatom Hamiltonian requires channels prepared by build_ScattSystem"
        logger.error(message)
        raise TypeError(message)

    basis = system.basis
    potential = system.potential
    cos_theta, weights = gauss_legendre_dvr(-1.0, 1.0, n_theta)
    theta = np.arccos(cos_theta)
    V_basis = vmat.prepare(basis, theta, weights)

    def Vgrid(radial_points: RadialInput):
        return get_lambda_grid_atom_diatom(potential, radial_points, basis.monomer.vib.grids, theta)

    def Vmat(radial_points: RadialInput):
        return vmat.contract(V_basis, Vgrid(radial_points))

    return ScattHamiltonian(
        basis=basis,
        reduced_mass=system.reduced_mass,
        interaction=Vmat,
        approx=Approx.EXACT,
        potential_grid_size=prod(V_basis.grid_shape),
    )
