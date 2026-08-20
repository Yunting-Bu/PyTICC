from loguru import logger

from pyticc.basis.delves import DelvesBasis
from pyticc.basis.monomer.delves import DelvesMonomer
from pyticc.pes.total import TotalPES
from pyticc.scattering.delves_hamiltonian import DelvesHamiltonian
from pyticc.system import ScattSystem


def build_hamiltonian(
    system: ScattSystem,
    *,
    overlap_cut: float = 1.0e-4,
) -> DelvesHamiltonian:
    """
    Build a Delves Hamiltonian from a system with preselected ABC channels.

    Inputs:
        system: ScattSystem - common scattering system containing a prepared
            Delves monomer, channel basis, and native total PES
        overlap_cut: float - canonical channel-overlap eigenvalue cutoff

    Returns:
        hamiltonian: DelvesHamiltonian - reactive Hamiltonian accepted by the
            common ``solve`` entry point
    """
    if not isinstance(system, ScattSystem):
        message = "Reactive atom-diatom Hamiltonian requires a ScattSystem"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.monomer_X, DelvesMonomer) or system.monomer_Y is not None:
        message = "Reactive atom-diatom Hamiltonian requires a prepared Delves monomer"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.basis, DelvesBasis):
        message = "Reactive atom-diatom Hamiltonian requires Delves channels prepared by build_ScattSystem"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.total_potential, TotalPES):
        message = "Reactive atom-diatom Hamiltonian requires a TotalPES"
        logger.error(message)
        raise TypeError(message)
    basis = system.basis
    native_total_potential = system.total_potential
    total_potential = native_total_potential
    if basis.energy_zero != 0.0:
        total_potential = TotalPES(lambda bonds: native_total_potential(bonds) - basis.energy_zero)
    return DelvesHamiltonian(
        basis=basis,
        total_potential=total_potential,
        energy_zero=basis.energy_zero,
        overlap_cut=overlap_cut,
    )


# ----------------------------------------------------------------------------------------
